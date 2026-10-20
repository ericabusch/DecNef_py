"""Ordered compute/commit engine (docs/architecture.md).

Compute (DICOM -> NIfTI) runs in a thread pool, in any order. Commit (every
stateful stage: unwarp, MC, FD/DVARS, regression, transform, smooth, score)
runs strictly in scan order, one scan at a time, in whichever worker
finished last. A scan whose compute never arrives is skipped after
`commit_wait_timeout_s` once a newer scan is ready; late results for an
already-committed scan are dropped. This is the original `parallel_ordered`
engine; the `legacy` engine was removed.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)
COMPUTE_RETRY_SLEEP_S = 0.2


@dataclass
class ResultEnvelope:
    scan: int
    volume_idx: int
    dicom_path: Path
    volume_timestamp: float
    success: bool
    raw_nii: Optional[Path] = None
    error: Optional[str] = None
    traceback_text: Optional[str] = None
    attempts: int = 0


class OrderedEngine:
    """`compute(path, scan, volume_idx) -> ResultEnvelope` runs in the pool;
    `commit(envelope)` runs in scan order. `commit` must not raise for a failed
    envelope (it decides how to record the failure)."""

    def __init__(self, compute: Callable[[Path, int, int], ResultEnvelope], commit: Callable[[ResultEnvelope], None],
                 max_workers: int = 6, commit_wait_timeout_s: float = 1.0, max_retries: int = 3):
        self._compute = compute
        self._commit = commit
        self.max_workers = max(1, min(int(max_workers), 10))
        self.commit_wait_timeout_s = float(commit_wait_timeout_s)
        self.max_retries = max(1, int(max_retries))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._pending_known: set[int] = set()
        self._result_buffer: dict[int, ResultEnvelope] = {}
        self._scan_first_seen: dict[int, float] = {}
        self._next_scan_to_commit: Optional[int] = None
        self.last_committed_scan: int = 0
        self._timed_out: set[int] = set()
        self.committed: int = 0
        self.failed: int = 0

    # ---- submission
    def submit(self, path: Path, scan: int) -> None:
        with self._lock:
            if scan in self._inflight or scan <= self.last_committed_scan:
                return
            self._inflight.add(scan)
            self._pending_known.discard(scan)
            self._scan_first_seen.setdefault(scan, time.time())
            if self._next_scan_to_commit is None:
                self._advance_locked()      # honours scans the watcher has seen but not yet submitted
        fut = self._executor.submit(self._compute_with_retries, path, scan)
        fut.add_done_callback(self._on_compute_done)

    def note_pending(self, scans: set[int]) -> None:
        """Scans queued by the watcher but not yet submitted (keeps the commit order waiting for them)."""
        now = time.time()
        with self._lock:
            self._pending_known = set(scans)
            for s in scans:                     # the commit-wait timeout counts from when a scan was first seen
                self._scan_first_seen.setdefault(s, now)

    def _compute_with_retries(self, path: Path, scan: int) -> ResultEnvelope:
        err_txt = tb_txt = None
        for attempt in range(1, self.max_retries + 1):
            try:
                env = self._compute(path, scan, scan)
                env.attempts = attempt
                return env
            except Exception as exc:
                err_txt = str(exc)
                tb_txt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                log.warning("[ENGINE] compute retry scan=%s attempt=%s/%s: %s", scan, attempt, self.max_retries, exc)
                time.sleep(COMPUTE_RETRY_SLEEP_S)
        return ResultEnvelope(scan=scan, volume_idx=scan, dicom_path=path, volume_timestamp=time.time(), success=False,
                              error=err_txt or "unknown compute failure", traceback_text=tb_txt, attempts=self.max_retries)

    # ---- ordering
    def _on_compute_done(self, future) -> None:
        try:
            env: ResultEnvelope = future.result()
        except CancelledError:
            return
        except Exception as exc:
            log.error("[ENGINE] compute future failed: %s\n%s", exc, traceback.format_exc())
            return
        with self._lock:
            if env.scan <= self.last_committed_scan or env.scan in self._timed_out:
                log.warning("[ENGINE] dropping late envelope scan=%s (last_committed=%s)", env.scan, self.last_committed_scan)
                self._inflight.discard(env.scan)
                return
            self._result_buffer[env.scan] = env
        self.drain()

    def _advance_locked(self) -> None:
        candidates = sorted(s for s in self._inflight | self._pending_known | set(self._result_buffer) if s > self.last_committed_scan)
        self._next_scan_to_commit = candidates[0] if candidates else None

    def drain(self) -> None:
        """Commit every scan that is ready, in order; skip a missing scan after the timeout."""
        while True:
            with self._lock:
                if self._next_scan_to_commit is None:
                    self._advance_locked()
                    if self._next_scan_to_commit is None:
                        return
                expected = self._next_scan_to_commit
                env = self._result_buffer.pop(expected, None)
                if env is None:
                    has_newer_ready = any(s > expected for s in self._result_buffer)
                    is_missing = expected not in self._inflight and expected not in self._pending_known
                    if is_missing and has_newer_ready:
                        first_seen = self._scan_first_seen.get(expected, time.time())
                        if time.time() - first_seen > self.commit_wait_timeout_s:
                            env = ResultEnvelope(scan=expected, volume_idx=expected, dicom_path=Path("missing"),
                                                 volume_timestamp=time.time(), success=False,
                                                 error=f"commit timeout waiting for scan {expected}")
                            self._timed_out.add(expected)
                    if env is None:
                        return
            self._commit_one(env)

    def _commit_one(self, env: ResultEnvelope) -> None:
        try:
            self._commit(env)
            if env.success:
                self.committed += 1
            else:
                self.failed += 1
        except Exception as exc:
            self.failed += 1
            log.error("[ENGINE] commit raised for scan=%s: %s\n%s", env.scan, exc, traceback.format_exc())
        finally:
            with self._lock:
                if env.scan > self.last_committed_scan:
                    self.last_committed_scan = env.scan
                self._inflight.discard(env.scan)
                self._scan_first_seen.pop(env.scan, None)
                self._advance_locked()

    # ---- lifecycle
    @property
    def idle(self) -> bool:
        with self._lock:
            return not self._inflight and not self._result_buffer

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
