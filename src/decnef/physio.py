"""BIOPAC RetroTS physio regressors.

The BIOPAC PC streams one JSON line per volume over TCP:
    {"kind": "retrots", "volume_idx": 12, "n_regressors": 8, "regressors": [...], "timestamp": ..., "tr": ...}
`Receiver` listens, buffers per-volume regressors and hands them to RtpRegress
(`get_retrots`); a volume that does not arrive within `timeout` is zero-filled
and recorded as missing. It also sends `handshake` (with the TR), `run_start`
and `run_end` control lines. `FileBuffer` reads the same regressors from a CSV
that a `Receiver` in another process keeps appending to (physio.listener mode).

Unchanged behavior from `biopac_rt/biopac_receiver.py`; the lab's default IP
is gone (host must come from the config).
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ReceiverConfig:
    host: str
    port: int
    timeout: float = 0.3
    expected_regressors: Optional[int] = None
    handshake_tr: Optional[float] = None
    subject: Optional[str] = None
    day: Optional[str] = None
    run: Optional[str] = None
    output_path: Optional[Path] = None


class Receiver:
    def __init__(self, config: ReceiverConfig):
        self.config = config
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._regressors_by_vol: dict[int, np.ndarray] = {}
        self._timestamps_by_vol: dict[int, float] = {}
        self._n_reg: Optional[int] = None
        self._missing_vols: set[int] = set()
        self._server_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._output_lock = threading.Lock()
        self._bound = threading.Event()

    @property
    def port(self) -> int:
        """The bound port (useful when configured with port 0)."""
        return self._server_sock.getsockname()[1] if self._server_sock is not None else self.config.port

    def start(self, wait_bound: float = 2.0):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._bound.wait(timeout=wait_bound)
        log.info("BIOPAC receiver listening on %s:%s", self.config.host, self.port)

    def stop(self):
        self._stop.set()
        self._send_run_end()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_retrots(self, TR: float, vol_idx: int, tshift: float, timeout: Optional[float] = None):
        wait_time = self.config.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, wait_time)
        with self._cond:
            while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            n_reg = self._ensure_regressor_count()
            retro = np.zeros((vol_idx, n_reg), dtype=np.float32)
            for idx, reg in self._regressors_by_vol.items():
                if 1 <= idx <= vol_idx:
                    retro[idx - 1, : reg.shape[0]] = reg
            if vol_idx not in self._regressors_by_vol:
                if vol_idx not in self._missing_vols:
                    log.warning("[BIOPAC] Missing physio regressors for vol %s; using zeros.", vol_idx)
                self._missing_vols.add(vol_idx)
            return retro

    def was_missing(self, vol_idx: int) -> bool:
        with self._lock:
            return vol_idx in self._missing_vols

    def get_trigger_timestamp(self, vol_idx: int) -> Optional[float]:
        with self._lock:
            return self._timestamps_by_vol.get(vol_idx)

    def _ensure_regressor_count(self) -> int:
        if self._n_reg is None:
            self._n_reg = self.config.expected_regressors if self.config.expected_regressors is not None else 8
        return self._n_reg

    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(1)
            server.settimeout(0.5)
            self._server_sock = server
            self._bound.set()
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                log.info("[BIOPAC] Connected from %s:%s", addr[0], addr[1])
                with self._lock:
                    self._conn = conn
                if self.config.handshake_tr is not None:
                    payload = json.dumps({"kind": "handshake", "tr": self.config.handshake_tr, "timestamp": time.time()})
                    try:
                        conn.sendall((payload + "\n").encode("utf-8"))
                    except OSError:
                        log.warning("[BIOPAC] Failed to send handshake.")
                with conn:
                    conn.settimeout(0.5)
                    buffer = ""
                    while not self._stop.is_set():
                        try:
                            chunk = conn.recv(4096)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not chunk:
                            break
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                self._handle_line(line)
                log.info("[BIOPAC] Connection closed.")
                with self._lock:
                    self._conn = None

    def send_run_start(self):
        if not (self.config.subject and self.config.day and self.config.run):
            return
        self._send_control({"kind": "run_start", "subject": self.config.subject, "day": self.config.day,
                            "run": self.config.run, "timestamp": time.time()})

    def send_run_end(self):
        self._send_run_end()

    def _send_run_end(self):
        self._send_control({"kind": "run_end", "timestamp": time.time()})

    def _send_control(self, payload: dict):
        with self._lock:
            conn = self._conn
        if conn is None:
            return
        try:
            conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            log.warning("[BIOPAC] Failed to send control message.")

    def _handle_line(self, line: str):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.warning("[BIOPAC] Malformed JSON message ignored.")
            return
        if payload.get("kind") != "retrots":
            return
        vol_idx = payload.get("volume_idx")
        regressors = payload.get("regressors")
        if not isinstance(vol_idx, int) or not isinstance(regressors, list):
            return
        reg = np.asarray(regressors, dtype=np.float32)
        if reg.ndim != 1:
            return
        with self._cond:
            if self._n_reg is None:
                self._n_reg = int(payload.get("n_regressors", reg.shape[0]))
            self._regressors_by_vol[vol_idx] = reg
            ts = payload.get("timestamp")
            if ts is not None:
                try:
                    self._timestamps_by_vol[vol_idx] = float(ts)
                except (TypeError, ValueError):
                    pass
            self._cond.notify_all()
        self._append_received(vol_idx, reg, payload.get("timestamp"))
        log.debug("[BIOPAC] Received regressors for vol %s (%d values).", vol_idx, reg.shape[0])

    def _append_received(self, vol_idx: int, reg: np.ndarray, timestamp: Optional[float]) -> None:
        if self.config.output_path is None:
            return
        with self._output_lock:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            try:
                with open(path, "a", newline="") as f:
                    if not exists:
                        f.write(",".join(["volume_idx", "timestamp"] + [f"reg_{i + 1:02d}" for i in range(reg.shape[0])]) + "\n")
                    ts = time.time() if timestamp is None else float(timestamp)
                    f.write(",".join([str(vol_idx), f"{ts:.6f}"] + [f"{v:.6f}" for v in reg.tolist()]) + "\n")
            except OSError as exc:
                log.warning("[BIOPAC] Failed to write regressors CSV: %s", exc)

    def wait_for_volume(self, vol_idx: int, timeout: Optional[float] = None) -> bool:
        """Block until `vol_idx` is buffered; no side effects (no zero-fill, nothing marked missing)."""
        with self._cond:
            if timeout is None:
                while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                    self._cond.wait(timeout=0.5)
                return (vol_idx in self._regressors_by_vol) and (not self._stop.is_set())
            deadline = time.monotonic() + max(0.0, timeout)
            while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(0.5, remaining))
            return vol_idx in self._regressors_by_vol


class FileBuffer:
    """RetroTS regressors tailed from the CSV a `Receiver` (in another process) writes."""

    def __init__(self, path: Path, timeout: float = 0.3, expected_regressors: Optional[int] = None, poll_interval: float = 0.05):
        self.path = Path(path)
        self.timeout = timeout
        self.expected_regressors = expected_regressors
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._regressors_by_vol: dict[int, np.ndarray] = {}
        self._missing_vols: set[int] = set()
        self._n_reg: Optional[int] = None
        self._file_pos = 0
        self._header_skipped = False
        self._remainder = ""

    def _ensure_regressor_count(self) -> int:
        if self._n_reg is None:
            self._n_reg = self.expected_regressors if self.expected_regressors is not None else 8
        return self._n_reg

    def _update_from_file(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                chunk = f.read()
                self._file_pos = f.tell()
        if not chunk:
            return
        chunk = f"{self._remainder}{chunk}"
        if chunk and not chunk.endswith("\n"):
            chunk, self._remainder = chunk.rsplit("\n", 1)
        else:
            self._remainder = ""
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if not self._header_skipped:
                self._header_skipped = True
                if line.lower().startswith("volume_idx"):
                    continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                continue
            try:
                vol_idx = int(parts[0])
                reg = np.asarray([float(v) for v in parts[2:]], dtype=np.float32)
            except ValueError:
                continue
            if self._n_reg is None:
                self._n_reg = reg.shape[0]
            self._regressors_by_vol[vol_idx] = reg

    def wait_for_volume(self, vol_idx: int, timeout: Optional[float] = None) -> bool:
        wait_time = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, wait_time)
        while time.monotonic() < deadline:
            self._update_from_file()
            if vol_idx in self._regressors_by_vol:
                return True
            time.sleep(self.poll_interval)
        return vol_idx in self._regressors_by_vol

    def get_retrots(self, TR: float, vol_idx: int, tshift: float, timeout: Optional[float] = None):
        self.wait_for_volume(vol_idx, timeout=timeout)
        self._update_from_file()
        n_reg = self._ensure_regressor_count()
        retro = np.zeros((vol_idx, n_reg), dtype=np.float32)
        for idx, reg in self._regressors_by_vol.items():
            if 1 <= idx <= vol_idx:
                retro[idx - 1, : reg.shape[0]] = reg
        if vol_idx not in self._regressors_by_vol:
            if vol_idx not in self._missing_vols:
                log.warning("[BIOPAC] Missing physio regressors for vol %s in file; using zeros.", vol_idx)
            self._missing_vols.add(vol_idx)
        return retro

    def was_missing(self, vol_idx: int) -> bool:
        with self._lock:
            return vol_idx in self._missing_vols

    def get_trigger_timestamp(self, vol_idx: int) -> Optional[float]:
        return None
