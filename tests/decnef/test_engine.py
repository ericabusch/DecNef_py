import threading
import time
from pathlib import Path

from decnef.engine import OrderedEngine, ResultEnvelope


def _engine(order_log, compute_delay=None, fail_scans=(), **kw):
    def compute(path, scan, volume_idx):
        if compute_delay:
            time.sleep(compute_delay.get(scan, 0.0))
        if scan in fail_scans:
            raise RuntimeError(f"boom {scan}")
        return ResultEnvelope(scan=scan, volume_idx=volume_idx, dicom_path=path, volume_timestamp=time.time(), success=True,
                              raw_nii=Path(f"vol_{scan}.nii"))

    lock = threading.Lock()

    def commit(env):
        with lock:
            order_log.append((env.scan, env.success, env.error))

    return OrderedEngine(compute, commit, **kw)


def _wait(engine, n, log, timeout=5.0):
    deadline = time.time() + timeout
    while len(log) < n and time.time() < deadline:
        engine.drain()
        time.sleep(0.01)


def test_commits_in_scan_order_even_when_compute_finishes_out_of_order():
    log = []
    eng = _engine(log, compute_delay={1: 0.3, 2: 0.0, 3: 0.1}, max_workers=3, commit_wait_timeout_s=5.0)
    for scan in (1, 2, 3):
        eng.submit(Path(f"{scan}.dcm"), scan)
    _wait(eng, 3, log)
    assert [s for s, ok, _ in log] == [1, 2, 3] and all(ok for _, ok, _ in log)
    assert eng.committed == 3 and eng.failed == 0 and eng.idle
    eng.shutdown()


def test_compute_failure_is_committed_as_failed_envelope_after_retries():
    log = []
    eng = _engine(log, fail_scans={2}, max_workers=2, max_retries=2, commit_wait_timeout_s=5.0)
    import decnef.engine as E
    E.COMPUTE_RETRY_SLEEP_S = 0.0
    for scan in (1, 2, 3):
        eng.submit(Path(f"{scan}.dcm"), scan)
    _wait(eng, 3, log)
    assert [(s, ok) for s, ok, _ in log] == [(1, True), (2, False), (3, True)]
    assert "boom 2" in log[1][2] and eng.failed == 1
    eng.shutdown()


def test_missing_scan_is_skipped_after_timeout_and_late_result_dropped():
    log = []
    eng = _engine(log, max_workers=2, commit_wait_timeout_s=0.1)
    eng.submit(Path("1.dcm"), 1)
    _wait(eng, 1, log)
    eng.note_pending({2})                 # the watcher saw scan 2 ...
    eng.submit(Path("3.dcm"), 3)
    _wait(eng, 2, log, timeout=0.5)
    assert [s for s, _, _ in log] == [1]  # ... so 3 waits for it
    eng.note_pending(set())               # ... but it vanished (never submitted): skipped after the timeout
    _wait(eng, 3, log, timeout=3.0)
    assert [(s, ok) for s, ok, _ in log] == [(1, True), (2, False), (3, True)]
    assert "timeout" in log[1][2]
    eng.submit(Path("2.dcm"), 2)          # late: already past it
    time.sleep(0.2)
    eng.drain()
    assert len(log) == 3 and eng.last_committed_scan == 3
    eng.shutdown()


def test_pending_known_scans_hold_the_commit_order():
    log = []
    eng = _engine(log, max_workers=2, commit_wait_timeout_s=0.05)
    eng.note_pending({2})                 # watcher has seen scan 2 but it is not submitted yet
    eng.submit(Path("1.dcm"), 1)
    eng.submit(Path("3.dcm"), 3)
    _wait(eng, 1, log)
    time.sleep(0.2)
    eng.drain()
    assert [s for s, _, _ in log] == [1]  # 3 waits because 2 is known-pending
    eng.note_pending(set())
    eng.submit(Path("2.dcm"), 2)
    _wait(eng, 3, log)
    assert [s for s, _, _ in log] == [1, 2, 3]
    eng.shutdown()
