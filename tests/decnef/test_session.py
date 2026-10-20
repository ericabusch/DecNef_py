"""decnef.session with fake spawned processes (no scanner, no GPU, no pipeline import)."""
import os
import time
from pathlib import Path

import pytest

from decnef.session import ProcessSpec, Session, SessionError


def fake_pipeline(n, score_queue, stop_event):
    for i in range(1, n + 1):
        score_queue.put({"kind": "score", "source": "pipeline", "volume_idx": i, "timestamp": time.time(),
                         "watchdog_timestamp": time.time(), "score_raw": float(i), "reg_ready": i > 1, "stage": "cue"})
    while not stop_event.is_set():
        time.sleep(0.01)


def fake_pipeline_exits(n, score_queue, stop_event):
    for i in range(1, n + 1):
        score_queue.put({"kind": "tick", "source": "pipeline", "volume_idx": i})


def fake_pipeline_raises(score_queue, stop_event):
    score_queue.put({"kind": "tick", "source": "pipeline", "volume_idx": 1})
    raise RuntimeError("boom")


def fake_pipeline_ignores_stop(score_queue, stop_event):
    while True:
        time.sleep(0.05)


def fake_pipeline_with_grandchild(pid_file, score_queue, stop_event):
    import multiprocessing as mp

    child = mp.get_context("spawn").Process(target=_sleep_forever, daemon=True)
    child.start()
    Path(pid_file).write_text(str(child.pid))
    while not stop_event.is_set():
        time.sleep(0.01)


def _sleep_forever():
    while True:
        time.sleep(0.1)


def fake_scorer(score_queue, stop_event, label):
    score_queue.put({"kind": "scorer", "source": "pca", "volume_idx": 1, "feedback_score": 55.0, "score_label": label})
    while not stop_event.is_set():
        time.sleep(0.01)


def fake_physio(flag_file, stop_event):
    Path(flag_file).write_text("started")
    while not stop_event.is_set():
        time.sleep(0.01)
    Path(flag_file).write_text("stopped")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_events_then_end(tmp_path):
    with Session(tmp_path, fake_pipeline_exits, (4,), stop_timeout_s=2.0) as s:
        events = list(s.scores(timeout=0.05))
    assert [e.kind for e in events] == ["tick"] * 4 + ["end"] and [e.volume_idx for e in events[:4]] == [1, 2, 3, 4]
    assert s.exit_codes["pipeline"] == 0


def test_poll_and_cooperative_stop(tmp_path):
    s = Session(tmp_path, fake_pipeline, (3,), stop_timeout_s=2.0).start()
    got, deadline = [], time.time() + 5
    while len(got) < 3 and time.time() < deadline:
        got.extend(s.poll())
        time.sleep(0.01)
    assert [e.volume_idx for e in got] == [1, 2, 3] and got[0].reg_ready is False and got[2].stage == "cue"
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 2.5 and not s.pipeline_alive and s.exit_codes["pipeline"] == 0
    s.stop()
    assert s.poll()[-1].kind == "end" and s.poll() == []


def test_pipeline_error_propagates(tmp_path):
    kinds = []
    with Session(tmp_path, fake_pipeline_raises, (), stop_timeout_s=2.0) as s:
        with pytest.raises(SessionError):
            for ev in s.scores(timeout=0.05):
                kinds.append(ev.kind)
    assert kinds == ["tick", "error"] and s.exit_codes["pipeline"] not in (0, None)


def test_stop_terminates_a_stuck_pipeline(tmp_path):
    s = Session(tmp_path, fake_pipeline_ignores_stop, (), stop_timeout_s=0.3).start()
    time.sleep(0.2)
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 6 and not s.pipeline_alive


def test_no_orphaned_grandchild(tmp_path):
    pid_file = tmp_path / "pid"
    with Session(tmp_path, fake_pipeline_with_grandchild, (str(pid_file),), stop_timeout_s=3.0):
        deadline = time.time() + 5
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        pid = int(pid_file.read_text())
        assert _pid_alive(pid)
    deadline = time.time() + 3
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid)


def test_extra_process_and_physio(tmp_path):
    flag = tmp_path / "physio.txt"
    spec = ProcessSpec("pca", fake_scorer, kwargs={"label": "pc1"})
    with Session(tmp_path, fake_pipeline, (1,), physio_target=fake_physio, physio_args=(str(flag),),
                 extra_processes=[spec], stop_timeout_s=2.0) as s:
        got, deadline = [], time.time() + 5
        while len(got) < 2 and time.time() < deadline:
            got.extend(s.poll())
            time.sleep(0.01)
    by = {e.source: e for e in got}
    assert by["pipeline"].kind == "score" and by["pca"].kind == "scorer" and by["pca"].meta["score_label"] == "pc1"
    assert s.exit_codes["pca"] == 0 and s.exit_codes["physio"] == 0 and flag.read_text() == "stopped"


def test_cleanup_on_display_exception(tmp_path):
    s = Session(tmp_path, fake_pipeline, (1,), stop_timeout_s=2.0)
    with pytest.raises(ValueError):
        with s:
            raise ValueError("display crashed")
    assert not s.pipeline_alive and s.exit_codes["pipeline"] == 0
