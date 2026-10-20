"""rt_session with fake processes: no scanner, no GPU, no rt_pipeline import.

The fake targets are module-level so the spawn start method can pickle them.
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_session import (  # noqa: E402
    ProcessSpec, ScoreEvent, Session, SessionError, StageSchedule, label_for_volume, merge_session_metadata,
)


# ----------------------------------------------------------------------------- fake process targets
def fake_pipeline(n_events, score_queue, stop_event):
    """Emit n ticks/scores like rt_pipeline, then wait for the stop event (like run_rt_pipeline)."""
    for i in range(1, n_events + 1):
        score_queue.put({"kind": "score", "volume_idx": i, "timestamp": time.time(), "watchdog_timestamp": time.time(),
                         "analysis_timestamp": time.time(), "score_raw": float(i), "score_original": None,
                         "reg_ready": i > 2, "event_type": "cue" if i % 2 else None})
    while not stop_event.is_set():
        time.sleep(0.01)


def fake_pipeline_exits(n_events, score_queue, stop_event):
    for i in range(1, n_events + 1):
        score_queue.put({"kind": "tick", "volume_idx": i, "watchdog_timestamp": time.time()})


def fake_pipeline_raises(score_queue, stop_event):
    score_queue.put({"kind": "tick", "volume_idx": 1, "watchdog_timestamp": time.time()})
    raise RuntimeError("boom")


def fake_pipeline_ignores_stop(score_queue, stop_event):
    while True:
        time.sleep(0.05)


def fake_pipeline_with_grandchild(pid_file, score_queue, stop_event):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_sleep_forever, daemon=True)
    child.start()
    Path(pid_file).write_text(str(child.pid))
    score_queue.put({"kind": "tick", "volume_idx": 1, "watchdog_timestamp": time.time()})
    while not stop_event.is_set():
        time.sleep(0.01)


def _sleep_forever():
    while True:
        time.sleep(0.1)


def fake_scorer(score_queue, stop_event, label="pca"):
    """Legacy-style extra scorer: kind-less dicts like rs_pca_runtime, exits on stop."""
    score_queue.put({"volume_idx": 1, "timestamp": time.time(), "feedback_score": 55.0, "score_raw": 55.0,
                     "reg_ready": True, "blind_condition": True, "score_label": label})
    while not stop_event.is_set():
        time.sleep(0.01)


def fake_physio(flag_file, stop_event):
    Path(flag_file).write_text("started")
    while not stop_event.is_set():
        time.sleep(0.01)
    Path(flag_file).write_text("stopped")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ----------------------------------------------------------------------------- ScoreEvent
def test_from_message_pipeline_score_and_tick():
    ev = ScoreEvent.from_message({"kind": "score", "volume_idx": 7, "timestamp": 1.5, "analysis_timestamp": 1.6,
                                  "watchdog_timestamp": 1.0, "score_raw": 0.25, "score_original": None,
                                  "reg_ready": True, "event_type": "feedback", "score_z": -0.3})
    assert ev.kind == "score" and ev.volume_idx == 7 and ev.score_raw == 0.25 and ev.score_z == -0.3
    assert ev.reg_ready is True and ev.stage == "feedback" and ev.score_original is None and ev.meta == {}
    tick = ScoreEvent.from_message({"kind": "tick", "volume_idx": 3, "watchdog_timestamp": 2.0})
    assert tick.kind == "tick" and tick.score_raw is None and tick.reg_ready is None


def test_from_message_legacy_kinds_are_inferred():
    assert ScoreEvent.from_message({"volume_idx": 1, "watchdog_timestamp": 0.0}).kind == "tick"
    assert ScoreEvent.from_message({"volume_idx": 1, "score_raw": 1.0, "reg_ready": False}).kind == "score"
    pca = ScoreEvent.from_message({"volume_idx": 2, "timestamp": 0.0, "feedback_score": 60.0, "score_raw": 60.0,
                                   "reg_ready": True, "blind_condition": True}, source="pca")
    assert pca.kind == "scorer" and pca.source == "pca" and pca.meta["feedback_score"] == 60.0
    err = ScoreEvent.from_message({"volume_idx": 2, "timestamp": 0.0, "reg_ready": False, "error": "bad"}, source="pca")
    assert err.kind == "error" and err.meta["error"] == "bad"
    assert ScoreEvent.from_message({"volume_index": 9}).volume_idx == 9   # brainpong alias


def test_to_message_round_trip():
    msg = {"kind": "score", "volume_idx": 4, "timestamp": 1.0, "score_raw": 2.0, "reg_ready": True,
           "event_type": "iti", "custom": "x"}
    assert ScoreEvent.from_message(msg).to_message() == msg


# ----------------------------------------------------------------------------- stages schema
def test_label_for_volume_matches_nf_events_semantics():
    sched = StageSchedule(skip_first_trs=2, baseline_trs=3, cycle=[("iti", 1), ("cue", 2), ("feedback", 1)], n_cycles=2)
    labels = [sched.label_for_volume(v) for v in range(1, 16)]
    assert labels == ["background", "background", "background", "background", "background",
                      "iti", "cue", "cue", "feedback", "iti", "cue", "cue", "feedback", "post_task", "post_task"]
    rest = StageSchedule(skip_first_trs=1, baseline_trs=2, cycle=[("rest", 1)], n_cycles=None)
    assert [rest.label_for_volume(v) for v in (1, 2, 3, 4, 50)] == ["background", "background", "background", "rest", "rest"]
    assert label_for_volume(StageSchedule().to_metadata(), 5) is None


def test_stages_metadata_round_trip(tmp_path):
    sched = StageSchedule(skip_first_trs=0, baseline_trs=20, cycle=[("iti", 3), ("cue", 4)], n_cycles=20)
    merge_session_metadata(tmp_path, {"stages": sched.to_metadata(), "psychopy": {"script": "x"}})
    merge_session_metadata(tmp_path, {"other": 1})
    import json

    data = json.loads((tmp_path / "session_metadata.json").read_text())
    assert data["stages"]["version"] == 1 and data["stages"]["cycle"][1] == {"name": "cue", "trs": 4}
    assert data["psychopy"] == {"script": "x"} and data["other"] == 1


# ----------------------------------------------------------------------------- Session with fakes
def test_events_arrive_in_order_then_end(tmp_path):
    with Session(tmp_path, fake_pipeline_exits, (5,), stop_timeout_s=2.0) as s:
        events = list(s.scores(timeout=0.05))
    assert [e.kind for e in events] == ["tick"] * 5 + ["end"]
    assert [e.volume_idx for e in events[:5]] == [1, 2, 3, 4, 5]
    assert s.exit_codes["pipeline"] == 0 and not s.pipeline_alive


def test_poll_and_stop_while_pipeline_waits(tmp_path):
    s = Session(tmp_path, fake_pipeline, (3,), stop_timeout_s=2.0).start()
    got = []
    deadline = time.time() + 5
    while len(got) < 3 and time.time() < deadline:
        got.extend(s.poll())
        time.sleep(0.01)
    assert [e.volume_idx for e in got] == [1, 2, 3]
    assert got[0].reg_ready is False and got[2].reg_ready is True and got[0].stage == "cue"
    assert s.pipeline_alive
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 2.5          # cooperative stop, no terminate needed
    assert not s.pipeline_alive and s.exit_codes["pipeline"] == 0
    s.stop()                               # idempotent
    assert s.poll()[-1].kind == "end"      # end reported once after stop
    assert s.poll() == []


def test_pipeline_error_propagates(tmp_path):
    with Session(tmp_path, fake_pipeline_raises, (), stop_timeout_s=2.0) as s:
        kinds = []
        with pytest.raises(SessionError):
            for ev in s.scores(timeout=0.05):
                kinds.append(ev.kind)
    assert kinds == ["tick", "error"]
    assert s.exit_codes["pipeline"] not in (0, None)


def test_stop_terminates_a_pipeline_that_ignores_the_stop_event(tmp_path):
    s = Session(tmp_path, fake_pipeline_ignores_stop, (), stop_timeout_s=0.3).start()
    time.sleep(0.2)
    t0 = time.time()
    s.stop()
    assert time.time() - t0 < 6
    assert not s.pipeline_alive


def test_no_orphaned_grandchild_after_stop(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    with Session(tmp_path, fake_pipeline_with_grandchild, (str(pid_file),), stop_timeout_s=3.0) as s:
        deadline = time.time() + 5
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.01)
        pid = int(pid_file.read_text())
        assert _pid_alive(pid)
    deadline = time.time() + 3
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid), "daemon grandchild of the pipeline outlived the session"


def test_extra_process_hook_and_stop_order(tmp_path):
    spec = ProcessSpec("pca", fake_scorer, kwargs={"label": "pc1"})
    with Session(tmp_path, fake_pipeline, (1,), extra_processes=[spec], stop_timeout_s=2.0) as s:
        got = []
        deadline = time.time() + 5
        while len(got) < 2 and time.time() < deadline:
            got.extend(s.poll())
            time.sleep(0.01)
    by_source = {e.source: e for e in got}
    assert by_source["pipeline"].kind == "score"
    assert by_source["pca"].kind == "scorer" and by_source["pca"].meta["score_label"] == "pc1"
    assert s.exit_codes["pca"] == 0 and s.exit_codes["pipeline"] == 0
    assert not s.extra_processes["pca"].is_alive()


def test_physio_process_started_and_stopped(tmp_path):
    flag = tmp_path / "physio.txt"
    with Session(tmp_path, fake_pipeline, (1,), physio_target=fake_physio, physio_args=(str(flag),),
                 stop_timeout_s=2.0) as s:
        deadline = time.time() + 5
        while not flag.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert flag.read_text() == "started"
    assert flag.read_text() == "stopped" and s.exit_codes["physio"] == 0


def test_context_manager_cleans_up_on_display_exception(tmp_path):
    s = Session(tmp_path, fake_pipeline, (1,), stop_timeout_s=2.0)
    with pytest.raises(ValueError):
        with s:
            assert s.pipeline_alive
            raise ValueError("display crashed")
    assert not s.pipeline_alive and s.exit_codes["pipeline"] == 0
