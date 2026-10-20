"""rt_nf_events_parallel on the session API, headless: a fake `psychopy` and a fake
session drive the display loop through trigger -> baseline -> trials -> end, and
check the trial scores written and the 'q' key ending the run."""
import csv
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rt_session import ScoreEvent  # noqa: E402


# ----------------------------------------------------------------------------- fake psychopy
class _Stim:
    def __init__(self, *a, **k):
        self.text = ""
        self.fillColor = None
        self.lineColor = None

    def draw(self):
        pass

    def setRadius(self, *a, **k):
        pass


class _Window:
    size = (1000, 700)
    closed = 0

    def __init__(self, *a, **k):
        self.fullscr = False
        self.color = None
        self.mouseVisible = False

    def flip(self):
        pass

    def clearBuffer(self):
        pass

    def close(self):
        _Window.closed += 1


class _Core:
    @staticmethod
    def wait(_s):
        pass


class _Event:
    keys: list = []

    @classmethod
    def getKeys(cls):
        return cls.keys.pop(0) if cls.keys else []


def _install_fake_psychopy():
    psychopy = types.ModuleType("psychopy")
    visual = types.ModuleType("psychopy.visual")
    for name in ("TextStim", "ShapeStim", "Circle"):
        setattr(visual, name, _Stim)
    visual.Window = _Window
    psychopy.visual, psychopy.core, psychopy.event = visual, _Core, _Event
    sys.modules["psychopy"] = psychopy
    sys.modules["psychopy.visual"] = visual
    sys.modules["psychopy.core"] = _Core
    sys.modules["psychopy.event"] = _Event


class FakeSession:
    """Delivers one pipeline score event per poll(), like volumes arriving at each frame."""

    def __init__(self, n_volumes, reg_ready_from=1):
        self.events = [ScoreEvent(kind="score", volume_idx=v, timestamp=10.0 + v, watchdog_timestamp=9.5 + v,
                                  analysis_timestamp=10.0 + v, score_raw=float(v), reg_ready=v >= reg_ready_from)
                       for v in range(1, n_volumes + 1)]
        self.polls = 0

    def poll(self):
        self.polls += 1
        return [self.events.pop(0)] if self.events else []


def _run(tmp_path, keys, n_volumes, **kw):
    _install_fake_psychopy()
    _Event.keys = list(keys)
    _Window.closed = 0
    import importlib
    import rt_nf_events_parallel as nf
    importlib.reload(nf)
    session = FakeSession(n_volumes, reg_ready_from=kw.pop("reg_ready_from", 1))
    nf.run_nf_events_presentation(
        session=session, skip_first_trs=0, baseline_trs=2, n_trials=2, iti_trs=1, cue_trs=1, scans_trs=2,
        delay_trs=1, feedback_trs=1, score_delay=0, tr_seconds=1.0, trial_scores_path=tmp_path / "trial_scores.csv",
        reference_stats=None, **kw)
    return session


def test_full_schedule_scores_each_trial_from_its_scan_window(tmp_path):
    # 2 baseline + 2 trials x (1+1+2+1+1) = 14 volumes; 's' starts, then no keys until the end
    session = _run(tmp_path, keys=[["s"]], n_volumes=14)
    rows = list(csv.DictReader(open(tmp_path / "trial_scores.csv")))
    assert [int(r["trial"]) for r in rows] == [1, 2]
    # trial 1: cue at TR 4, scans TRs 5-6 -> window [4, 6] with score_delay 0 -> mean(4, 5, 6); trial 2: window [10, 12]
    assert [float(r["trial_score_raw"]) for r in rows] == [5.0, 11.0]
    assert [r["score_missing"] for r in rows] == ["0", "0"]
    assert _Window.closed == 1 and session.events == []


def test_scores_before_reg_ready_are_not_used(tmp_path):
    _run(tmp_path, keys=[["s"]], n_volumes=14, reg_ready_from=12)
    rows = list(csv.DictReader(open(tmp_path / "trial_scores.csv")))
    assert rows[0]["score_missing"] == "1" and rows[1]["n_scores_used"] == "1"   # only TR 12 of trial 2's window


def test_q_key_ends_the_run_cleanly(tmp_path):
    session = _run(tmp_path, keys=[["s"], [], [], ["q"]], n_volumes=14)
    assert _Window.closed == 1
    assert session.events, "display returned on 'q' before consuming the whole run"
    assert not (tmp_path / "trial_scores.csv").exists()


def test_q_during_trigger_wait_returns(tmp_path):
    _run(tmp_path, keys=[[], ["q"]], n_volumes=3)
    assert _Window.closed == 1
