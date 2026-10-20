"""examples/nf_events_launcher.py headless: a fake psychopy and a fake session drive
the display through trigger -> baseline -> trials -> end; 'q' ends the run."""
import csv
import importlib.util
import sys
import types
from pathlib import Path

from decnef.events import ScoreEvent, StageSchedule

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "nf_events_launcher.py"


class _Stim:
    def __init__(self, *a, **k):
        self.text = ""

    def draw(self):
        pass

    def setRadius(self, *a, **k):
        pass


class _Window:
    size = (1000, 700)
    closed = 0

    def __init__(self, *a, **k):
        self.mouseVisible = False

    def flip(self):
        pass

    def close(self):
        _Window.closed += 1


class _Core:
    @staticmethod
    def wait(_s):
        pass


class _Event:
    keys = []

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
    for name, mod in (("psychopy", psychopy), ("psychopy.visual", visual), ("psychopy.core", _Core), ("psychopy.event", _Event)):
        sys.modules[name] = mod


def _load_example():
    spec = importlib.util.spec_from_file_location("nf_events_launcher", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSession:
    def __init__(self, n, reg_ready_from=1):
        self.events = [ScoreEvent(kind="score", source="pipeline", volume_idx=v, timestamp=10.0 + v, watchdog_timestamp=9.5 + v,
                                  analysis_timestamp=10.0 + v, score_raw=float(v), reg_ready=v >= reg_ready_from)
                       for v in range(1, n + 1)]

    def poll(self):
        return [self.events.pop(0)] if self.events else []


def _run(tmp_path, keys, n_volumes, reg_ready_from=1, skip=0):
    _install_fake_psychopy()
    _Event.keys = list(keys)
    _Window.closed = 0
    nf = _load_example()
    schedule = StageSchedule(skip_first_trs=skip, baseline_trs=2, n_cycles=2,
                             cycle=[("iti", 1), ("cue", 1), ("scans", 2), ("delay", 1), ("feedback", 1)])
    session = FakeSession(n_volumes, reg_ready_from)
    nf.run_display(session, schedule, score_delay=0, tr_seconds=1.0, trial_scores_path=tmp_path / "trial_scores.csv",
                   reference_stats=None)
    return session


def test_full_schedule_scores_each_trial(tmp_path):
    session = _run(tmp_path, keys=[["s"]], n_volumes=14)         # 2 baseline + 2 x 6
    rows = list(csv.DictReader(open(tmp_path / "trial_scores.csv")))
    assert [int(r["trial"]) for r in rows] == [1, 2]
    assert [float(r["trial_score_raw"]) for r in rows] == [5.0, 11.0]    # mean over TRs [4,6] and [10,12]
    assert [r["score_missing"] for r in rows] == ["0", "0"] and session.events == [] and _Window.closed == 1
    speed = list(csv.DictReader(open(tmp_path / "acquisition_speed_rt.csv")))
    assert len(speed) == 13 and speed[0]["volume_idx"] == "2"   # volume 1 arrived during the trigger wait (no log yet)


def test_skip_first_trs_shifts_experiment_trs(tmp_path):
    _run(tmp_path, keys=[["s"]], n_volumes=16, skip=2)
    rows = list(csv.DictReader(open(tmp_path / "trial_scores.csv")))
    assert [float(r["trial_score_raw"]) for r in rows] == [7.0, 13.0]     # volumes 6-8 and 12-14


def test_not_ready_scores_are_missing(tmp_path):
    _run(tmp_path, keys=[["s"]], n_volumes=14, reg_ready_from=12)
    rows = list(csv.DictReader(open(tmp_path / "trial_scores.csv")))
    assert rows[0]["score_missing"] == "1" and rows[1]["n_scores_used"] == "1"


def test_q_ends_the_run(tmp_path):
    session = _run(tmp_path, keys=[["s"], [], [], ["q"]], n_volumes=14)
    assert _Window.closed == 1 and session.events and not (tmp_path / "trial_scores.csv").exists()
    _run(tmp_path / "b", keys=[[], ["escape"]], n_volumes=3)
    assert _Window.closed == 1
