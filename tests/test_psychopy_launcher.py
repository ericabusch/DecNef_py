"""rt_psychopy_parallel on the session API, headless: fake psychopy + fake session.
Checks that decoder scores and PCA-scorer events both reach scores_with_conditions.csv,
that a missing reg_ready counts as ready (as before), that max_trs ends the run,
and that 'q' ends it."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_nf_events_launcher as fake  # noqa: E402  (fake psychopy helpers)
from rt_session import ScoreEvent  # noqa: E402


class _Line(fake._Stim):
    def __init__(self, *a, **k):
        super().__init__()
        self.vertices = None


def _install():
    fake._install_fake_psychopy()
    sys.modules["psychopy.visual"].Line = _Line
    sys.modules["psychopy.visual"].ShapeStim = _Line
    fake._Window.closed = 0


class FakeSession:
    def __init__(self, events):
        self.events = list(events)

    def poll(self):
        return [self.events.pop(0)] if self.events else []


def _run(tmp_path, events, keys, max_trs=None):
    _install()
    fake._Event.keys = list(keys)
    import importlib
    import rt_psychopy_parallel as ps
    importlib.reload(ps)
    cond = ps.Condition(condition_id="c1", roi="LPFC", direction="up", symbol="#")
    session = FakeSession(events)
    ps.run_psychopy_presentation(session, 20, cond, tmp_path / "scores_with_conditions.csv", max_trs)
    return session


def test_decoder_scores_and_pca_scorer_events_are_recorded(tmp_path):
    events = [
        ScoreEvent(kind="score", volume_idx=1, timestamp=1.0, watchdog_timestamp=0.5, analysis_timestamp=1.0,
                   score_raw=0.3, reg_ready=False),                       # not ready: shown but not recorded
        ScoreEvent(kind="score", volume_idx=2, timestamp=2.0, watchdog_timestamp=1.5, analysis_timestamp=2.0,
                   score_raw=0.4, reg_ready=True),
        ScoreEvent(kind="tick", volume_idx=3, watchdog_timestamp=2.5),    # PCA mode: pipeline tick, no score
        ScoreEvent.from_message({"volume_idx": 3, "timestamp": 3.0, "feedback_score": 61.0, "score_raw": 61.0,
                                 "reg_ready": True, "blind_condition": True, "score_label": "PC1",
                                 "raw_component_score": 0.1, "directed_score": 0.2, "score_z": 0.9}, source="pca_scorer"),
        ScoreEvent.from_message({"volume_idx": 4, "score_raw": 0.5}),      # legacy message without reg_ready -> ready
    ]
    session = _run(tmp_path, events, keys=[["s"]], max_trs=4)
    rows = list(csv.DictReader(open(tmp_path / "scores_with_conditions.csv")))
    assert [r["volume_idx"] for r in rows] == ["2", "3", "4"]
    assert rows[1]["feedback_score"] == "61.0" and rows[1]["score_label"] == "PC1" and rows[1]["roi"] == ""  # blinded
    assert rows[0]["roi"] == "LPFC" and rows[2]["score_raw"] == "0.5"
    assert fake._Window.closed == 1 and session.events == []
    speed = list(csv.DictReader(open(tmp_path / "acquisition_speed_rt.csv")))
    assert [r["volume_idx"] for r in speed] == ["1", "2"]   # only events with watchdog+analysis timestamps


def test_q_ends_the_run(tmp_path):
    events = [ScoreEvent(kind="score", volume_idx=v, score_raw=1.0, reg_ready=True) for v in range(1, 50)]
    session = _run(tmp_path, events, keys=[["s"], [], ["q"]], max_trs=None)
    assert fake._Window.closed == 1 and session.events
