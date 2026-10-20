"""rt_pipeline.ScoreEventTracker: the generic "stages" schema labels volumes exactly
as the legacy metadata forms did, and the legacy forms still work."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_rt_pipeline_parallel as headless  # noqa: E402,F401  (installs the headless stubs, imports rt_pipeline)
import rt_pipeline  # noqa: E402
from rt_session import StageSchedule  # noqa: E402


def _write(tmp_path, payload):
    (tmp_path / "session_metadata.json").write_text(json.dumps(payload))
    return rt_pipeline.ScoreEventTracker(tmp_path)


def test_no_metadata_gives_no_labels(tmp_path):
    tracker = rt_pipeline.ScoreEventTracker(tmp_path)
    assert tracker.for_volume(1) is None and tracker.for_volume(100) is None


def test_stages_schema_matches_legacy_nf_events(tmp_path):
    params = dict(n_trials=3, skip_first_trs=2, baseline_trs=4, iti_trs=3, cue_trs=4, scans_trs=3, delay_trs=3, feedback_trs=3)
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    legacy = _write(tmp_path / "a", {"psychopy": {"script": "rt_nf_events_parallel.py", **params}})
    sched = StageSchedule(skip_first_trs=2, baseline_trs=4, n_cycles=3,
                         cycle=[("iti", 3), ("cue", 4), ("scans", 3), ("delay", 3), ("feedback", 3)])
    generic = _write(tmp_path / "b", {"stages": sched.to_metadata()})
    vols = range(1, 80)
    assert [generic.for_volume(v) for v in vols] == [legacy.for_volume(v) for v in vols]
    assert legacy.for_volume(1) == "background" and legacy.for_volume(7) == "iti" and legacy.for_volume(79) == "post_task"


def test_stages_schema_matches_legacy_rest(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    legacy = _write(tmp_path / "a", {"fixation_display": {"skip_first_trs": 10, "baseline_trs": 20}})
    generic = _write(tmp_path / "b", {"stages": StageSchedule(skip_first_trs=10, baseline_trs=20,
                                                               cycle=[("rest", 1)], n_cycles=None).to_metadata()})
    vols = range(1, 200)
    assert [generic.for_volume(v) for v in vols] == [legacy.for_volume(v) for v in vols]
    assert legacy.for_volume(31) == "rest"


def test_stages_takes_precedence_over_legacy_keys(tmp_path):
    tracker = _write(tmp_path, {"fixation_display": {"skip_first_trs": 0, "baseline_trs": 0},
                                "stages": StageSchedule(cycle=[("x", 2)], n_cycles=1).to_metadata()})
    assert [tracker.for_volume(v) for v in (1, 2, 3)] == ["x", "x", "post_task"]


def test_psychopy_without_matching_script_still_unlabelled(tmp_path):
    tracker = _write(tmp_path, {"psychopy": {"max_points": 20}})
    assert tracker.for_volume(5) is None
