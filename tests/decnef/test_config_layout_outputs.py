import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from decnef import config as C
from decnef.decoder import DecoderScorer
from decnef.events import ScoreEvent, StageSchedule
from decnef.layout import RunLayout
from decnef import outputs

MINIMAL = {"paths": {"base_data": "/data"}, "pipeline": {"TR": 1.4}, "decoder": {"template": "dec.nii"}}


# ----------------------------------------------------------------------------- config
def test_minimal_config_builds_with_documented_defaults():
    s = C.from_dict(MINIMAL)
    assert s.pipeline.TR == 1.4 and s.pipeline.analysis_space == "mni" and s.pipeline.censor.fd.thr == 0.3
    assert s.physio.enable is False and s.pipeline.motion.max_poly_order == math.inf
    assert s.paths.base_data == Path("/data") and s.decoder.template == Path("dec.nii")


@pytest.mark.parametrize("bad, msg", [
    ({"pipeline": {"TR": 1.4}}, "paths"),
    ({"paths": {"base_data": "/d"}, "pipeline": {}}, "pipeline.TR"),
    ({**MINIMAL, "decoder": {}}, "decoder.template"),
    ({**MINIMAL, "pipeline": {"TR": 1.4, "analysis_space": "mars"}}, "analysis_space"),
    ({**MINIMAL, "pipeline": {"TR": 1.4, "typo": 1}}, "unknown key"),
    ({**MINIMAL, "physio": {"listener": True}}, "physio.listener"),
    ({**MINIMAL, "pipeline": {"TR": 1.4, "mask": {"analysis_mask": "custom"}}}, "mask.file"),
])
def test_missing_or_invalid_values_are_errors(bad, msg):
    with pytest.raises(C.ConfigError, match=msg):
        C.from_dict(bad)


def test_overrides_file_round_trip_and_case_folding(tmp_path):
    f = tmp_path / "decnef.yaml"
    f.write_text("paths: {base_data: /data}\npipeline: {TR: 1.4, analysis_space: T1, motion: {max_poly_order: inf}}\n"
                 "decoder: {template: dec.nii}\nstages: {baseline_trs: 2, cycle: [[iti, 3], [cue, 4]], n_cycles: 5}\n")
    s = C.load_settings(f, overrides=["pipeline.censor.fd.thr=0.5", "physio.enable=true", "qc.stream_kind=reg"])
    assert s.pipeline.analysis_space == "t1" and s.pipeline.censor.fd.thr == 0.5 and s.physio.enable is True
    assert s.stages.cycle == [["iti", 3], ["cue", 4]] and s.qc.stream_kind == "reg"
    out = tmp_path / "copy.json"
    C.write_config_file(out, s)
    again = C.from_dict(json.loads(out.read_text()))
    assert again.to_dict() == s.to_dict() and again.pipeline.motion.max_poly_order == math.inf


def test_migrate_legacy_settings_covers_every_regressor_setting(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rt_global_settings import RegressorSettings

    old = RegressorSettings().to_json_dict()
    new = C.migrate_legacy(old, base_data="/data", incoming_root="/in", decoder_template="dec.nii")
    s = C.from_dict(new)
    assert s.pipeline.TR == old["TR"] and s.pipeline.censor.dvars.warmup == old["dvars_warmup"]
    assert s.qc.stream_kind == "mc"   # old 'unwarped' alias
    assert s.physio.enable == old["enable_biopac_physio"]
    assert C.migrate_legacy({"made_up": 1})["pipeline"] == {}          # ignored, like the old loader did
    with pytest.raises(C.ConfigError, match="no equivalent"):
        C.migrate_legacy({"made_up": 1}, unknown="error")


# ----------------------------------------------------------------------------- layout
def test_layout_paths_and_no_side_effects(tmp_path):
    lay = RunLayout(tmp_path, "01", "3", "4", ap_block=7, pa_block=8)
    assert lay.run_dir == tmp_path / "sub-01" / "3" / "func" / "4"
    assert lay.fmap_dir.name == "pair-ap007_pa008" and lay.trans_dir == tmp_path / "sub-01/3/func/trans"
    assert lay.mc_nii(12).name == "vol_00012_mc.nii" and lay.space_nii("mni", 3, original=True).name == "vol_00003_mni_orig.nii"
    assert not lay.run_dir.exists()          # properties never create directories
    lay.create_run_dirs()
    assert lay.raw_dir.is_dir() and lay.mc_dir.is_dir()
    assert lay.other_run("9").run_dir.name == "9"
    with pytest.raises(ValueError):
        RunLayout(tmp_path, "01", "3", "4", ap_block=7)


# ----------------------------------------------------------------------------- outputs
def test_csv_writers_match_original_columns(tmp_path):
    outputs.append_score(tmp_path / "scores.csv", 1, 0.5, None, True, 10.0, "cue")
    outputs.append_regression_status(tmp_path / "status.csv", 1, 0.1, 2.0, float("nan"), 0, 0, True, False)
    outputs.append_motion(tmp_path / "motion_rt.1D", np.arange(6, dtype=float))
    rows = list(csv.DictReader(open(tmp_path / "scores.csv")))
    assert list(rows[0]) == ["volume_idx", "timestamp", "score_raw", "score_original", "reg_ready", "event_type"]
    assert rows[0]["reg_ready"] == "1" and rows[0]["event_type"] == "cue"
    assert (tmp_path / "motion_rt.1D").read_text().strip() == "0.000000 1.000000 2.000000 3.000000 4.000000 5.000000"


def test_metadata_merge_is_atomic_and_shallow(tmp_path):
    p = tmp_path / "session_metadata.json"
    outputs.write_metadata(p, {"a": {"x": 1}, "b": 2})
    outputs.write_metadata(p, {"a": {"y": 2}})
    assert outputs.read_metadata(p) == {"a": {"y": 2}, "b": 2}
    assert not (tmp_path / ".session_metadata.json").exists()


def test_reference_score_stats_drop_first_reg_ready(tmp_path):
    for v, ready in [(1, 0), (2, 1), (3, 1), (4, 1)]:
        outputs.append_score(tmp_path / "scores.csv", v, float(v), reg_ready=bool(ready), timestamp=0.0)
        outputs.append_regression_status(tmp_path / "regression_status_rt.csv", v, 0, 0, 0, 0, 0, bool(ready), False)
    stats = outputs.load_reference_score_stats(tmp_path, "7")
    assert stats["n"] == 2 and stats["mean"] == 3.5 and stats["skipped_first_reg_ready"] and stats["run"] == "7"


# ----------------------------------------------------------------------------- decoder
def test_decoder_nifti_and_roi_formats_and_shape_check(tmp_path):
    import nibabel as nib

    w = np.zeros((3, 3, 3), dtype=np.float32); w[1, 1, 1] = 2.0; w[0, 0, 0] = -1.0
    nib.save(nib.Nifti1Image(w, np.eye(4)), tmp_path / "dec.nii")
    vol = np.ones((3, 3, 3), dtype=np.float32) * 3
    assert DecoderScorer(tmp_path / "dec.nii").score_from_array(vol) == 3.0
    roi = tmp_path / "roi.txt"
    roi.write_text("3 3 3\n2\n2 2 2 2.0 0\n1 1 1 -1.0 0\n0 0 0 0.25 0\n")
    assert DecoderScorer(tmp_path / "dec.nii", roi_txt=roi).score_from_array(vol) == 3.25
    with pytest.raises(ValueError, match="does not match decoder grid"):
        DecoderScorer(tmp_path / "dec.nii").score_from_array(np.ones((4, 3, 3)))
    with pytest.raises(FileNotFoundError):
        DecoderScorer(tmp_path / "missing.nii")


# ----------------------------------------------------------------------------- events
def test_score_event_requires_kind_source_and_reg_ready():
    ev = ScoreEvent.from_message({"kind": "score", "source": "pipeline", "volume_idx": 3, "score_raw": 0.1,
                                  "reg_ready": True, "stage": "cue", "extra": 1})
    assert ev.volume_idx == 3 and ev.reg_ready is True and ev.meta == {"extra": 1}
    assert ScoreEvent.from_message(ev.to_message()) == ev
    for bad in ({"volume_idx": 1}, {"kind": "score", "source": "pipeline", "volume_idx": 1}, {"kind": "x", "source": "s"}):
        with pytest.raises(ValueError):
            ScoreEvent.from_message(bad)


def test_stage_schedule_labels_and_windows():
    s = StageSchedule(skip_first_trs=2, baseline_trs=3, cycle=[("iti", 1), ("cue", 2), ("feedback", 1)], n_cycles=2)
    assert [s.label_for_volume(v) for v in range(1, 16)] == (
        ["background"] * 5 + ["iti", "cue", "cue", "feedback", "iti", "cue", "cue", "feedback", "post_task", "post_task"])
    assert s.windows()[0] == {"cycle": 0, "stage": "background", "start_tr": 1, "end_tr": 3}
    assert s.windows()[-1] == {"cycle": 2, "stage": "feedback", "start_tr": 11, "end_tr": 11}
    assert StageSchedule(cycle=[("rest", 1)]).label_for_volume(99) == "rest"
    assert StageSchedule().label_for_volume(5) is None


def test_incoming_watcher_handles_created_closed_and_moved_events(tmp_path):
    """inotify (Linux) reports a file renamed into the folder as moved-to, FSEvents (macOS) as created."""
    from watchdog.events import FileClosedEvent, FileCreatedEvent, FileMovedEvent

    from decnef.ingest import IncomingWatcher

    w = IncomingWatcher(4)
    w.on_created(FileCreatedEvent(str(tmp_path / "001_000004_000002.dcm")))
    w.on_moved(FileMovedEvent(str(tmp_path / ".staging" / "x.dcm"), str(tmp_path / "001_000004_000001.dcm")))
    w.on_closed(FileClosedEvent(str(tmp_path / "001_000004_000002.dcm")))          # duplicate: ignored
    w.on_moved(FileMovedEvent(str(tmp_path / "a"), str(tmp_path / "001_000005_000001.dcm")))   # other run: ignored
    assert w.next_pending()[0] == 1 and w.next_pending()[0] == 2 and w.next_pending() is None
