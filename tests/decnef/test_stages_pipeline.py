"""Stages and process_volume on synthetic data (no dcm2niix, no PyHySCO, stub volreg):
the numeric helpers, censor logic, the EPI-space path end to end, provenance and
prerequisite checks."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from decnef import calibration, config as C, outputs, pipeline, stages
from decnef.events import StageSchedule
from decnef.layout import RunLayout

SHAPE = (6, 6, 4)


# ----------------------------------------------------------------------------- fixtures
class StubVolreg:
    """Pretends every volume is shifted by `shift` mm along dS; leaves data unchanged."""

    def __init__(self):
        self._motion = []

    def do_proc(self, img, vol_idx):
        self._motion.append(np.array([0, 0, 0, 0.1 * vol_idx, 0, 0], dtype=float))


class IdentityApplier:
    def apply_volume(self, data):
        return np.asarray(data, dtype=np.float32) * 1.0


def make_subject(tmp_path, with_calibration=True, decoder=True):
    lay = RunLayout(tmp_path / "data", "01", "1", "4", incoming_root=tmp_path / "incoming")
    lay.incoming_root.mkdir(parents=True)
    lay.trans_dir.mkdir(parents=True)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    ref = np.full(SHAPE, 100.0, dtype=np.float32)
    nib.save(nib.Nifti1Image(ref, aff), lay.trans_dir / "epi_unwarped_mean.nii")
    mask = np.zeros(SHAPE, dtype=np.uint8); mask[1:5, 1:5, 1:3] = 1
    nib.save(nib.Nifti1Image(mask, aff), lay.trans_dir / "epi_mask_mean.nii")
    if with_calibration:
        fm = lay.fmap_dir; fm.mkdir(parents=True)
        for name in ("AP", "PA"):
            nib.save(nib.Nifti1Image(ref, aff), fm / f"{name}.nii")
            (fm / f"{name}.json").write_text(json.dumps({"PhaseEncodingDirection": "j" if name == "AP" else "j-", "TotalReadoutTime": 0.05}))
        native, field, manifest = calibration.native_paths(fm)
        native.mkdir()
        for name in ("AP", "PA"):
            calibration.save_native_mean(fm / f"{name}.nii", native / f"{name}_mean.nii")
        nib.save(nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), aff), field)
        manifest.write_text(json.dumps(calibration.calibration_spec(fm / "AP.nii", fm / "PA.nii")))
    dec = None
    if decoder:
        w = np.zeros(SHAPE, dtype=np.float32); w[2, 2, 1] = 1.0; w[3, 3, 1] = 1.0
        dec = tmp_path / "dec.nii"
        nib.save(nib.Nifti1Image(w, aff), dec)
    return lay, dec


def settings_for(lay, dec, **pipeline_over):
    data = {"paths": {"base_data": str(lay.base_data), "incoming_root": str(lay.incoming_root)},
            "pipeline": {"TR": 1.0, "analysis_space": "epi", "voxel_norm_ref_volumes": 2, "skip_first_trs": 1,
                         "motion": {"enable_regression": False},
                         "censor": {"fd": {"thr": 0.2}, "dvars": {"warmup": 2}}, **pipeline_over},
            "decoder": {"template": str(dec), "enable_scoring": dec is not None},
            "stages": {"baseline_trs": 1, "cycle": [["cue", 1], ["fb", 1]], "n_cycles": 2}}
    return C.from_dict(data)


def make_state(lay, settings, dec):
    ref_img = nib.load(str(lay.motion_ref_epi))
    ref_mask = np.asanyarray(nib.load(str(lay.motion_ref_mask)).dataobj) > 0.5
    lay.create_run_dirs()
    from decnef.decoder import DecoderScorer
    return stages.RunState(settings=settings, layout=lay, decoder_template=dec, ref_affine=ref_img.affine,
                           volreg=StubVolreg(), applier=IdentityApplier(),
                           voxel_normalizer=stages.VoxelNormalizer(settings.pipeline.voxel_norm_ref_volumes, ref_mask),
                           scorer=DecoderScorer(dec) if dec else None, score_tracker=StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs),
                           dvars_mask=ref_mask)


def raw_volume(lay, v, value):
    lay.raw_dir.mkdir(parents=True, exist_ok=True)
    p = lay.raw_nii(v)
    data = np.full(SHAPE, 100.0, dtype=np.float32); data[2, 2, 1] = value; data[3, 3, 1] = value
    nib.save(nib.Nifti1Image(data, nib.load(str(lay.motion_ref_epi)).affine), p)
    return p


# ----------------------------------------------------------------------------- numeric helpers
def test_fd_dvars_robust_z():
    assert stages.fd_from_rtpspy_delta([0, 0, 0, 1, 2, 3], 50.0) == 6.0
    assert np.isclose(stages.fd_from_rtpspy_delta([1, 0, 0, 0, 0, 0], 50.0), np.deg2rad(1) * 50)
    a, b = np.zeros((2, 2, 2)), np.ones((2, 2, 2)) * 2
    assert stages.compute_dvars(a, b, None) == 2.0
    assert np.isnan(stages.robust_z(1.0, [1, 2, 3])) and np.isnan(stages.robust_z(1.0, [1, 1, 1, 1, 1]))
    assert np.isclose(stages.robust_z(3.0, [0, 1, 2, 3, 4]), (3 - 2) / (1.4826 * 1))


def test_voxel_normalizer_percent_signal_and_clip():
    vn = stages.VoxelNormalizer(ref_volumes=2, brain_mask=np.array([[[True, False]]]))
    out1 = vn.apply(np.array([[[50.0, 50.0]]]))
    assert out1.tolist() == [[[100.0, 0.0]]]
    out2 = vn.apply(np.array([[[150.0, 50.0]]]))          # mean of 50,150 = 100 -> 150%
    assert out2.tolist() == [[[150.0, 0.0]]]
    assert vn.apply(np.array([[[900.0, 50.0]]])).tolist() == [[[200.0, 0.0]]]   # clipped, mean frozen


def test_censor_flags_with_plus_one(tmp_path):
    lay, dec = make_subject(tmp_path)
    st = make_state(lay, settings_for(lay, dec), dec)
    mc = np.zeros(SHAPE, dtype=np.float32)
    assert stages.censor_stage(st, mc, 1, 0.5, float("nan")) == (0, 0)        # NaN FD (skipped TR): no FD hit
    assert stages.censor_stage(st, mc, 2, 0.5, 0.5) == (1, 0)                 # FD hit
    assert stages.censor_stage(st, mc, 3, 0.0, 0.0) == (1, 0)                 # +1 TR
    assert stages.censor_stage(st, mc, 4, 0.0, 0.0) == (0, 0)
    assert len(st.dvars_hist) == 4 and st.dvars_hist[0] == 0.0               # first volume contributes 0.0


# ----------------------------------------------------------------------------- end to end (EPI space)
def test_process_volume_epi_space_scores_and_labels(tmp_path):
    lay, dec = make_subject(tmp_path)
    settings = settings_for(lay, dec)
    st = make_state(lay, settings, dec)
    messages = [pipeline.process_volume(st, raw_volume(lay, v, 100.0 + v), v, 10.0 + v) for v in (1, 2, 3, 4)]
    assert [m["kind"] for m in messages] == ["score"] * 4
    assert [m["stage"] for m in messages] == ["background", "background", "cue", "fb"]   # skip 1, baseline 1, then cycle
    assert all(m["reg_ready"] for m in messages)
    # score = sum over the 2 decoder voxels of the percent-signal value (voxel normaliser, ref mean of vols 1-2)
    assert messages[1]["score_raw"] == pytest.approx(2 * 102 / 101.5 * 100)
    rows = list(csv.DictReader(open(lay.scores_csv)))
    assert [r["event_type"] for r in rows] == ["background", "background", "cue", "fb"]
    fd = list(csv.DictReader(open(lay.fd_csv)))
    assert fd[0]["fd"] == "nan" and float(fd[1]["fd"]) == pytest.approx(0.1)   # stub volreg: 0.1 mm per volume
    assert lay.mc_nii(3).exists() and lay.reg_nii(3).exists() and lay.unwarped_nii(3).exists()
    status = list(csv.DictReader(open(lay.regression_status_csv)))
    assert [r["reg_ready"] for r in status] == ["1"] * 4


def test_scoring_off_yields_ticks(tmp_path):
    lay, dec = make_subject(tmp_path, decoder=False)
    settings = settings_for(lay, None)
    st = make_state(lay, settings, None)
    m = pipeline.process_volume(st, raw_volume(lay, 1, 100.0), 1, 10.0)
    assert m["kind"] == "tick" and m["volume_idx"] == 1 and "score_raw" not in m and not lay.scores_csv.exists()


def test_bad_grid_is_an_error_not_a_zero_motion_row(tmp_path):
    lay, dec = make_subject(tmp_path)
    st = make_state(lay, settings_for(lay, dec), dec)
    bad = lay.raw_nii(1)
    lay.raw_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), bad)
    with pytest.raises(ValueError, match="grids differ"):
        pipeline.process_volume(st, bad, 1, 0.0)
    assert not lay.motion_1d.exists()


# ----------------------------------------------------------------------------- run-level checks
def test_prerequisites_and_provenance(tmp_path, monkeypatch):
    lay, dec = make_subject(tmp_path)
    settings = settings_for(lay, dec)
    monkeypatch.setattr(pipeline, "require", lambda *names: {n: n for n in names})
    assert pipeline.check_prerequisites(settings, lay) == dec
    pipeline.validate_run_provenance(settings, lay, dec)
    assert lay.provenance_json.exists()
    pipeline.validate_run_provenance(settings, lay, dec)             # same settings: fine
    changed = settings_for(lay, dec, fieldmap={"epi_phase_encoding": "AP"})
    with pytest.raises(pipeline.PipelineError, match="fresh run"):
        pipeline.validate_run_provenance(changed, lay, dec)

    missing_cal, _ = make_subject(tmp_path / "b", with_calibration=False)
    with pytest.raises(pipeline.PipelineError, match="calibration"):
        pipeline.check_prerequisites(settings_for(missing_cal, dec), missing_cal)
    no_dec = settings_for(lay, None)
    no_dec.pipeline.analysis_space = "mni"
    with pytest.raises(pipeline.PipelineError, match="resampling reference"):
        pipeline.check_prerequisites(no_dec, lay)


def test_metadata_payload_has_pipeline_keys_and_full_settings(tmp_path):
    lay, dec = make_subject(tmp_path)
    settings = settings_for(lay, dec)
    payload = outputs.session_metadata_payload(settings, lay, dec, None, calibration.PIPELINE_ORDER)
    assert payload["regression"]["analysis_space"] == "epi" and payload["settings"]["pipeline"]["TR"] == 1.0
    assert payload["tr"] == 1.0 and payload["decoder_template"] == str(dec)
