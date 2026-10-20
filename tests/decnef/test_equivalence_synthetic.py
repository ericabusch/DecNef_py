"""Old pipeline (rt_pipeline.py) vs new package (decnef.pipeline) on a synthetic subject:
the numeric outputs must be identical, for every processing option that can run
without a scanner. See synth.py for what is real and what is stubbed. Skipped
when the old tree is gone (after its removal) or librtp cannot be loaded.
"""
import multiprocessing as mp
from pathlib import Path

import pytest

import synth
from synth import N_VOL, OLD_SETTINGS, REPO

pytestmark = [pytest.mark.skipif(not (REPO / "rt_pipeline.py").exists(), reason="old tree removed"),
              pytest.mark.skipif(not synth.librtp_loads(), reason="librtp (AFNI volreg) cannot be loaded on this machine")]

OUTPUT_FILES = ("scores.csv", "fd_rt.csv", "motion_rt.1D", "regression_status_rt.csv", "dvars_rt.csv")


def _equivalent(tmp_path, old_settings, new_extra, old_cfg_extra=None, prepare=None, runs=("901", "902"), expect_files=OUTPUT_FILES):
    from decnef.cli import compare_dirs
    from decnef.layout import RunLayout

    ctx = mp.get_context("spawn")
    lay, decoder, vols = synth.make_subject(tmp_path)
    if prepare is not None:
        decoder = prepare(tmp_path, lay, decoder) or decoder
    old_run, new_run = runs
    synth.stage_run(lay, old_run, vols)
    seen, msgs, code = synth.run_until(ctx, synth.old_pipeline, (REPO, lay.base_data, lay.incoming_root, decoder, old_settings, old_run, old_cfg_extra), N_VOL)
    assert (len(seen), code) == (N_VOL, 0), f"old pipeline: {len(seen)} volumes, exit {code}"

    synth.stage_run(lay, new_run, vols)
    settings = synth.new_settings(lay, decoder, old_settings, **new_extra)
    lay_new = RunLayout(lay.base_data, "99", "1", new_run, incoming_root=lay.incoming_root)
    seen, msgs, code = synth.run_until(ctx, synth.new_pipeline, (settings, lay_new), N_VOL)
    errors = [m for m in msgs if m.get("kind") == "error"]
    assert (len(seen), errors, code) == (N_VOL, [], 0)

    old_dir = lay.base_data / "sub-99" / "1" / "func" / old_run
    for name in expect_files:
        assert (old_dir / name).exists() and (lay_new.run_dir / name).exists(), name
    assert len((lay_new.run_dir / "scores.csv").read_text().splitlines()) == N_VOL + 1
    problems = compare_dirs(old_dir, lay_new.run_dir)
    assert problems == [], "\n".join(problems)
    return old_dir, lay_new.run_dir, msgs


def test_baseline_epi_space(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    _, new_dir, msgs = _equivalent(tmp_path, OLD_SETTINGS, {})
    assert (new_dir / "regressors_rt.csv").exists()
    assert all(m["kind"] == "score" and "reg_ready" in m and m["source"] == "pipeline" for m in msgs)
    ready = [m["reg_ready"] for m in msgs]
    assert ready[0] is False and ready[-1] is True, "regression must become active within the run for the check to mean anything"


def test_final_masking_and_smoothing(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    old = {**OLD_SETTINGS, "smoothing_fwhm_mm": 4.0}
    _, new_dir, _ = _equivalent(tmp_path, old, {})
    assert (new_dir / "smooth" / "mask.nii").exists() and (new_dir / "smooth" / "vol_00001_smooth.nii").exists()


def test_regression_off_voxel_normalizer(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    old = {**OLD_SETTINGS, "enable_motion_regression": False}
    _equivalent(tmp_path, old, {}, expect_files=("scores.csv", "fd_rt.csv", "motion_rt.1D", "regression_status_rt.csv"))


def test_original_score(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    _, new_dir, msgs = _equivalent(tmp_path, OLD_SETTINGS, {"decoder.enable_original_score": True},
                                   old_cfg_extra={"enable_original_score": True})
    assert all(m.get("score_original") is not None for m in msgs)


def test_zscore_reference_run(tmp_path, monkeypatch):
    """Second pair of runs z-scored against the first pair's scores.csv (old 901 -> 903, new 902 -> 904)."""
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    from decnef.cli import compare_dirs
    from decnef.layout import RunLayout

    ctx = mp.get_context("spawn")
    lay, decoder, vols = synth.make_subject(tmp_path)
    for run, target, args in (("901", synth.old_pipeline, (REPO, lay.base_data, lay.incoming_root, decoder, OLD_SETTINGS, "901", None)),
                              ("902", synth.new_pipeline, (synth.new_settings(lay, decoder), RunLayout(lay.base_data, "99", "1", "902", incoming_root=lay.incoming_root)))):
        synth.stage_run(lay, run, vols)
        seen, _, code = synth.run_until(ctx, target, args, N_VOL)
        assert (len(seen), code) == (N_VOL, 0)
    synth.stage_run(lay, "903", vols)
    seen, _, code = synth.run_until(ctx, synth.old_pipeline, (REPO, lay.base_data, lay.incoming_root, decoder, OLD_SETTINGS, "903", {"reference_score_run": "901"}), N_VOL)
    assert (len(seen), code) == (N_VOL, 0)
    synth.stage_run(lay, "904", vols)
    settings = synth.new_settings(lay, decoder, **{"decoder.reference_score_run": "902"})
    seen, msgs, code = synth.run_until(ctx, synth.new_pipeline, (settings, RunLayout(lay.base_data, "99", "1", "904", incoming_root=lay.incoming_root)), N_VOL)
    assert (len(seen), code) == (N_VOL, 0) and all("score_z" in m for m in msgs)
    old_dir, new_dir = lay.base_data / "sub-99/1/func/903", lay.base_data / "sub-99/1/func/904"
    assert (old_dir / "scores_z.csv").exists() and (new_dir / "scores_z.csv").exists()
    problems = [p for p in compare_dirs(old_dir, new_dir) if "ref_run" not in p]   # the reference run ids differ by construction
    assert problems == [], "\n".join(problems)


def test_physio_file_mode(tmp_path, monkeypatch):
    """RICOR8 regressors from a CSV (as the listener writes it), one volume missing -> zero-filled."""
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    physio_csv = tmp_path / "biopac_regressors_rx.csv"
    synth.write_physio_csv(physio_csv, N_VOL, missing=(6,))
    old = {**OLD_SETTINGS, "enable_biopac_physio": True, "biopac_mode": "file", "biopac_file": str(physio_csv),
           "biopac_phys_reg": "RICOR8", "biopac_timeout": 0.05, "biopac_poll_interval": 0.01}
    _, new_dir, _ = _equivalent(tmp_path, old, {})
    header = (new_dir / "regressors_rt.csv").read_text().splitlines()[0]
    assert header.count("RICOR") == 8 or header.count("ricor") == 8 or len(header.split(",")) > 7


def test_mni_space_via_gpu_resampler_on_cpu(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    old = {**OLD_SETTINGS, "analysis_space": "mni", "use_gpu_resampler": True, "gpu_resampler_device": "cpu"}
    _, new_dir, _ = _equivalent(tmp_path, old, {}, prepare=lambda root, lay, dec: synth.add_mni_grid(root, lay))
    assert (new_dir / "mni" / "vol_00001_mni.nii").exists()
    import nibabel as nib
    assert nib.load(str(new_dir / "mni" / "vol_00001_mni.nii")).shape == synth.MNI_SHAPE
