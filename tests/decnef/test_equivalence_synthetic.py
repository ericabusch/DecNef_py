"""Old pipeline (rt_pipeline.py) vs new package (decnef.pipeline) on a synthetic subject:
the numeric outputs must be identical.

What makes it runnable without a scanner: raw volumes are written directly as
`raw/vol_<v>.nii` (both pipelines then skip dcm2niix; the `.dcm` files only
need their names, and a stub `dcm2niix` on PATH satisfies the prerequisite
check); `analysis_space=epi` needs no ANTs; physio and the viewer are off.
What is real: PyHySCO unwarping (small synthetic field on the staggered grid),
RtpVolreg (librtp), RtpRegress with WM/Vent regressors, FD/DVARS censoring,
the decoder. Skipped when the old tree is not present (after its removal) or
librtp cannot be loaded on this machine.
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SHAPE = (24, 24, 12)
AFF = np.diag([3.0, 3.0, 3.0, 1.0])
N_VOL = 10

pytestmark = pytest.mark.skipif(not (REPO / "rt_pipeline.py").exists(), reason="old tree removed")


def _librtp_loads() -> bool:
    try:
        from decnef.rtpspy import rtp_volreg

        rtp_volreg.RtpVolreg(regmode="heptic")
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- synthetic subject
def make_subject(root: Path):
    from decnef import calibration
    from decnef.layout import RunLayout

    lay = RunLayout(root / "data", "99", "1", "1", incoming_root=root / "incoming")
    lay.trans_dir.mkdir(parents=True)
    lay.anat_dir.mkdir(parents=True)
    lay.incoming_root.mkdir()
    rng = np.random.default_rng(0)
    x, y, z = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    brain = ((x - 11.5) ** 2 / 100 + (y - 11.5) ** 2 / 100 + (z - 5.5) ** 2 / 25) < 1.0
    base = np.where(brain, 1000.0 + 200 * np.exp(-((x - 11.5) ** 2 + (y - 11.5) ** 2) / 60), 20.0).astype(np.float32)
    nib.save(nib.Nifti1Image(base, AFF), lay.trans_dir / "epi_unwarped_mean.nii")
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), AFF), lay.trans_dir / "epi_mask_mean.nii")
    wm = brain & (np.abs(x - 11.5) < 3) & (np.abs(y - 11.5) < 3)
    vent = brain & (z <= 3) & (np.abs(x - 11.5) < 2)
    nib.save(nib.Nifti1Image(wm.astype(np.uint8), AFF), lay.wm_mask)
    nib.save(nib.Nifti1Image(vent.astype(np.uint8), AFF), lay.vent_mask)
    fm = lay.fmap_dir
    fm.mkdir(parents=True)
    for name in ("AP", "PA"):
        nib.save(nib.Nifti1Image(base, AFF), fm / f"{name}.nii")
        (fm / f"{name}.json").write_text(json.dumps({"PhaseEncodingDirection": "j" if name == "AP" else "j-", "TotalReadoutTime": 0.05}))
    native, field, manifest = calibration.native_paths(fm)
    native.mkdir()
    for name in ("AP", "PA"):
        calibration.save_native_mean(fm / f"{name}.nii", native / f"{name}_mean.nii")
    # PyHySCO keeps the field on the staggered grid: one extra sample along the PE axis ('j' -> axis index 1)
    fshape = (SHAPE[0], SHAPE[1] + 1, SHAPE[2])
    fx, fy, fz = np.meshgrid(*[np.arange(s) for s in fshape], indexing="ij")
    fbrain = ((fx - 11.5) ** 2 / 100 + (fy - 12) ** 2 / 100 + (fz - 5.5) ** 2 / 25) < 1.0
    nib.save(nib.Nifti1Image((0.3 * np.sin(fx / 4.0) * fbrain).astype(np.float32), AFF), field)
    manifest.write_text(json.dumps(calibration.calibration_spec(fm / "AP.nii", fm / "PA.nii")))
    dec = np.zeros(SHAPE, dtype=np.float32)
    dec[brain & (x > 11)] = 1.0 / brain.sum()
    dec[brain & (x <= 11)] = -0.5 / brain.sum()
    nib.save(nib.Nifti1Image(dec, AFF), root / "decoder.nii")
    vols = []
    for v in range(1, N_VOL + 1):   # drift + small shifts + noise
        shift = int(round(0.6 * np.sin(v / 2.0)))
        data = np.roll(base, shift, axis=0) * (1 + 0.002 * v) + rng.normal(0, 5, SHAPE).astype(np.float32)
        data[~brain] = 20 + rng.normal(0, 2, SHAPE)[~brain]
        vols.append(data.astype(np.float32))
    return lay, root / "decoder.nii", vols


def stage_run(lay, run: str, vols):
    rd = lay.base_data / "sub-99" / "1" / "func" / run / "raw"
    rd.mkdir(parents=True)
    for v, data in enumerate(vols, 1):
        nib.save(nib.Nifti1Image(data, AFF), rd / f"vol_{v:05d}.nii")
        (lay.incoming_root / f"001_{int(run):06d}_{v:06d}.dcm").write_bytes(b"placeholder")


# ----------------------------------------------------------------------------- process targets
def old_pipeline(repo, base_data, incoming, decoder, settings_dict, run, q, stop):
    sys.path.insert(0, str(repo))
    from rt_pipeline import REGRESSOR_SETTINGS, RTSessionConfig, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings_dict)
    cfg = RTSessionConfig(subject="99", day="1", run=run, incoming_root=incoming, base_data=base_data, decoder_template=decoder)
    run_rt_pipeline(cfg, q, stop_event=stop)


def new_pipeline(settings_dict, layout, q, stop):
    from decnef.config import from_dict
    from decnef.pipeline import run_pipeline

    run_pipeline(from_dict(settings_dict), layout, q, stop)


def _run(ctx, target, args, n, timeout=180):
    q, stop = ctx.Queue(), ctx.Event()
    p = ctx.Process(target=target, args=(*args, q, stop))
    p.start()
    seen, errors, t0 = set(), [], time.time()
    while len(seen) < n and time.time() - t0 < timeout and (p.is_alive() or not q.empty()):
        try:
            m = q.get(timeout=0.5)
        except Exception:
            continue
        if m.get("kind") == "error":
            errors.append(m)
        if m.get("volume_idx"):
            seen.add(m["volume_idx"])
    stop.set()
    p.join(60)
    return len(seen), errors, p.exitcode


OLD_SETTINGS = dict(TR=1.5, analysis_space="epi", voxel_norm_ref_volumes=3, skip_first_trs=1, enable_biopac_physio=False,
                    fd_thr=0.3, dvars_warmup=3, enable_volume_streamer=False, max_workers=2, pyhysco_device="cpu",
                    use_gpu_resampler=False, save_intermediate_unwarped=True, smoothing_fwhm_mm=0.0)


@pytest.mark.skipif(not _librtp_loads(), reason="librtp (AFNI volreg) cannot be loaded on this machine")
def test_old_and_new_pipelines_produce_identical_outputs(tmp_path, monkeypatch):
    from decnef.cli import compare_dirs
    from decnef.config import migrate_legacy
    from decnef.layout import RunLayout

    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "dcm2niix").write_text("#!/bin/sh\nexit 1\n")   # present for the prerequisite check, never called
    (shim / "dcm2niix").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    ctx = mp.get_context("spawn")
    lay, decoder, vols = make_subject(tmp_path)

    stage_run(lay, "901", vols)
    n_old, err_old, code_old = _run(ctx, old_pipeline, (REPO, lay.base_data, lay.incoming_root, decoder, OLD_SETTINGS, "901"), N_VOL)
    assert (n_old, code_old) == (N_VOL, 0), f"old pipeline: {n_old} volumes, exit {code_old}, errors {err_old}"

    stage_run(lay, "902", vols)
    new = migrate_legacy(OLD_SETTINGS, base_data=lay.base_data, incoming_root=lay.incoming_root, decoder_template=decoder)
    lay2 = RunLayout(lay.base_data, "99", "1", "902", incoming_root=lay.incoming_root)
    n_new, err_new, code_new = _run(ctx, new_pipeline, (new, lay2), N_VOL)
    assert (n_new, err_new, code_new) == (N_VOL, [], 0)

    old_dir = lay.base_data / "sub-99" / "1" / "func" / "901"
    for name in ("scores.csv", "fd_rt.csv", "motion_rt.1D", "regression_status_rt.csv", "dvars_rt.csv", "regressors_rt.csv"):
        assert (old_dir / name).exists() and (lay2.run_dir / name).exists(), name
    assert len((lay2.run_dir / "scores.csv").read_text().splitlines()) == N_VOL + 1
    problems = compare_dirs(old_dir, lay2.run_dir)
    assert problems == [], "\n".join(problems)
