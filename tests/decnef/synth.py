"""Synthetic subject + helpers shared by the equivalence and behaviour tests.

Raw volumes are written directly as `raw/vol_<v>.nii` (both pipelines then skip
dcm2niix; the `.dcm` files only need their names), `analysis_space=epi` needs
no ANTs, and the T1/MNI path is covered through the GPU resampler with a
synthetic sampling grid. Process targets are module-level so `spawn` can
import them.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SHAPE = (24, 24, 12)
AFF = np.diag([3.0, 3.0, 3.0, 1.0])
MNI_SHAPE = (20, 20, 10)
MNI_AFF = np.diag([2.0, 2.0, 2.0, 1.0])
N_VOL = 20   # enough volumes for RtpRegress to pass its warm-up so regression really runs

OLD_SETTINGS = dict(TR=1.5, analysis_space="epi", voxel_norm_ref_volumes=3, skip_first_trs=1, enable_biopac_physio=False,
                    fd_thr=0.3, dvars_warmup=3, enable_volume_streamer=False, max_workers=2, pyhysco_device="cpu",
                    use_gpu_resampler=False, save_intermediate_unwarped=True, smoothing_fwhm_mm=0.0)


def librtp_loads() -> bool:
    try:
        from decnef.rtpspy import rtp_volreg

        rtp_volreg.RtpVolreg(regmode="heptic")
        return True
    except Exception:
        return False


def install_dcm2niix_stub(tmp_path, monkeypatch) -> None:
    """A dcm2niix on PATH for the prerequisite check; never called (raw volumes pre-exist)."""
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "dcm2niix").write_text("#!/bin/sh\nexit 1\n")
    (shim / "dcm2niix").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")


# ----------------------------------------------------------------------------- subject
def make_subject(root: Path, n_vol: int = N_VOL):
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
    for v in range(1, n_vol + 1):   # drift + small shifts + noise
        shift = int(round(0.6 * np.sin(v / 2.0)))
        data = np.roll(base, shift, axis=0) * (1 + 0.002 * v) + rng.normal(0, 5, SHAPE).astype(np.float32)
        data[~brain] = 20 + rng.normal(0, 2, SHAPE)[~brain]
        vols.append(data.astype(np.float32))
    return lay, root / "decoder.nii", vols


def add_mni_grid(root: Path, lay):
    """An 'MNI' decoder on another grid, dummy transform files, and a cached sampling grid
    (a known voxel shift) so both pipelines can warp with the GPU resampler on CPU."""
    mx, my, mz = np.meshgrid(*[np.arange(s) for s in MNI_SHAPE], indexing="ij")
    dec = np.zeros(MNI_SHAPE, dtype=np.float32)
    dec[(mx > 8) & (my > 3) & (my < 16) & (mz > 1) & (mz < 8)] = 1e-3
    dec[(mx <= 8) & (my > 3) & (my < 16) & (mz > 1) & (mz < 8)] = -5e-4
    decoder = root / "decoder_mni.nii"
    nib.save(nib.Nifti1Image(dec, MNI_AFF), decoder)
    lay.t1_to_mni_warp.write_bytes(b"dummy")           # existence is all the cached-grid path checks
    lay.epi2t1_composite.write_bytes(b"dummy")
    out_dir = lay.gpu_grids_dir / "epi_to_mni"
    out_dir.mkdir(parents=True)
    # fixed voxel (i,j,k) samples moving voxel (i+2, j+1, k) -> normalized grid_sample coords, [Z,Y,X] layout, [x,y,z] components
    cx, cy, cz = (mx + 2).astype(np.float32), (my + 1).astype(np.float32), mz.astype(np.float32)
    gx = 2.0 * cx / (SHAPE[0] - 1) - 1.0
    gy = 2.0 * cy / (SHAPE[1] - 1) - 1.0
    gz = 2.0 * cz / (SHAPE[2] - 1) - 1.0
    grid = np.transpose(np.stack([gx, gy, gz], axis=-1), (2, 1, 0, 3))[None].astype(np.float32)
    np.savez_compressed(out_dir / "sampling_grid.npz", grid=grid, moving_shape=np.asarray(SHAPE, dtype=np.int32),
                        fixed_shape=np.asarray(MNI_SHAPE, dtype=np.int32), fixed_affine=MNI_AFF.astype(np.float64))
    return decoder


def write_physio_csv(path: Path, n_vol: int, missing=()):
    """RICOR8 regressors per volume, like the listener writes them; `missing` volumes are absent."""
    rng = np.random.default_rng(7)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["volume_idx", "timestamp"] + [f"reg_{i:02d}" for i in range(1, 9)])
        for v in range(1, n_vol + 1):
            if v in missing:
                continue
            w.writerow([v, f"{1000 + 1.5 * v:.6f}"] + [f"{x:.6f}" for x in rng.normal(0, 1, 8)])


def stage_run(lay, run: str, vols, incoming=None):
    rd = lay.base_data / "sub-99" / "1" / "func" / run / "raw"
    rd.mkdir(parents=True)
    for v, data in enumerate(vols, 1):
        nib.save(nib.Nifti1Image(data, AFF), rd / f"vol_{v:05d}.nii")
    if incoming is not False:
        drop_dicoms(lay, run, range(1, len(vols) + 1))


def drop_dicoms(lay, run: str, volumes):
    for v in volumes:
        (lay.incoming_root / f"001_{int(run):06d}_{v:06d}.dcm").write_bytes(b"placeholder")


# ----------------------------------------------------------------------------- process targets
def old_pipeline(repo, base_data, incoming, decoder, settings_dict, run, extra_cfg, q, stop):
    sys.path.insert(0, str(repo))
    from rt_pipeline import REGRESSOR_SETTINGS, RTSessionConfig, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings_dict)
    cfg = RTSessionConfig(subject="99", day="1", run=run, incoming_root=incoming, base_data=base_data, decoder_template=decoder,
                          **(extra_cfg or {}))
    run_rt_pipeline(cfg, q, stop_event=stop)


def new_pipeline(settings_dict, layout, q, stop):
    from decnef.config import from_dict
    from decnef.pipeline import run_pipeline

    run_pipeline(from_dict(settings_dict), layout, q, stop)


def run_until(ctx, target, args, n, timeout=180, on_tick=None):
    """Start `target(*args, q, stop)`, collect messages until `n` distinct volumes (or exit), stop, join.
    `on_tick(seen)` is called after each message (used to drop DICOMs during the run)."""
    q, stop = ctx.Queue(), ctx.Event()
    p = ctx.Process(target=target, args=(*args, q, stop))
    p.start()
    seen, messages, t0 = set(), [], time.time()
    try:
        while len(seen) < n and time.time() - t0 < timeout and (p.is_alive() or not q.empty()):
            try:
                m = q.get(timeout=0.5)
            except Exception:
                if on_tick is not None:
                    on_tick(seen)
                continue
            messages.append(m)
            if m.get("volume_idx"):
                seen.add(m["volume_idx"])
            if on_tick is not None:
                on_tick(seen)
    finally:
        stop.set()
        p.join(60)
    return seen, messages, p.exitcode


def new_settings(lay, decoder, old_settings=OLD_SETTINGS, **extra):
    from decnef.config import migrate_legacy

    data = migrate_legacy(old_settings, base_data=lay.base_data, incoming_root=lay.incoming_root, decoder_template=decoder)
    for key, value in extra.items():
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data
