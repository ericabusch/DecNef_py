"""EPI-space preparation without a T1: what a phantom (or a quick pilot) needs to run the real-time
pipeline in `analysis_space: epi`.

From one incoming folder it builds, under `sub-<sub>/<day>/`:

    fmap/pair-apAAA_paPPP/AP.nii.gz, PA.nii.gz (+ .json)      the fieldmap pair, converted with dcm2niix
    fmap/pair-apAAA_paPPP/native_unwarp_v1/                      the PyHySCO calibration (calibration.build_native_calibration)
    func/trans/transfer_raw/vol_000NN.nii                        the kept volumes of the transfer EPI block
    func/trans/epi_mc.nii, motion.1D                             transfer volumes unwarped, then motion-corrected to the first one
    func/trans/epi_unwarped_mean.nii (+ rt_ref_epi.nii)          the motion/analysis reference (mean of the corrected volumes)
    func/trans/epi_mask_mean.nii (+ rt_ref_epi_mask.nii)         intensity-threshold mask (no SynthStrip: a phantom has no skull)
    func/trans/phantom_decoder.nii                               optional: random weights inside the mask, so scoring can run

Same order as the full preparation (`fmri_rt_preproc/pipeline.py`: unwarp -> MC to the first corrected
volume -> mean), same RtpVolreg settings as the real-time run. Not produced: WM/ventricle masks
(FastSurfer), T1/MNI transforms - set `nuisance.use_wm/use_vent: false` and `analysis_space: epi`.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from . import calibration
from .config import Settings
from .ingest import convert_dicom, parse_dicom_name
from .layout import RunLayout
from .tools import require, run

log = logging.getLogger(__name__)


def block_dicoms(incoming: Path, block: int) -> list[Path]:
    """This series' DICOMs in scan order (`SSS_RRRRRR_VVVVVV.dcm`, RRRRRR == block)."""
    found = []
    for p in Path(incoming).iterdir():
        parsed = parse_dicom_name(p.name)
        if parsed is not None and parsed[1] == int(block):
            found.append((parsed[2], p))
    return [p for _, p in sorted(found)]


def convert_block(incoming: Path, block: int, dest: Path, log_=log) -> Path:
    """All DICOMs of one series -> one NIfTI at `dest` (e.g. fmap/AP.nii.gz) with its dcm2niix sidecar.
    An existing `dest` (or its .nii / .nii.gz twin) is kept."""
    dest = Path(dest)
    twin = dest.with_suffix("") if dest.suffix == ".gz" else dest.with_suffix(".nii.gz")
    for existing in (dest, twin):
        if existing.exists():
            log_.info("[PREP] %s exists; not converting block %d again", existing, block)
            return existing
    dicoms = block_dicoms(incoming, block)
    if not dicoms:
        raise FileNotFoundError(f"no DICOMs for series {block} in {incoming}")
    prefix = dest.name.removesuffix(".gz").removesuffix(".nii")
    staging = dest.parent / f".{prefix}_dicoms"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for src in dicoms:
        shutil.copy2(src, staging / src.name)
    run(["dcm2niix", "-z", "y", "-b", "y", "-f", prefix, "-o", str(staging), str(staging)])
    produced = sorted(staging.glob(f"{prefix}*.nii*"))
    if not produced:
        raise FileNotFoundError(f"dcm2niix produced nothing for series {block} ({len(dicoms)} files)")
    chosen = produced[0]
    sidecar = chosen.with_name(chosen.name.removesuffix(".gz").removesuffix(".nii") + ".json")
    if sidecar.exists():
        shutil.copy2(sidecar, dest.parent / f"{prefix}.json")
    shutil.move(str(chosen), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    log_.info("[PREP] series %d (%d DICOMs) -> %s", block, len(dicoms), dest)
    return dest


def convert_epi_block(incoming: Path, block: int, out_dir: Path, keep: tuple[int, int] = (11, 30), log_=log) -> list[Path]:
    """The transfer EPI: scans keep[0]..keep[1] of the series, one NIfTI per volume (`vol_000NN.nii`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = keep
    dicoms = [(parse_dicom_name(p.name)[2], p) for p in block_dicoms(incoming, block)]
    dicoms = [(scan, p) for scan, p in dicoms if first <= scan <= last]
    if not dicoms:
        raise FileNotFoundError(f"no DICOMs of series {block} with scan numbers {first}..{last} in {incoming}")
    if len(dicoms) < last - first + 1:
        log_.warning("[PREP] transfer series %d: only %d of %d requested volumes present", block, len(dicoms), last - first + 1)
    vols = [convert_dicom(p, out_dir, scan) for scan, p in dicoms]
    log_.info("[PREP] transfer series %d: %d volumes -> %s", block, len(vols), out_dir)
    return vols


def intensity_mask(mean: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Voxels above `threshold` x the 98th percentile of the mean image (no anatomy assumed)."""
    high = float(np.percentile(mean[np.isfinite(mean)], 98))
    return (mean > threshold * high).astype(np.uint8)


def random_decoder(mask: np.ndarray, affine, out: Path, seed: int = 0) -> Path:
    """Zero-mean random weights inside the mask: a stand-in decoder so the scoring stage runs."""
    rng = np.random.default_rng(seed)
    w = np.zeros(mask.shape, dtype=np.float32)
    inside = mask > 0
    w[inside] = rng.standard_normal(int(inside.sum())).astype(np.float32) / max(int(inside.sum()), 1)
    nib.save(nib.Nifti1Image(w, affine), str(out))
    return out


def prepare_epi(settings: Settings, layout: RunLayout, incoming: Path, ap_block: int, pa_block: int, epi_block: int, *,
                keep: tuple[int, int] = (11, 30), mask_threshold: float = 0.5, decoder: str = "random",
                max_iter: int = 50, log_=log) -> dict:
    """Build the EPI-space prerequisites for `layout`'s subject/day from `incoming`. Idempotent: existing
    products are kept (delete `func/trans/` or the pair folder to rebuild). Returns the paths."""
    import torch

    from .unwarp import PreloadedApplier

    require("dcm2niix")
    p = settings.pipeline
    lay = RunLayout(layout.base_data, layout.sub, layout.day, layout.run, incoming_root=Path(incoming), ap_block=ap_block, pa_block=pa_block)
    fm, trans = lay.fmap_dir, lay.trans_dir
    fm.mkdir(parents=True, exist_ok=True)
    trans.mkdir(parents=True, exist_ok=True)
    lay.anat_dir.mkdir(parents=True, exist_ok=True)

    # 1. fieldmap pair + calibration
    convert_block(incoming, ap_block, fm / "AP.nii.gz", log_)
    convert_block(incoming, pa_block, fm / "PA.nii.gz", log_)
    spec = calibration.build_native_calibration(fm, max_iter=max_iter, log_=log_)
    native, field, _ = calibration.native_paths(fm)

    # 2. transfer EPI -> unwarp -> MC to the first corrected volume -> mean
    epi_mean, epi_mask = trans / "epi_unwarped_mean.nii", trans / "epi_mask_mean.nii"
    mc_path, motion_path = trans / "epi_mc.nii", trans / "motion.1D"
    if not (epi_mean.exists() and epi_mask.exists()):
        vols = convert_epi_block(incoming, epi_block, trans / "transfer_raw", keep, log_)
        proto = nib.load(str(native / "AP_mean.nii"))
        applier = PreloadedApplier(prototype_vol_path=native / "AP_mean.nii", fieldmap_path=field,
                                   phase_encoding_direction=spec["phase_encoding_axis"],
                                   polarity=1 if p.fieldmap.epi_phase_encoding == "AP" else -1,
                                   device=p.fieldmap.device, dtype=torch.float32, interpolation_backend=p.fieldmap.backend)
        from .rtpspy.rtp_volreg import RtpVolreg

        volreg = RtpVolreg(regmode=p.motion.volreg_mode, max_scan_length=max(len(vols) + 1, 2))
        volreg.ignore_init = 0
        volreg.save_proc = False
        mc = np.zeros((*proto.shape[:3], len(vols)), dtype=np.float32)
        motion = np.zeros((len(vols), 6), dtype=np.float32)
        first_uw = trans / "transfer_first_unwarped.nii"
        for t, vol in enumerate(vols):
            img = nib.load(str(vol))
            calibration.validate_bold(img, fm, p.fieldmap.epi_phase_encoding, vol)
            calibration.require_same_grid(img, proto)
            uw = applier.apply_volume(np.asarray(img.dataobj, dtype=np.float32))
            uw_img = nib.Nifti1Image(uw, img.affine)
            if t == 0:
                nib.save(uw_img, str(first_uw))
                volreg.set_ref_vol(str(first_uw))             # as the preparation: the first corrected volume
            uw_img.set_filename(f"vol_{t:04d}.nii")           # RTPSpy wants a name
            volreg.do_proc(uw_img, vol_idx=t)
            mc[..., t] = np.asarray(uw_img.dataobj, dtype=np.float32)
            motion[t] = np.asarray(volreg._motion[t], dtype=np.float32)
        nib.save(nib.Nifti1Image(mc, proto.affine), str(mc_path))
        np.savetxt(str(motion_path), motion)
        mean = mc.mean(axis=-1)
        nib.save(nib.Nifti1Image(mean, proto.affine), str(epi_mean))
        nib.save(nib.Nifti1Image(intensity_mask(mean, mask_threshold), proto.affine), str(epi_mask))
        for src, dst in ((epi_mean, trans / "rt_ref_epi.nii"), (epi_mask, trans / "rt_ref_epi_mask.nii")):
            if not dst.exists():
                shutil.copyfile(src, dst)
        log_.info("[PREP] reference %s from %d volumes; mask %d voxels", epi_mean, len(vols), int(intensity_mask(mean, mask_threshold).sum()))
    else:
        log_.info("[PREP] %s exists; reference kept", epi_mean)

    # 3. optional stand-in decoder on the reference grid
    decoder_path: Optional[Path] = None
    if decoder == "random":
        decoder_path = trans / "phantom_decoder.nii"
        if not decoder_path.exists():
            mask_img = nib.load(str(epi_mask))
            random_decoder(np.asarray(mask_img.dataobj), mask_img.affine, decoder_path)
            log_.info("[PREP] stand-in decoder %s", decoder_path)
    return {"fmap_dir": fm, "calibration": native, "reference": epi_mean, "mask": epi_mask, "mc": mc_path, "motion": motion_path,
            "decoder": decoder_path}
