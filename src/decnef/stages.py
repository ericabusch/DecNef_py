"""The per-volume processing stages, in order:

    unwarp -> motion correction (+FD) -> censor flags (+DVARS) -> regression
    -> transform to the analysis space -> final mask/smooth -> score

Each stage is a function of the explicit `RunState` (what the old handler
object held) and returns what the next stage needs. The numeric code is the
original `process_volume` body; only the error handling changed: a stage that
cannot do its job raises, instead of substituting zeros or an un-regressed
volume.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from . import outputs
from .calibration import require_same_grid, validate_bold
from .rtpspy.rtp_regress import RtpRegress
from .smoothing import smooth_file
from .tools import run

log = logging.getLogger(__name__)


class StageError(RuntimeError):
    pass


def log_step(stage: str, volume_idx: int, extra: str = "", start_t: Optional[float] = None) -> None:
    dt = f" ({time.time() - start_t:.3f}s)" if start_t is not None else ""
    log.info("[%s] vol %05d %s%s", stage, volume_idx, extra, dt)


# ----------------------------------------------------------------------------- helpers (unchanged numerics)
def fd_from_rtpspy_delta(delta: np.ndarray, radius_mm: float) -> float:
    """Power-style FD from RTPSpy [roll, pitch, yaw (deg), dS, dL, dP (mm)] deltas."""
    delta = np.asarray(delta, dtype=float)
    if delta.shape[-1:] != (6,):
        raise ValueError(f"Expected motion vectors with 6 values, got shape {delta.shape}")
    rotation_mm = np.deg2rad(delta[..., :3]) * float(radius_mm)
    return float(np.sum(np.abs(rotation_mm), axis=-1) + np.sum(np.abs(delta[..., 3:]), axis=-1))


def compute_dvars(prev_vol: np.ndarray, cur_vol: np.ndarray, mask: Optional[np.ndarray]) -> float:
    diff = (cur_vol - prev_vol)
    if mask is not None:
        diff = diff[mask]
    diff = diff.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def robust_z(x: float, history: list) -> float:
    """z against the median/MAD of `history`; NaN with fewer than 5 values or MAD ~ 0."""
    if len(history) < 5:
        return float("nan")
    med = float(np.median(history))
    mad = float(np.median(np.abs(np.asarray(history) - med)))
    if mad < 1e-8:
        return float("nan")
    return float((x - med) / (1.4826 * mad))


class ProcSrc:
    """Current 3D volume for RTPSpy mask regressors (GS/WM/Vent read `.proc_data`)."""

    def __init__(self):
        self.proc_data = None


class VoxelNormalizer:
    """RTPSpy-style percent-signal scaling used when regression is off:
    divide by the running mean of the first `ref_volumes` volumes, x100, clip at 200."""

    def __init__(self, ref_volumes: int = 1, brain_mask: Optional[np.ndarray] = None):
        self.ref_volumes = max(1, int(ref_volumes))
        self._sum = None
        self._count = 0
        self._y_mean = None
        self._y_mean_mask = None
        self._brain_mask = brain_mask.astype(bool, copy=False) if brain_mask is not None else None

    def apply(self, vol_data: np.ndarray) -> np.ndarray:
        data = np.asarray(vol_data, dtype=np.float32)
        if self._count < self.ref_volumes:
            if self._sum is None:
                self._sum = np.zeros_like(data, dtype=np.float64)
            self._sum += data
            self._count += 1
            self._y_mean = (self._sum / float(self._count)).astype(np.float32)
            self._y_mean_mask = np.abs(self._y_mean) > 1e-6
            if self._brain_mask is not None:
                self._y_mean_mask &= self._brain_mask
            if self._count >= self.ref_volumes:
                self._sum = None
        out = np.zeros_like(data, dtype=np.float32)
        mask = self._y_mean_mask
        out[mask] = data[mask] / self._y_mean[mask] * 100.0
        out[out > 200.0] = 200.0
        return out


class MotionRegressor:
    """RtpRegress with the pipeline's parameters. `apply` raises on failure."""

    def __init__(self, volreg, reg_mask: Optional[Path], gs_mask: Optional[Path], wm_mask: Optional[Path],
                 vent_mask: Optional[Path], mot_reg: str, max_poly_order: float, TR: float, max_scan_length: int,
                 norm_ref_volumes: int, phys_reg: str = "None", rtp_physio=None):
        wait_num = max(0, int(norm_ref_volumes) - 1)
        self._regress = RtpRegress(mot_reg=mot_reg, volreg=volreg, mask_file=str(reg_mask) if reg_mask is not None else 0,
                                   TR=TR, wait_num=wait_num, max_poly_order=max_poly_order, save_proc=False,
                                   online_saving=False, reg_retro_proc=False, max_scan_length=max_scan_length,
                                   spike_reg_num=200, phys_reg=phys_reg, rtp_physio=rtp_physio)
        for name, mask in (("GS", gs_mask), ("WM", wm_mask), ("Vent", vent_mask)):
            if mask is not None:
                self._regress.set_param(f"{name}_reg", True)
                self._regress.set_param(f"{name}_mask", str(mask))
        self._ready = False

    def apply(self, mc_img, volume_idx: int, fd_censor: int = 0, dvars_censor: int = 0) -> tuple[np.ndarray, bool]:
        """Returns (cleaned volume, reg_ready). RTPSpy works in place on `mc_img`."""
        if not self._ready:
            self._ready = bool(self._regress.ready_proc())
        if not self._ready:
            return np.asanyarray(mc_img.dataobj), False
        prev_vol = getattr(self._regress, "_vol_num", 0)
        self._regress.do_proc(mc_img, vol_idx=volume_idx - 1, fd_censor=fd_censor, dvars_censor=dvars_censor)
        cur_vol = getattr(self._regress, "_vol_num", prev_vol + 1)
        regressed = cur_vol > getattr(self._regress, "wait_num", 0)
        return np.asarray(mc_img.dataobj, dtype=np.float32), regressed

    def get_regressors(self, volume_idx: int) -> tuple[Optional[list], Optional[np.ndarray]]:
        des_mtx = getattr(self._regress, "desMtx", None)
        if des_mtx is None:
            return None, None
        try:
            row = des_mtx[volume_idx - 1]
        except (IndexError, KeyError, TypeError):
            return None, None
        if hasattr(row, "detach"):
            row = row.detach().cpu().numpy()
        row = np.asarray(row, dtype=float)
        reg_names = list(getattr(self._regress, "reg_names", []))
        if len(reg_names) < row.shape[0]:
            reg_names.extend([f"poly_{i:02d}" for i in range(row.shape[0] - len(reg_names))])
        return reg_names, row


# ----------------------------------------------------------------------------- state
@dataclass
class RunState:
    """Everything the stages share across volumes (ex DICOMHandler attributes)."""
    settings: object
    layout: object
    decoder_template: Optional[Path]
    ref_affine: np.ndarray
    volreg: object
    applier: object = None                      # unwarp.PreloadedApplier or None (file-based fallback)
    fieldmap_path: Optional[Path] = None
    motion_regressor: Optional[MotionRegressor] = None
    voxel_normalizer: Optional[VoxelNormalizer] = None
    proc_src: ProcSrc = field(default_factory=ProcSrc)
    gpu_resampler: object = None
    smoother: object = None
    scorer: object = None
    score_tracker: object = None                # StageSchedule or None
    reference_score_stats: Optional[dict] = None
    physio: object = None                       # physio.Receiver when BIOPAC runs in-process
    dvars_mask: Optional[np.ndarray] = None
    # per-run mutable state
    prev_motion: Optional[np.ndarray] = None
    prev_mc_for_dvars: Optional[np.ndarray] = None
    dvars_hist: list = field(default_factory=list)
    censor_next_fd: bool = False
    censor_next_dvars: bool = False
    last_dvars_val: float = float("nan")
    last_dvars_z: float = float("nan")
    biopac_timelag_sum: float = 0.0
    biopac_timelag_count: int = 0


# ----------------------------------------------------------------------------- stages
def unwarp_stage(state: RunState, raw_nii: Path, volume_idx: int) -> nib.Nifti1Image:
    """Raw BOLD -> unwarped BOLD in calibration geometry (before motion estimation)."""
    s, lay = state.settings.pipeline, state.layout
    t0 = time.time()
    raw_img = nib.load(str(raw_nii))
    validate_bold(raw_img, lay.fmap_dir, s.fieldmap.epi_phase_encoding, raw_nii)
    require_same_grid(raw_img, nib.load(str(lay.motion_ref_epi)))
    raw_data = np.asarray(raw_img.dataobj, dtype=np.float32)
    out_path = lay.unwarped_nii(volume_idx)
    if state.applier is not None:
        corrected = state.applier.apply_volume(raw_data)
    else:
        from .unwarp import apply_fieldmap_file

        spec = validate_bold(raw_img, lay.fmap_dir, s.fieldmap.epi_phase_encoding, raw_nii)
        apply_fieldmap_file(epi_path=raw_nii, fieldmap_path=state.fieldmap_path, out_path=out_path,
                            phase_encoding_direction=spec["phase_encoding_axis"],
                            polarity=1 if s.fieldmap.epi_phase_encoding == "AP" else -1, device=s.fieldmap.device)
        corrected = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32).copy()
    img = nib.Nifti1Image(corrected, raw_img.affine)
    if s.fieldmap.save_intermediate_unwarped:
        nib.save(img, str(out_path))
    log_step("FMAP", volume_idx, start_t=t0)
    return img


def motion_stage(state: RunState, img: nib.Nifti1Image, volume_idx: int) -> tuple[nib.Nifti1Image, Path, float, float]:
    """RtpVolreg motion correction of the unwarped volume; writes mc/, motion_rt.1D and fd_rt.csv.
    Returns (mc image, mc path, FD value, FD as saved (NaN during skip_first_trs))."""
    s, lay = state.settings.pipeline, state.layout
    t0 = time.time()
    mc_nii = lay.mc_nii(volume_idx)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))
    state.volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)        # in place on tmp_img.dataobj
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    mc_img = nib.Nifti1Image(mc_data, state.ref_affine)          # fixed corrected reference grid
    nib.save(mc_img, str(mc_nii))

    try:
        motion_vec = np.asarray(state.volreg._motion[volume_idx - 1]).astype(float)
    except (IndexError, TypeError, AttributeError) as exc:
        raise StageError(f"no motion estimate for volume {volume_idx}: {exc}") from exc
    outputs.append_motion(lay.motion_1d, motion_vec)
    delta = np.zeros_like(motion_vec) if state.prev_motion is None else motion_vec - state.prev_motion
    state.prev_motion = motion_vec.copy()
    fd_value = fd_from_rtpspy_delta(delta, s.motion.brain_radius_mm)
    fd_to_save = float("nan") if volume_idx <= s.skip_first_trs else fd_value
    outputs.append_fd(lay.fd_csv, volume_idx, fd_to_save)
    log_step("MC", volume_idx, f"FD={fd_to_save:.4f}" if np.isfinite(fd_to_save) else "FD=NaN (skipped TR)", start_t=t0)
    return mc_img, mc_nii, fd_value, fd_to_save


def censor_stage(state: RunState, mc_data: np.ndarray, volume_idx: int, fd_value: float, fd_to_save: float) -> tuple[int, int]:
    """FD and DVARS censor flags (+1 TR), DVARS history and dvars_rt.csv. Returns (fd_censor, dvars_censor)."""
    c, lay = state.settings.pipeline.censor, state.layout
    fd_censor = dvars_censor = 0
    if c.fd.enable:
        hit_fd = np.isfinite(fd_to_save) and (fd_value > c.fd.thr)
        if state.censor_next_fd:
            fd_censor = 1
            state.censor_next_fd = False
        if hit_fd:
            fd_censor = 1
            if c.plus1:
                state.censor_next_fd = True
    if c.dvars.enable and state.dvars_mask is not None:
        if state.prev_mc_for_dvars is not None:
            dvars_val = compute_dvars(state.prev_mc_for_dvars, mc_data, state.dvars_mask)
            z = robust_z(dvars_val, state.dvars_hist)       # against prior history (excludes current)
            state.dvars_hist.append(dvars_val)
            outputs.append_dvars(lay.dvars_csv, volume_idx, fd_to_save, dvars_val, z)
            state.last_dvars_val, state.last_dvars_z = dvars_val, z
            enough = len(state.dvars_hist) >= c.dvars.warmup
            hit_dvars = enough and np.isfinite(z) and (z > c.dvars.thr_robust_z)
            if state.censor_next_dvars:
                dvars_censor = 1
                state.censor_next_dvars = False
            if hit_dvars:
                dvars_censor = 1
                if c.plus1:
                    state.censor_next_dvars = True
        else:
            state.dvars_hist.append(0.0)
            state.last_dvars_val = state.last_dvars_z = float("nan")
    state.prev_mc_for_dvars = mc_data.copy()
    return fd_censor, dvars_censor


def regression_stage(state: RunState, mc_img: nib.Nifti1Image, volume_idx: int, fd_censor: int, dvars_censor: int) -> tuple[Path, bool]:
    """Nuisance regression (or voxel normalisation when regression is off); writes reg/ and regressors_rt.csv.
    Returns (reg path, reg_ready)."""
    lay = state.layout
    t0 = time.time()
    reg_nii = lay.reg_nii(volume_idx)
    if state.motion_regressor is not None:
        state.proc_src.proc_data = np.asanyarray(mc_img.dataobj)
        cleaned, reg_ready = state.motion_regressor.apply(mc_img, volume_idx, fd_censor=fd_censor, dvars_censor=dvars_censor)
        nib.save(nib.Nifti1Image(cleaned, mc_img.affine), str(reg_nii))
        log_step("REG", volume_idx, "motion", start_t=t0)
        reg_names, reg_row = state.motion_regressor.get_regressors(volume_idx)
        if reg_names and reg_row is not None:
            outputs.append_regressors(lay.regressors_csv, volume_idx, reg_names, reg_row)
    else:
        cleaned = state.voxel_normalizer.apply(np.asanyarray(mc_img.dataobj))
        reg_ready = True
        nib.save(nib.Nifti1Image(cleaned, mc_img.affine), str(reg_nii))
        log_step("REG", volume_idx, "skipped (voxel-normalized)", start_t=t0)
    return reg_nii, reg_ready


def status_stage(state: RunState, volume_idx: int, fd_to_save: float, fd_censor: int, dvars_censor: int,
                 reg_ready: bool, volume_timestamp: float) -> None:
    """regression_status_rt.csv (+ biopac_timelag.csv when enabled)."""
    lay = state.layout
    biopac_missing = state.physio.was_missing(volume_idx) if state.physio is not None else False
    outputs.append_regression_status(lay.regression_status_csv, volume_idx, fd_to_save, state.last_dvars_val,
                                     state.last_dvars_z, fd_censor, dvars_censor, reg_ready, biopac_missing)
    if state.settings.physio.timelag and state.physio is not None:
        trigger_ts = state.physio.get_trigger_timestamp(volume_idx)
        if trigger_ts is not None:
            timelag = volume_timestamp - trigger_ts
            state.biopac_timelag_sum += timelag
            state.biopac_timelag_count += 1
            outputs.append_biopac_timelag(lay.biopac_timelag_csv, volume_idx, trigger_ts, volume_timestamp, timelag,
                                          state.biopac_timelag_sum / state.biopac_timelag_count)


def _warp(state: RunState, src: Path, dst: Path, transforms: list) -> None:
    if state.gpu_resampler is not None:
        state.gpu_resampler.resample_nifti_to_nifti(src, dst)
        return
    cmd = ["antsApplyTransforms", "-d", "3", "-i", str(src), "-r", str(state.decoder_template), "-o", str(dst)]
    for t in transforms:
        cmd.extend(["-t", str(t)])
    cmd.extend(["-n", "Linear", "--float", "1"])
    run(cmd)


def transform_stage(state: RunState, reg_nii: Path, mc_nii: Path, volume_idx: int) -> tuple[Path, Optional[Path]]:
    """EPI passthrough, or warp to T1 / MNI onto the decoder grid (one code path for both).
    Returns (score input, original-score input or None)."""
    s, lay = state.settings.pipeline, state.layout
    t0 = time.time()
    want_orig = state.settings.decoder.enable_original_score
    if s.analysis_space == "epi":
        log_step("TRANS", volume_idx, "skipped (EPI space)", start_t=t0)
        return reg_nii, (mc_nii if want_orig else None)
    space = s.analysis_space
    transforms = [lay.epi2t1_composite] if space == "t1" else [lay.t1_to_mni_warp, lay.epi2t1_composite]
    missing = [p for p in [state.decoder_template, *transforms] if p is None or not Path(p).exists()]
    if missing:
        raise StageError(f"missing decoder/transform files for {space}-space warp: {missing}")
    out_dir = lay.space_dir(space)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = lay.space_nii(space, volume_idx)
    _warp(state, reg_nii, dst, transforms)
    orig_dst = None
    if want_orig:
        orig_dst = lay.space_nii(space, volume_idx, original=True)
        _warp(state, mc_nii, orig_dst, transforms)
    log_step("TRANS", volume_idx, f"warp->{space.upper()}", start_t=t0)
    return dst, orig_dst


def smooth_stage(state: RunState, score_input: Path, score_input_orig: Optional[Path], volume_idx: int) -> tuple[Path, Optional[Path]]:
    """Final masking / normalized Gaussian smoothing into smooth/ or masked/ (no-op without a smoother)."""
    if state.smoother is None:
        return score_input, score_input_orig
    kind = state.smoother.output_kind
    lay = state.layout
    out = smooth_file(state.smoother, score_input, lay.final_nii(kind, volume_idx))
    out_orig = smooth_file(state.smoother, score_input_orig, lay.final_nii(kind, volume_idx, original=True)) if score_input_orig else None
    return out, out_orig


def score_stage(state: RunState, score_input: Path, score_input_orig: Optional[Path], volume_idx: int,
                reg_ready: bool) -> dict:
    """Decoder score (+ z against the reference run); writes scores.csv / scores_z.csv.
    Returns the score message (without kind/source)."""
    lay = state.layout
    t0 = time.time()
    score_data = np.asanyarray(nib.load(str(score_input)).dataobj)
    original_score = None
    if score_input_orig is not None:
        original_score = state.scorer.score_from_array(np.asanyarray(nib.load(str(score_input_orig)).dataobj))
    raw_score = state.scorer.score_from_array(score_data)
    stage = state.score_tracker.label_for_volume(volume_idx) if state.score_tracker is not None else None
    analysis_timestamp = time.time()
    timestamp = outputs.append_score(lay.scores_csv, volume_idx, raw_score, original_score=original_score,
                                     reg_ready=reg_ready, timestamp=analysis_timestamp, event_type=stage)
    message = {"volume_idx": volume_idx, "timestamp": timestamp, "analysis_timestamp": analysis_timestamp,
               "score_raw": raw_score, "score_original": original_score, "reg_ready": reg_ready, "stage": stage}
    if state.reference_score_stats is not None:
        stats = state.reference_score_stats
        z_score = (raw_score - stats["mean"]) / stats["std"]
        outputs.append_score_z(lay.scores_z_csv, volume_idx, timestamp, raw_score, z_score, stats, reg_ready=reg_ready)
        message["score_z"] = z_score
    log_step("SCORE", volume_idx, f"raw={raw_score:.4f}" + (f" z={message['score_z']:.4f}" if "score_z" in message else ""), start_t=t0)
    return message
