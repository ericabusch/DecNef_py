#!/usr/bin/env python
import time
import csv
import logging
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, CancelledError
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from fmri_rt_preproc.native_fieldmap import (
    PIPELINE_ORDER, native_paths, load_calibration, find_source,
    validate_bold, require_same_grid, source_identity,
)

from motion_fd import fd_from_rtpspy_delta
from fmri_rt_preproc.volume_smooth import prepare_smoother, smooth_file
from fmri_rt_preproc.analysis_mask import final_output_kind, mask_provenance

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from fmri_rt_preproc.RTPSpy_tools.rtp_regress import RtpRegress
from fmri_rt_preproc.pyhysco_apply import apply_pyhysco_fieldmap, PreloadedPyHyscoApplier
from fmri_rt_preproc.utils import run  # your existing run() wrapper

from decoder_score import DecoderScorer
from biopac_rt.biopac_receiver import (
    BiopacReceiverConfig,
    BiopacRetroTSReceiver,
    BiopacRetroTSFileBuffer,
)
from rt_global_settings import load_regressor_settings
from volume_streamer import VolumeStreamerConfig, VolumeStreamerHandle

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("rt_pipeline")

# Silence noisy sub-loggers (keep only warnings+)
logging.getLogger("fmri_rt_preproc").setLevel(logging.WARNING)
logging.getLogger("RtpVolreg").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)

# ---------- Regressor config ----------
REGRESSOR_SETTINGS = load_regressor_settings()


@dataclass
class ResultEnvelope:
    scan: int
    volume_idx: int
    dicom_path: Path
    volume_timestamp: float
    success: bool
    error: Optional[str] = None
    traceback_text: Optional[str] = None
    attempts: int = 0
    raw_nii: Optional[Path] = None
    mc_nii: Optional[Path] = None
    score_input_nii: Optional[Path] = None
    score_input_orig_nii: Optional[Path] = None
    mc_data: Optional[np.ndarray] = None
    affine: Optional[np.ndarray] = None
    motion_vec: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

def log_step(step: str, vol: int, extra: str = "", start_t=None):
    """Compact colored/clean log."""
    v = f"{vol:05d}"
    if start_t is not None:
        dt = time.time() - start_t
        log.info(f"[{step:<5}] vol {v}  {extra}  ({dt*1000:.1f} ms)")
    else:
        log.info(f"[{step:<5}] vol {v}  {extra}")




class ScoreEventTracker:
    """Infer per-volume event/stage labels from session_metadata.json.

    Generic form: a "stages" dict written by rt_session.StageSchedule (schema in
    docs/session_api.md). Legacy forms, kept so existing runs and launchers keep
    their labels: "fixation_display" (rest runs) and "psychopy" with
    script == "rt_nf_events_parallel.py" (the NF events task).
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._mode = None
        self._skip_first_trs = 0
        self._baseline_trs = 0
        self._n_trials = 0
        self._stage_defs: list[tuple[str, int]] = []
        self._stages: Optional[dict] = None
        self._load_metadata()

    def _load_metadata(self) -> None:
        metadata_path = self.run_dir / "session_metadata.json"
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        stages = metadata.get("stages")
        if isinstance(stages, dict) and isinstance(stages.get("cycle", []), list):
            self._mode = "stages"
            self._stages = stages
            return

        if isinstance(metadata.get("fixation_display"), dict):
            fix = metadata["fixation_display"]
            self._mode = "rest"
            self._skip_first_trs = int(fix.get("skip_first_trs", 0) or 0)
            self._baseline_trs = int(fix.get("baseline_trs", 0) or 0)

        psychopy = metadata.get("psychopy")
        if not isinstance(psychopy, dict):
            return
        script = psychopy.get("script")
        if script != "rt_nf_events_parallel.py":
            return

        self._mode = "nf_events"
        self._skip_first_trs = int(psychopy.get("skip_first_trs", 0) or 0)
        self._baseline_trs = int(psychopy.get("baseline_trs", 0) or 0)
        self._n_trials = int(psychopy.get("n_trials", 0) or 0)
        self._stage_defs = [
            ("iti", int(psychopy.get("iti_trs", 0) or 0)),
            ("cue", int(psychopy.get("cue_trs", 0) or 0)),
            ("scans", int(psychopy.get("scans_trs", 0) or 0)),
            ("delay", int(psychopy.get("delay_trs", 0) or 0)),
            ("feedback", int(psychopy.get("feedback_trs", 0) or 0)),
        ]

    def for_volume(self, volume_idx: int) -> Optional[str]:
        if self._mode is None:
            return None
        if self._mode == "stages":
            from rt_session import label_for_volume

            return label_for_volume(self._stages, volume_idx)
        if volume_idx <= self._skip_first_trs:
            return "background"

        exp_tr = volume_idx - self._skip_first_trs

        if self._mode == "rest":
            if exp_tr <= self._baseline_trs:
                return "background"
            return "rest"

        if exp_tr <= self._baseline_trs:
            return "background"

        if self._mode != "nf_events":
            return None

        stage_cycle: list[str] = []
        for stage, dur in self._stage_defs:
            stage_cycle.extend([stage] * max(0, dur))
        if not stage_cycle:
            return None

        tr_after_baseline = exp_tr - self._baseline_trs
        cycle_len = len(stage_cycle)
        trial_idx = (tr_after_baseline - 1) // cycle_len
        if trial_idx >= self._n_trials:
            return "post_task"
        stage_pos = (tr_after_baseline - 1) % cycle_len
        return stage_cycle[stage_pos]


def append_score(
    csv_path: Path,
    volume_idx: int,
    raw_score: float,
    original_score: Optional[float] = None,
    reg_ready: Optional[bool] = None,
    timestamp: Optional[float] = None,
    event_type: Optional[str] = None,
) -> float:
    if timestamp is None:
        timestamp = time.time()
    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["volume_idx", "timestamp", "score_raw", "score_original", "reg_ready", "event_type"])
        writer.writerow([volume_idx, timestamp, raw_score, original_score, int(reg_ready) if reg_ready is not None else "", event_type or ""])
    return timestamp


def append_score_z(
    csv_path: Path,
    volume_idx: int,
    timestamp: float,
    raw_score: float,
    z_score: float,
    ref_stats: dict,
    reg_ready: Optional[bool] = None,
) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "volume_idx",
                    "timestamp",
                    "score_raw",
                    "score_z",
                    "ref_run",
                    "ref_mean",
                    "ref_std",
                    "ref_n",
                    "ref_used_reg_ready",
                    "reg_ready",
                ]
            )
        writer.writerow(
            [
                volume_idx,
                timestamp,
                raw_score,
                z_score,
                ref_stats["run"],
                ref_stats["mean"],
                ref_stats["std"],
                ref_stats["n"],
                int(ref_stats.get("used_reg_ready", False)),
                int(reg_ready) if reg_ready is not None else "",
            ]
        )

def append_motion(motion_path: Path, motion_vec: np.ndarray):
    """
    Append a single 6-parameter motion vector to a text file (AFNI-style 1D).
    """
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(motion_path, "a") as f:
        f.write(" ".join(f"{x:.6f}" for x in motion_vec) + "\n")

def append_fd(fd_path: Path, volume_idx: int, fd_value: float):
    """
    Append a single FD value to a CSV file: volume_idx, fd.
    """
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    exists = fd_path.exists()
    with open(fd_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", "fd"])
        w.writerow([volume_idx, fd_value])

class ProcSrc:
    """Holds the current 3D volume for RTPSpy mask regressors (GS/WM/Vent)."""
    def __init__(self):
        self.proc_data = None


class RTPStyleVoxelNormalizer:
    """
    Lightweight voxel-wise intensity scaling that matches RTPSpy's Y_mean scaling.

    When motion regression is disabled we still output volumes in percent-signal
    style units by dividing each voxel by a fixed reference mean and multiplying
    by 100 (with RTPSpy's clipping/zero-mask behavior).
    """

    def __init__(self, ref_volumes: int = 1, brain_mask: Optional[np.ndarray] = None):
        self.ref_volumes = max(1, int(ref_volumes))
        self._sum: Optional[np.ndarray] = None
        self._count = 0
        self._y_mean: Optional[np.ndarray] = None
        self._y_mean_mask: Optional[np.ndarray] = None
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
    def __init__(
        self,
        volreg: RtpVolreg,
        reg_mask: Optional[Path] = None,
        gs_mask: Optional[Path] = None,
        wm_mask: Optional[Path] = None,
        vent_mask: Optional[Path] = None,
        mot_reg: str = "mot6",
        max_poly_order: float = np.inf,
        TR: float = 1,
        max_scan_length: int = 1000,
        norm_ref_volumes: int = 1,
        enable_fd_censor_reg: bool = False,
        enable_dvars_censor_reg: bool = False,
        phys_reg: str = "None",
        rtp_physio: Optional[object] = None,
    ):
        wait_num = max(0, int(norm_ref_volumes) - 1)
        kwargs = dict(
            mot_reg=mot_reg,
            volreg=volreg,
            mask_file=str(reg_mask) if reg_mask is not None else 0,
            TR=TR,
            wait_num=wait_num,
            max_poly_order=max_poly_order,
            save_proc=False,
            online_saving=False,
            reg_retro_proc=False,
            max_scan_length=max_scan_length,
            spike_reg_num=200,
            phys_reg=phys_reg,
            rtp_physio=rtp_physio,
        )
        self._regress = RtpRegress(**kwargs)

        if gs_mask is not None:
            self._regress.set_param("GS_reg", True)
            self._regress.set_param("GS_mask", str(gs_mask))
        if wm_mask is not None:
            self._regress.set_param("WM_reg", True)
            self._regress.set_param("WM_mask", str(wm_mask))
        if vent_mask is not None:
            self._regress.set_param("Vent_reg", True)
            self._regress.set_param("Vent_mask", str(vent_mask))

        self._ready = False

    def apply(
        self,
        mc_img: nib.Nifti1Image,
        volume_idx: int,
        fd_censor: int = 0,
        dvars_censor: int = 0,
    ) -> tuple[np.ndarray, bool]:
        """
        Returns (cleaned_vol, regressed_bool). Always returns a tuple.
        """

        # Ensure regressor init/ready state
        if not self._ready:
            try:
                self._ready = bool(self._regress.ready_proc())
            except Exception as exc:
                log.error(f"[REG] Failed to prepare regressor: {exc}")
                return np.asanyarray(mc_img.dataobj), False

        if not self._ready:
            return np.asanyarray(mc_img.dataobj), False

        # Do regression
        try:
            prev_vol = getattr(self._regress, "_vol_num", 0)

            # RTPSpy modifies mc_img in-place
            self._regress.do_proc(
                mc_img,
                vol_idx=volume_idx - 1,
                fd_censor=fd_censor,
                dvars_censor=dvars_censor,
            )

            cur_vol = getattr(self._regress, "_vol_num", prev_vol + 1)
            regressed = cur_vol > getattr(self._regress, "wait_num", 0)

            cleaned = np.asarray(mc_img.dataobj, dtype=np.float32)
            return cleaned, regressed

        except Exception as exc:
            log.error(f"[REG] Motion regression failed at vol {volume_idx:05d}: {exc}")
            return np.asanyarray(mc_img.dataobj), False

    def get_regressors(self, volume_idx: int) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
        des_mtx = getattr(self._regress, "desMtx", None)
        if des_mtx is None:
            return None, None

        idx = volume_idx - 1
        try:
            row = des_mtx[idx]
        except Exception:
            return None, None

        if hasattr(row, "detach"):
            row = row.detach().cpu().numpy()

        row = np.asarray(row, dtype=float)

        reg_names = list(getattr(self._regress, "reg_names", []))
        if len(reg_names) < row.shape[0]:
            extra = row.shape[0] - len(reg_names)
            reg_names.extend([f"poly_{i:02d}" for i in range(extra)])

        return reg_names, row





# ---------- Simple config for this RT session ----------

@dataclass
class RTSessionConfig:
    subject: str
    day: str
    run: str
    incoming_root: Path
    base_data: Path
    decoder_template: Optional[Path] = None
    decoder_roi_txt: Optional[Path] = None
    reference_score_run: Optional[str] = None
    reference_score_stats: Optional[dict] = None
    enable_scoring: bool = True
    enable_original_score: bool = False

    # NEW
    ap_block: Optional[int] = None
    pa_block: Optional[int] = None

    @property
    def fmap_dir(self) -> Path:
        if self.ap_block is None and self.pa_block is None:
            return self.day_root / "fmap"

        if self.ap_block is None or self.pa_block is None:
            raise ValueError("Set both --ap-block and --pa-block, or neither.")

        return (
            self.day_root
            / "fmap"
            / f"pair-ap{int(self.ap_block):03d}_pa{int(self.pa_block):03d}"
        )

    @property
    def subject_root(self) -> Path:
        return self.base_data / f"sub-{self.subject}"

    @property
    def day_root(self) -> Path:
        return self.subject_root / self.day

    @property
    def trans_dir(self) -> Path:
        # precomputed transforms live here (from offline pipeline)
        return self.day_root / "func" / "trans"

    @property
    def rt_wm_mask(self) -> Path:
        return self.trans_dir / "rt_WM_mask.nii"

    @property
    def rt_vent_mask(self) -> Path:
        return self.trans_dir / "rt_Vent_mask.nii"

    @property
    def rt_work_dir(self) -> Path:
        """
        Where we put per-volume NIfTIs, logs, etc.

        Runs are stored under func/XXX, where XXX corresponds to the middle
        element of the DICOM name (historically called "block").
        """
        d = self.day_root / "func" / self.run
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def incoming_dir(self) -> Path:
        # The full incoming folder, you gave an example root:
        return self.incoming_root

    @property
    def rt_raw_dir(self) -> Path:
        d = self.rt_work_dir / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_mc_dir(self) -> Path:
        d = self.rt_work_dir / "mc"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_reg_dir(self) -> Path:
        d = self.rt_work_dir / "reg"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_mni_dir(self) -> Path:
        d = self.rt_work_dir / "mni"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_unwarp_dir(self) -> Path:
        d = self.rt_work_dir / "unwarped"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_distorted_motion_ref_epi(self) -> Path:
        """
        Legacy filename retained for external callers; not the online MC target.
        Older sessions contain distorted data; new preparations contain corrected data.
        """
        return self.trans_dir / "rt_ref_epi.nii"

    @property
    def rt_distorted_motion_ref_mask(self) -> Path:
        """
        Legacy reference-mask filename; not the online DVARS mask.
        """
        return self.trans_dir / "rt_ref_epi_mask.nii"

    @property
    def rt_unwarped_analysis_ref_epi(self) -> Path:
        """
        Unwarped EPI reference used for EPI->T1/MNI transform grids.
        """
        return _prefer_uncompressed_nifti(self.trans_dir / "epi_unwarped_mean.nii")

    @property
    def rt_unwarped_analysis_ref_mask(self) -> Path:
        """
        Brain mask in the unwarped EPI reference grid.
        """
        return _prefer_uncompressed_nifti(self.trans_dir / "epi_mask_mean.nii")

    @property
    def rt_motion_ref_epi(self) -> Path:
        """Corrected reference used for motion correction after unwarping."""
        return self.rt_unwarped_analysis_ref_epi

    @property
    def rt_motion_ref_mask(self) -> Path:
        """Corrected reference mask used after unwarping and motion correction."""
        return self.rt_unwarped_analysis_ref_mask

    @property
    def rt_ref_epi(self) -> Path:
        """Deprecated alias for the unwarped analysis reference."""
        return self.rt_unwarped_analysis_ref_epi

    @property
    def rt_ref_mask(self) -> Path:
        """Deprecated alias for the unwarped analysis-reference mask."""
        return self.rt_unwarped_analysis_ref_mask


def resolve_decoder_template(cfg: RTSessionConfig) -> Path:
    return cfg.decoder_template or (
        Path(cfg.base_data).parent
        / "decoders"
        / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"
    )


def maybe_init_gpu_resampler(cfg: RTSessionConfig):
    if not bool(getattr(REGRESSOR_SETTINGS, "use_gpu_resampler", False)):
        return None

    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    if analysis_space not in {"t1", "mni"}:
        log.info("[GPU] Disabled: analysis_space=%s does not require warp.", analysis_space)
        return None

    from gpu_ants_like_resampler import GpuAntsLikeResampler, precompute_sampling_grid

    if analysis_space == "t1":
        fixed_ref = resolve_decoder_template(cfg)
        transforms = [cfg.trans_dir / "epi2t1_Composite.h5"]
        out_dir = cfg.trans_dir / "gpu_grids" / "epi_to_t1"
    else:
        fixed_ref = resolve_decoder_template(cfg)
        transforms = [
            cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii",
            cfg.trans_dir / "epi2t1_Composite.h5",
        ]
        out_dir = cfg.trans_dir / "gpu_grids" / "epi_to_mni"

    moving_ref = cfg.rt_unwarped_analysis_ref_epi
    missing = [p for p in [fixed_ref, *transforms, moving_ref] if not p.exists()]
    if missing:
        log.warning("[GPU] Disabled: missing inputs for GPU resampler: %s", missing)
        return None

    grid_path_npz = out_dir / "sampling_grid.npz"
    grid_path_pt = out_dir / "sampling_grid.pt"
    grid_path = grid_path_npz if grid_path_npz.exists() else grid_path_pt
    if grid_path.exists():
        try:
            fixed_img = nib.load(str(fixed_ref))
            payload = GpuAntsLikeResampler._load_payload(grid_path)
            cached_shape = tuple(int(v) for v in payload["fixed_shape"])
            cached_affine = np.asarray(payload["fixed_affine"], dtype=np.float64)
            if cached_shape != fixed_img.shape[:3] or not np.allclose(
                cached_affine,
                fixed_img.affine,
                atol=1e-3,
            ):
                log.warning(
                    "[GPU] Existing sampling grid does not match current fixed reference; "
                    "recomputing %s",
                    out_dir,
                )
                for stale in [
                    grid_path_npz,
                    grid_path_pt,
                    out_dir / "coord_x_in_fixed.nii",
                    out_dir / "coord_y_in_fixed.nii",
                    out_dir / "coord_z_in_fixed.nii",
                ]:
                    if stale.exists():
                        stale.unlink()
                grid_path = grid_path_npz
        except Exception as exc:
            log.warning("[GPU] Could not validate existing sampling grid %s: %s", grid_path, exc)
            for stale in [
                grid_path_npz,
                grid_path_pt,
                out_dir / "coord_x_in_fixed.nii",
                out_dir / "coord_y_in_fixed.nii",
                out_dir / "coord_z_in_fixed.nii",
            ]:
                if stale.exists():
                    stale.unlink()
            grid_path = grid_path_npz
    if not grid_path.exists():
        log.info("[GPU] Precomputing fixed sampling grid at %s", out_dir)
        grid_path = precompute_sampling_grid(
            moving_ref=moving_ref,
            fixed_ref=fixed_ref,
            transforms=transforms,
            out_dir=out_dir,
        )

    device = str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_device", "cuda"))
    resampler = GpuAntsLikeResampler(grid_path=grid_path, device=device, mode="bilinear")
    log.info("[GPU] Resampler ready (%s, device=%s).", grid_path, resampler.device)
    return resampler


def _build_pyhysco_applier(
    cfg: RTSessionConfig,
    prototype_vol_path: Path,
) -> Optional[PreloadedPyHyscoApplier]:
    method = str(getattr(REGRESSOR_SETTINGS, "fieldmap_method", "pyhysco")).lower()
    if method != "pyhysco":
        return None
    if not bool(getattr(REGRESSOR_SETTINGS, "use_preloaded_pyhysco", True)):
        return None

    fmap_dir = cfg.fmap_dir
    pyhysco_field = _preferred_pyhysco_fieldmap(fmap_dir)
    if not pyhysco_field.exists():
        log.warning("[FMAP] Preloaded PyHySCO disabled: missing fieldmap at %s", pyhysco_field)
        return None
    if not prototype_vol_path.exists():
        log.warning("[FMAP] Preloaded PyHySCO disabled: missing prototype volume at %s", prototype_vol_path)
        return None

    epi_pe = str(getattr(REGRESSOR_SETTINGS, "epi_phase_encoding", "PA")).upper()
    polarity = 1 if epi_pe == "AP" else -1
    phase_encoding_direction = load_calibration(fmap_dir)["phase_encoding_axis"]
    device = str(getattr(REGRESSOR_SETTINGS, "pyhysco_device", "cuda"))
    backend = str(getattr(REGRESSOR_SETTINGS, "pyhysco_backend", "grid_sample")).lower()
    applier = PreloadedPyHyscoApplier(
        prototype_vol_path=prototype_vol_path,
        fieldmap_path=pyhysco_field,
        phase_encoding_direction=phase_encoding_direction,
        polarity=polarity,
        device=device,
        dtype=torch.float32,
        interpolation_backend=backend,
    )
    log.info(
        "[FMAP] Preloaded PyHySCO applier ready (device=%s, backend=%s, prototype=%s).",
        applier.device,
        backend,
        prototype_vol_path,
    )
    return applier


def _resolve_pyhysco_fieldmap(fmap_dir: Path) -> Path:
    return native_paths(fmap_dir)[1]


def maybe_init_pyhysco_applier(cfg: RTSessionConfig) -> Optional[PreloadedPyHyscoApplier]:
    # Prototype is calibration-local, independent of the session motion target.
    return _build_pyhysco_applier(cfg, native_paths(cfg.fmap_dir)[0] / "AP_mean.nii")


def _prefer_uncompressed_nifti(path_nii: Path) -> Path:
    """Prefer .nii, but transparently fall back to .nii.gz when needed."""
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    if gz.exists():
        return gz
    return path_nii


def resolve_decoder_roi_txt(cfg: RTSessionConfig) -> Optional[Path]:
    if cfg.decoder_roi_txt is None:
        return None
    if not cfg.decoder_roi_txt.exists():
        log.warning("[SCORE] Decoder ROI txt not found at %s; falling back to decoder NIfTI mask.", cfg.decoder_roi_txt)
        return None
    return cfg.decoder_roi_txt


def load_reference_score_stats(cfg: RTSessionConfig, run_id: Optional[str]) -> Optional[dict]:
    if not run_id:
        return None
    run_dir = cfg.day_root / "func" / str(run_id)
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        log.warning("[SCORE] Reference scores.csv not found at %s", scores_path)
        return None

    reg_ready_map = None
    reg_status_path = run_dir / "regression_status_rt.csv"
    if reg_status_path.exists():
        reg_ready_map = {}
        with open(reg_status_path, newline="") as f:
            reader = csv.DictReader(f)
            if "volume_idx" in reader.fieldnames and "reg_ready" in reader.fieldnames:
                for row in reader:
                    try:
                        vol = int(row["volume_idx"])
                        reg_ready_map[vol] = bool(int(row["reg_ready"]))
                    except (TypeError, ValueError):
                        continue
            else:
                log.warning("[SCORE] reg_ready column missing in %s", reg_status_path)
                reg_ready_map = None

    scores = []
    scores_all = []
    reg_ready_scores = []
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        has_score_reg_ready = bool(reader.fieldnames and "reg_ready" in reader.fieldnames)
        for row in reader:
            try:
                vol = int(row["volume_idx"])
                raw = float(row["score_raw"])
            except (TypeError, ValueError, KeyError):
                continue
            if np.isnan(raw):
                continue
            scores_all.append(raw)
            score_reg_ready = None
            if has_score_reg_ready:
                try:
                    score_reg_ready = bool(int(row["reg_ready"]))
                except (TypeError, ValueError, KeyError):
                    score_reg_ready = None
            ready = reg_ready_map.get(vol, False) if reg_ready_map is not None else score_reg_ready
            if ready is None:
                ready = True
            if ready:
                scores.append(raw)
            if ready:
                reg_ready_scores.append((vol, raw))

    used_reg_ready = (reg_ready_map is not None) or has_score_reg_ready
    skipped_first_reg_ready = False
    if reg_ready_scores:
        # Drop the first regressed sample from RS reference normalization.
        # RTP regression output can still show a transient at the first reg_ready TR.
        scores = [raw for _, raw in reg_ready_scores[1:]]
        skipped_first_reg_ready = True
    if used_reg_ready and not scores:
        log.warning(
            "[SCORE] No reg_ready scores found in %s; falling back to all scores.",
            scores_path,
        )
        scores = scores_all
        used_reg_ready = False

    if len(scores) < 2:
        log.warning("[SCORE] Not enough reference scores in %s to compute z-score.", scores_path)
        return None

    values = np.asarray(scores, dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        log.warning("[SCORE] Reference score std invalid (%s) in %s.", std, scores_path)
        return None

    return {
        "run": str(run_id),
        "mean": mean,
        "std": std,
        "n": int(len(scores)),
        "used_reg_ready": used_reg_ready,
        "skipped_first_reg_ready": skipped_first_reg_ready,
    }

def _prefer_uncompressed_nifti(path_nii: Path) -> Path:
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    if gz.exists():
        return gz
    return path_nii


def _preferred_pyhysco_fieldmap(fmap_dir: Path) -> Path:
    return native_paths(fmap_dir)[1]


def _fieldmap_ready(cfg: RTSessionConfig) -> bool:
    if str(REGRESSOR_SETTINGS.fieldmap_method).lower() != "pyhysco":
        raise ValueError("unwarp -> MC supports PyHySCO only; select fieldmap_method=pyhysco.")
    try:
        load_calibration(cfg.fmap_dir)
        return True
    except (OSError, ValueError):
        return False


def ensure_fieldmap_only(cfg: RTSessionConfig) -> None:
    """
    Ensure the selected AP/PA pair has fieldmap outputs.

    This is RT-side prerequisite checking.
    It does not run full preprocessing.
    It does not require structural block.
    It does not require a separate EPI block.
    It averages raw AP/PA without motion correction or a session reference.
    """

    if (cfg.ap_block is None) != (cfg.pa_block is None):
        raise ValueError("Set both --ap-block and --pa-block, or neither.")
    if _fieldmap_ready(cfg):
        log.info("[FMAP] Native fieldmap ready in %s", cfg.fmap_dir)
        return

    log.warning(
        "[FMAP] Fieldmap missing for AP/PA pair ap=%s pa=%s in %s; building fieldmap only.",
        cfg.ap_block,
        cfg.pa_block,
        cfg.fmap_dir,
    )

    # Reuse the DICOM staging/conversion helper from run_preproc.
    # It copies only the requested AP/PA DICOM blocks and converts them to fixed names.
    from run_preproc import _stage_convert_to_target
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
    from types import SimpleNamespace

    cfg.fmap_dir.mkdir(parents=True, exist_ok=True)

    ap_path = cfg.fmap_dir / "AP.nii.gz"
    pa_path = cfg.fmap_dir / "PA.nii.gz"

    if cfg.ap_block is not None:
        _stage_convert_to_target(cfg.incoming_dir, int(cfg.ap_block), ap_path)
        _stage_convert_to_target(cfg.incoming_dir, int(cfg.pa_block), pa_path)
    ap_path = find_source(cfg.fmap_dir, "AP")
    pa_path = find_source(cfg.fmap_dir, "PA")

    # Minimal config object for FMRIRealtimePreprocessor._prepare_fieldmap().
    fieldmap_cfg = SimpleNamespace(
        subject_id=cfg.subject,
        day_id=cfg.day,
        root=cfg.day_root,
        subject_root=cfg.subject_root,
        ap_file=ap_path,
        pa_file=pa_path,
        fieldmap_dir=cfg.fmap_dir,
        fieldmap_method=str(REGRESSOR_SETTINGS.fieldmap_method).lower(),
        epi_phase_encoding=str(REGRESSOR_SETTINGS.epi_phase_encoding).upper(),
        decoder_template=Path("unused_decoder_template.nii"),
    )

    pipe = FMRIRealtimePreprocessor(fieldmap_cfg)
    pipe._prepare_fieldmap()

    if not _fieldmap_ready(cfg):
        raise FileNotFoundError(
            f"[FMAP] Fieldmap-only build finished, but required outputs are still missing in {cfg.fmap_dir}"
        )

    log.info("[FMAP] Fieldmap-only build complete: %s", cfg.fmap_dir)

def write_session_metadata(cfg: RTSessionConfig, decoder_template: Path) -> None:
    metadata_path = cfg.rt_work_dir / "session_metadata.json"
    payload = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": cfg.subject,
            "day": cfg.day,
            "run": cfg.run,
            "incoming_root": str(cfg.incoming_root),
            "base_data": str(cfg.base_data),
            "decoder_template": str(decoder_template),
            "decoder_roi_txt": str(cfg.decoder_roi_txt) if cfg.decoder_roi_txt else None,
            "reference_score_run": cfg.reference_score_run,
            "reference_score_stats": cfg.reference_score_stats,
            "enable_original_score": cfg.enable_original_score,
            "preprocessing_order": PIPELINE_ORDER,
            "smoothing_fwhm_mm": float(getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0)),
            "analysis_mask": getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain"),
            "analysis_mask_file": getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
            "analysis_mask_space": getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"),
            "fieldmap_dir": str(cfg.fmap_dir),
            "motion_reference": str(cfg.rt_motion_ref_epi),
            "tr": REGRESSOR_SETTINGS.TR,
            "regression": {
                "enable_motion_regression": REGRESSOR_SETTINGS.enable_motion_regression,
                "mot_reg": REGRESSOR_SETTINGS.mot_reg,
                "max_poly_order": REGRESSOR_SETTINGS.max_poly_order,
                "use_gs": REGRESSOR_SETTINGS.use_gs,
                "use_wm": REGRESSOR_SETTINGS.use_wm,
                "use_vent": REGRESSOR_SETTINGS.use_vent,
                "enable_fd_censor_reg": REGRESSOR_SETTINGS.enable_fd_censor_reg,
                "fd_thr": REGRESSOR_SETTINGS.fd_thr,
                "enable_dvars_censor_reg": REGRESSOR_SETTINGS.enable_dvars_censor_reg,
                "dvars_thr_robust_z": REGRESSOR_SETTINGS.dvars_thr_robust_z,
                "censor_plus1": REGRESSOR_SETTINGS.censor_plus1,
                "dvars_warmup": REGRESSOR_SETTINGS.dvars_warmup,
                "dvars_mask_source": REGRESSOR_SETTINGS.dvars_mask_source,
                "analysis_space": REGRESSOR_SETTINGS.analysis_space,
                "voxel_norm_ref_volumes": REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
                "rt_max_scan_length": REGRESSOR_SETTINGS.rt_max_scan_length,
                "pipeline_engine": REGRESSOR_SETTINGS.pipeline_engine,
                "commit_wait_timeout_s": REGRESSOR_SETTINGS.commit_wait_timeout_s,
            },
            "biopac": {
                "enabled": REGRESSOR_SETTINGS.enable_biopac_physio,
                "phys_reg": REGRESSOR_SETTINGS.biopac_phys_reg,
                "mode": REGRESSOR_SETTINGS.biopac_mode,
                "file": str(REGRESSOR_SETTINGS.biopac_file) if REGRESSOR_SETTINGS.biopac_file else None,
                "host": REGRESSOR_SETTINGS.biopac_host,
                "port": REGRESSOR_SETTINGS.biopac_port,
                "timeout": REGRESSOR_SETTINGS.biopac_timeout,
                "handshake": REGRESSOR_SETTINGS.biopac_handshake,
                "start_online_only": REGRESSOR_SETTINGS.biopac_start_online_only,
                "poll_interval": REGRESSOR_SETTINGS.biopac_poll_interval,
            },
        }
    )
    temporary = metadata_path.with_name(".session_metadata.json")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temporary.replace(metadata_path)


# ---------- Filename parsing ----------

def parse_dicom_name(name: str):
    """
    Parse a Siemens-like DICOM filename: 001_000004_000003.dcm
    Returns (series_id, run_id, scan).

    001_000004_000003
     ^    ^       ^
     |    |       +-- scan (volume index)
     |    +---------- "block" in your description
     +--------------- constant "001"
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    series_str, block_str, scan_str = parts
    try:
        series_id = int(series_str)
        run_id = int(block_str)
        scan = int(scan_str)
    except ValueError:
        return None
    return series_id, run_id, scan


# ---------- Watchdog event handler ----------

class DICOMHandler(FileSystemEventHandler):
    def __init__(
        self,
        cfg: RTSessionConfig,
        score_queue: Optional[object] = None,
        start_biopac: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.current_run = int(cfg.run)
        self.next_volume_idx = 1
        self.score_queue = score_queue
        self.biopac_receiver = None
        self._biopac_timeout = None
        self._pending = queue.PriorityQueue()
        self._pending_scans: set[int] = set()
        self._processed_scans: set[int] = set()
        self._inflight_scans: set[int] = set()
        self._lock = threading.Lock()
        self._order_cv = threading.Condition(self._lock)
        self._result_buffer: dict[int, ResultEnvelope] = {}
        self._scan_first_seen: dict[int, float] = {}
        self._next_scan_to_commit: Optional[int] = None
        self._next_scan_to_process: Optional[int] = None
        self._last_committed_scan: int = 0
        self._timed_out_scans: set[int] = set()
        self._engine_mode = str(getattr(REGRESSOR_SETTINGS, "pipeline_engine", "parallel_ordered")).lower()
        self._commit_wait_timeout_s = float(getattr(REGRESSOR_SETTINGS, "commit_wait_timeout_s", 1.0))
        max_workers = int(REGRESSOR_SETTINGS.max_workers)
        if max_workers > 10:
            log.warning(
                "[WATCHDOG] max_workers=%d is high for realtime offline mode; capping to 10.",
                max_workers,
            )
            max_workers = 10
        if max_workers < 1:
            log.warning("[WATCHDOG] max_workers=%d is invalid; using 1.", max_workers)
            max_workers = 1
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        log.info(
            "[ENGINE] mode=%s workers=%d commit_wait_timeout_s=%.2f",
            self._engine_mode,
            max_workers,
            self._commit_wait_timeout_s,
        )
        self._online_mode = False
        self._biopac_started = start_biopac
        self._biopac_run_started = False
        self._biopac_timelag_sum = 0.0
        self._biopac_timelag_count = 0
        self._biopac_timelag_path = self.cfg.rt_work_dir / "biopac_timelag.csv"
        self.reference_score_stats = cfg.reference_score_stats
        self.score_event_tracker = ScoreEventTracker(cfg.rt_work_dir)
        if self.reference_score_stats is not None:
            log.info(
                "[SCORE] Using reference run %s (mean=%.4f, std=%.4f, n=%s, reg_ready=%s)",
                self.reference_score_stats["run"],
                self.reference_score_stats["mean"],
                self.reference_score_stats["std"],
                self.reference_score_stats["n"],
                self.reference_score_stats.get("used_reg_ready", False),
            )
        self.gpu_resampler = maybe_init_gpu_resampler(cfg)
        self.pyhysco_applier = maybe_init_pyhysco_applier(cfg)
        self.smoother = prepare_smoother(
            cfg, getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0),
            str(REGRESSOR_SETTINGS.analysis_space).lower(),
            cfg.rt_unwarped_analysis_ref_epi if REGRESSOR_SETTINGS.analysis_space == "epi"
            else resolve_decoder_template(cfg), run,
            mask_type=getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain"),
            custom_file=getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
            custom_space=getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"),
        )
        self._pyhysco_unwarped_save_notice_emitted = False
        self.volume_streamer = VolumeStreamerHandle(
            VolumeStreamerConfig(
                enabled=bool(getattr(REGRESSOR_SETTINGS, "enable_volume_streamer", False)),
                kind=str(getattr(REGRESSOR_SETTINGS, "volume_stream_kind", "unwarped")),
                every_n=int(getattr(REGRESSOR_SETTINGS, "volume_stream_every_n", 1)),
                max_queue=int(getattr(REGRESSOR_SETTINGS, "volume_stream_max_queue", 2)),
                window_title=str(getattr(REGRESSOR_SETTINGS, "volume_stream_window_title", "RT volume QC")),
            ),
            logger=log,
        )
        self.volume_streamer.start()
        log.info(
            "[FMAP] Effective settings: use_preloaded_pyhysco=%s save_intermediate_unwarped=%s "
            "pyhysco_backend=%s pyhysco_device=%s",
            bool(getattr(REGRESSOR_SETTINGS, "use_preloaded_pyhysco", True)),
            bool(getattr(REGRESSOR_SETTINGS, "save_intermediate_unwarped", True)),
            str(getattr(REGRESSOR_SETTINGS, "pyhysco_backend", "grid_sample")),
            str(getattr(REGRESSOR_SETTINGS, "pyhysco_device", "cuda")),
        )

        # --- RTPSpy Volreg ---
        self.volreg = RtpVolreg(
            regmode='heptic',
            max_scan_length=REGRESSOR_SETTINGS.rt_max_scan_length,
        )
        self.volreg.ignore_init = 0
        self.volreg.save_proc = False

        # --- Motion reference: corrected session mean, fixed across runs ---
        self.ref_set = False
        ref_epi = self.cfg.rt_motion_ref_epi
        if not ref_epi.exists():
            raise FileNotFoundError(
                f"RT reference EPI not found at {ref_epi}. "
                f"Run the offline preprocessing pipeline first so "
                f"epi_unwarped_mean.nii is created in {self.cfg.trans_dir}."
            )
        self.volreg.set_ref_vol(str(ref_epi))
        self.ref_set = True

        # --- Motion / FD state ---
        self.motion_file = self.cfg.rt_work_dir / "motion_rt.1D"
        self.fd_file = self.cfg.rt_work_dir / "fd_rt.csv"
        self.prev_motion = None              # previous 6-vector
        self.brain_radius_mm = 50.0          # standard radius for FD
        self.pre_trial_scans = max(0, int(REGRESSOR_SETTINGS.skip_first_trs))

        # --- NEW: DVARS state ---
        self.prev_mc_for_dvars = None
        self.dvars_hist = []  # store prior DVARS to compute robust stats
        self.censor_next_fd = False
        self.censor_next_dvars = False
        self.last_dvars_val = float("nan")
        self.last_dvars_z = float("nan")

        # --- DVARS uses corrected, motion-aligned volumes and mask ---
        self.dvars_mask = None
        if REGRESSOR_SETTINGS.enable_dvars_censor_reg:
            mpath = cfg.rt_motion_ref_mask
            if not mpath.exists():
                log.warning(f"[DVARS] Mask missing at {mpath}; DVARS censor disabled.")
            else:
                mimg = nib.load(str(mpath))
                mdat = np.asanyarray(mimg.dataobj)
                self.dvars_mask = (mdat > 0.5)
                log.info(f"[DVARS] Using mask for DVARS: {mpath}")

        # --- Nuisance masks (made offline by pipeline.py) ---
        def resolve_mask(label: str, path: Path, enabled: bool) -> Optional[Path]:
            if not enabled:
                log.info(f"[REG] {label} regressor disabled by config.")
                return None
            if not path.exists():
                log.warning(f"[REG] {label} mask missing at {path}; skipping {label} regressor.")
                return None
            return path

        gs = resolve_mask("GS", cfg.rt_unwarped_analysis_ref_mask, REGRESSOR_SETTINGS.use_gs)
        wm = resolve_mask("WM", cfg.rt_wm_mask, REGRESSOR_SETTINGS.use_wm)
        vent = resolve_mask("Vent", cfg.rt_vent_mask, REGRESSOR_SETTINGS.use_vent)

        used = {k: v for k, v in {"GS": gs, "WM": wm, "Vent": vent}.items() if v is not None}
        if used:
            log.info("[REG] Using nuisance masks:\n" + "\n".join([f"  {k}={v}" for k, v in used.items()]))
        else:
            log.info("[REG] No nuisance masks enabled; running motion-only regression.")

        phys_reg = "None"
        rtp_physio = None
        if REGRESSOR_SETTINGS.enable_biopac_physio:
            if not REGRESSOR_SETTINGS.enable_motion_regression:
                log.warning("[BIOPAC] Motion regression disabled; skipping physio regressors.")
            else:
                expected_regressors = {
                    "RICOR8": 8,
                    "RVT5": 5,
                    "RVT+RICOR13": 13,
                }.get(REGRESSOR_SETTINGS.biopac_phys_reg, 8)
                biopac_cfg = BiopacReceiverConfig(
                    host=REGRESSOR_SETTINGS.biopac_host,
                    port=REGRESSOR_SETTINGS.biopac_port,
                    timeout=REGRESSOR_SETTINGS.biopac_timeout,
                    expected_regressors=expected_regressors,
                    handshake_tr=REGRESSOR_SETTINGS.TR if REGRESSOR_SETTINGS.biopac_handshake else None,
                    subject=cfg.subject,
                    day=cfg.day,
                    run=cfg.run,
                    output_path=self.cfg.rt_work_dir / "biopac_regressors_rx.csv",
                )
                if REGRESSOR_SETTINGS.biopac_mode == "file":
                    if REGRESSOR_SETTINGS.biopac_file is None:
                        raise ValueError("[BIOPAC] biopac_mode=file requires biopac_file to be set.")
                    rtp_physio = BiopacRetroTSFileBuffer(
                        REGRESSOR_SETTINGS.biopac_file,
                        timeout=biopac_cfg.timeout,
                        expected_regressors=expected_regressors,
                        poll_interval=REGRESSOR_SETTINGS.biopac_poll_interval,
                    )
                    phys_reg = REGRESSOR_SETTINGS.biopac_phys_reg
                    log.info(
                        "[BIOPAC] Using file-backed physio regressors (%s) from %s",
                        phys_reg,
                        REGRESSOR_SETTINGS.biopac_file,
                    )
                else:
                    self.biopac_receiver = BiopacRetroTSReceiver(biopac_cfg)
                    self._biopac_timeout = biopac_cfg.timeout
                    if start_biopac:
                        self.biopac_receiver.start()
                    else:
                        biopac_cfg.timeout = 0.0
                        log.info("[BIOPAC] Deferring receiver start until online mode.")
                    phys_reg = REGRESSOR_SETTINGS.biopac_phys_reg
                    rtp_physio = self.biopac_receiver
                    log.info(
                        "[BIOPAC] Enabled physio regressors (%s) on %s:%s",
                        phys_reg,
                        biopac_cfg.host,
                        biopac_cfg.port,
                    )

        if REGRESSOR_SETTINGS.enable_motion_regression:
            reg_mask = cfg.rt_unwarped_analysis_ref_mask if cfg.rt_unwarped_analysis_ref_mask.exists() else None
            if reg_mask is None:
                log.warning("[REG] Regression mask missing at %s; using non-zero voxels from first volume.", cfg.rt_unwarped_analysis_ref_mask)
            self.motion_regressor = MotionRegressor(
                self.volreg,
                reg_mask=reg_mask,
                gs_mask=gs,
                wm_mask=wm,
                vent_mask=vent,
                mot_reg=REGRESSOR_SETTINGS.mot_reg,
                max_poly_order=REGRESSOR_SETTINGS.max_poly_order,
                TR=REGRESSOR_SETTINGS.TR,
                max_scan_length=REGRESSOR_SETTINGS.rt_max_scan_length,
                norm_ref_volumes=REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
                enable_fd_censor_reg=REGRESSOR_SETTINGS.enable_fd_censor_reg,
                enable_dvars_censor_reg=REGRESSOR_SETTINGS.enable_dvars_censor_reg,
                phys_reg=phys_reg,
                rtp_physio=rtp_physio,
            )
        else:
            log.info("[REG] Motion regression disabled by config.")
            self.motion_regressor = None

        norm_mask = None
        if cfg.rt_unwarped_analysis_ref_mask.exists():
            norm_mask = np.asanyarray(nib.load(str(cfg.rt_unwarped_analysis_ref_mask)).dataobj) > 0.5
        else:
            log.warning(
                "[REG] Voxel normalization mask missing at %s; using Y-mean nonzero mask only.",
                cfg.rt_unwarped_analysis_ref_mask,
            )
        self.voxel_normalizer = RTPStyleVoxelNormalizer(
            ref_volumes=REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
            brain_mask=norm_mask,
        )

        # --- Source container for GS/WM/Vent regressors (RTPSpy expects mask_src_proc.proc_data) ---
        self.proc_src = ProcSrc()
        if self.motion_regressor is not None:
            self.motion_regressor._regress.mask_src_proc = self.proc_src

        # --- Decoder / scorer ---
        if cfg.enable_scoring:
            decoder_path = resolve_decoder_template(cfg)
            roi_txt = resolve_decoder_roi_txt(cfg)
            if roi_txt is None:
                log.info("[SCORE] Decoder ROI txt not provided; using decoder NIfTI nonzero mask.")
            else:
                log.info("[SCORE] Using decoder ROI txt: %s", roi_txt)

            self.scorer = DecoderScorer(
                decoder_path,
                roi_txt=roi_txt,
                # Keep decoder baseline aligned with the shared normalization
                # window used by the RT pipeline.
                n_baseline=max(1, int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)),
            )
        else:
            log.info("[SCORE] Scoring disabled; skipping decoder initialization.")
            self.scorer = None

    def stop(self):
        if self.biopac_receiver is not None:
            if self._biopac_run_started:
                self.biopac_receiver.send_run_end()
            self.biopac_receiver.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.volume_streamer.stop()

    def start_biopac(self):
        if self.biopac_receiver is None:
            return
        if self._biopac_timeout is not None:
            self.biopac_receiver.config.timeout = self._biopac_timeout
        self.biopac_receiver.start()
        self._biopac_started = True
        if self._online_mode and not self._biopac_run_started:
            self.biopac_receiver.send_run_start()
            self._biopac_run_started = True

    def enable_online_mode(self) -> None:
        self._online_mode = True

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        # `on_created` can fire before the writer closes/flushed the file.
        # We still enqueue as a fallback for filesystems that don't emit
        # close events, but the preferred path is `on_closed` below.
        log.info(f"[WATCHDOG] File created: {path}")
        self.enqueue_path(path)

    def on_closed(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        log.info(f"[WATCHDOG] File closed: {path}")
        self.enqueue_path(path)

    def enqueue_path(self, path: Path) -> None:
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        _, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return
        if self._online_mode and not self._biopac_run_started:
            if not self._biopac_started:
                self.start_biopac()
            elif self.biopac_receiver is not None:
                self.biopac_receiver.send_run_start()
                self._biopac_run_started = True
        if scan in self._processed_scans or scan in self._pending_scans:
            return
        self._pending.put((scan, path))
        self._pending_scans.add(scan)

    def next_pending(self, timeout: float = 0.1) -> Optional[tuple[int, Path]]:
        try:
            scan, path = self._pending.get(timeout=timeout)
        except queue.Empty:
            return None
        self._pending_scans.discard(scan)
        return scan, path

    def mark_processed(self, scan: int) -> None:
        self._processed_scans.add(scan)

    def submit_pending(self) -> bool:
        pending = self.next_pending(timeout=0.0)
        if pending is None:
            return False
        scan, path = pending
        with self._lock:
            if scan in self._processed_scans or scan in self._inflight_scans:
                return True
            volume_idx = scan
            self._inflight_scans.add(scan)
            self._scan_first_seen.setdefault(scan, time.time())
            if self._next_scan_to_commit is None:
                self._next_scan_to_commit = scan
        if self._engine_mode == "legacy":
            fut = self._executor.submit(self._process_scan, path, scan, volume_idx)
            fut.add_done_callback(lambda f, s=scan, v=volume_idx: self._on_scan_future_done(f, s, v))
        else:
            fut = self._executor.submit(self._compute_scan, path, scan, volume_idx)
            fut.add_done_callback(self._on_compute_future_done)
        return True

    def _on_scan_future_done(self, future, scan: int, volume_idx: int) -> None:
        try:
            exc = future.exception()
        except CancelledError:
            return
        if exc is None:
            return

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log.error(
            "[WATCHDOG] Unhandled exception for scan %s (vol %05d): %s\n%s",
            scan,
            volume_idx,
            exc,
            tb,
        )

        with self._lock:
            if scan in self._inflight_scans:
                self._inflight_scans.discard(scan)
                self.mark_processed(scan)
                self._advance_expected_scan_locked()

    def _advance_expected_scan_locked(self) -> None:
        all_candidates = self._inflight_scans.union(self._pending_scans).union(self._result_buffer.keys())
        candidates = sorted(s for s in all_candidates if s > self._last_committed_scan)
        self._next_scan_to_commit = candidates[0] if candidates else None
        self._next_scan_to_process = self._next_scan_to_commit
        self._order_cv.notify_all()

    def _on_compute_future_done(self, future) -> None:
        try:
            env: ResultEnvelope = future.result()
        except Exception as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log.error("[ENGINE] compute future failed: %s\n%s", exc, tb)
            return
        with self._lock:
            if env.scan <= self._last_committed_scan or env.scan in self._timed_out_scans:
                log.warning("[ENGINE] dropping late envelope scan=%s (last_committed=%s timed_out=%s)", env.scan, self._last_committed_scan, env.scan in self._timed_out_scans)
                return
            self._result_buffer[env.scan] = env
            self._order_cv.notify_all()
        self._drain_commit_ready()

    def _drain_commit_ready(self) -> None:
        while True:
            with self._lock:
                if self._next_scan_to_commit is None:
                    self._advance_expected_scan_locked()
                    if self._next_scan_to_commit is None:
                        return
                expected_scan = self._next_scan_to_commit
                env = self._result_buffer.pop(expected_scan, None)
                if env is None:
                    has_newer_ready = any(s > expected_scan for s in self._result_buffer)
                    is_missing = (expected_scan not in self._inflight_scans) and (expected_scan not in self._pending_scans)
                    if is_missing and has_newer_ready:
                        first_seen = self._scan_first_seen.get(expected_scan, time.time())
                        if time.time() - first_seen > self._commit_wait_timeout_s:
                            env = ResultEnvelope(
                                scan=expected_scan,
                                volume_idx=expected_scan,
                                dicom_path=Path("missing"),
                                volume_timestamp=time.time(),
                                success=False,
                                error=f"Commit timeout waiting for scan {expected_scan}",
                            )
                            self._timed_out_scans.add(expected_scan)
                            self._pending_scans.discard(expected_scan)
                    if env is None:
                        return
            self._commit_scan(env)

    def _process_scan(self, path: Path, scan: int, volume_idx: int) -> None:
        # Legacy path (kept for backward compatibility).
        log.info(f"[WATCHDOG] Preparing volume idx {volume_idx} (run={self.current_run}, scan={scan})")

        prepared_raw: Optional[Path] = None
        volume_timestamp: Optional[float] = None
        for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
            prepared_raw, volume_timestamp = prepare_volume_input(self.cfg, path, volume_idx)
            if prepared_raw is not None:
                break
            log.warning(
                "[WATCHDOG] Retry scan %s prep (attempt %s/%s)",
                scan,
                attempt,
                REGRESSOR_SETTINGS.max_retries,
            )
            time.sleep(0.2)

        ok = False
        if prepared_raw is not None and volume_timestamp is not None:
            with self._order_cv:
                while self._next_scan_to_process is not None and scan != self._next_scan_to_process:
                    self._order_cv.wait(timeout=0.1)
            log.info(f"[WATCHDOG] Processing volume idx {volume_idx} (run={self.current_run}, scan={scan})")
            for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
                ok = process_volume(
                    self.cfg,
                    self,
                    path,
                    volume_idx,
                    raw_nii=prepared_raw,
                    volume_timestamp=volume_timestamp,
                )
                if ok:
                    break
                log.warning(
                    "[WATCHDOG] Retry scan %s post-prep (attempt %s/%s)",
                    scan,
                    attempt,
                    REGRESSOR_SETTINGS.max_retries,
                )
                time.sleep(0.2)

        if not ok:
            log.error("[WATCHDOG] Giving up on scan %s after %s attempts.", scan, REGRESSOR_SETTINGS.max_retries)
        with self._lock:
            self._inflight_scans.discard(scan)
            self.mark_processed(scan)
            self._advance_expected_scan_locked()

    def _compute_scan(self, path: Path, scan: int, volume_idx: int) -> ResultEnvelope:
        err_txt = None
        tb_txt = None
        for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
            try:
                env = compute_stage(self.cfg, path, scan=scan, volume_idx=volume_idx, attempt=attempt)
                env.attempts = attempt
                return env
            except Exception as exc:
                err_txt = str(exc)
                tb_txt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                log.warning("[ENGINE] compute retry scan=%s attempt=%s/%s err=%s", scan, attempt, REGRESSOR_SETTINGS.max_retries, exc)
                time.sleep(0.2)
        return ResultEnvelope(
            scan=scan,
            volume_idx=volume_idx,
            dicom_path=path,
            volume_timestamp=time.time(),
            success=False,
            error=err_txt or "unknown compute failure",
            traceback_text=tb_txt,
            attempts=REGRESSOR_SETTINGS.max_retries,
        )

    def _commit_scan(self, env: ResultEnvelope) -> None:
        try:
            commit_stage(self.cfg, self, env)
        except Exception as exc:
            log.error("[ENGINE] commit failure scan=%s vol=%05d err=%s", env.scan, env.volume_idx, exc)
        finally:
            with self._lock:
                if env.scan <= self._last_committed_scan:
                    log.warning("[COMMIT] Ignoring already-committed/late scan=%s (last=%s)", env.scan, self._last_committed_scan)
                else:
                    self._last_committed_scan = env.scan
                self._inflight_scans.discard(env.scan)
                self.mark_processed(env.scan)
                self._scan_first_seen.pop(env.scan, None)
                self._advance_expected_scan_locked()

    def process_file(self, path: Path):
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        _, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return

        if scan in self._processed_scans:
            return
        self.enqueue_path(path)


# ---------- Core processing hook (DICOM -> NIfTI -> MC -> FMAP) ----------

def prepare_volume_input(cfg: RTSessionConfig, dicom_path: Path, volume_idx: int) -> tuple[Optional[Path], Optional[float]]:
    """
    Convert DICOM to raw NIfTI.
    This stage is safe to run ahead of ordered/stateful realtime steps.
    """
    volume_timestamp = time.time()

    # ---------- 1) DICOM -> raw NIfTI ----------
    t0 = volume_timestamp
    raw_dir = cfg.rt_raw_dir
    raw_nii = raw_dir / f"vol_{volume_idx:05d}.nii"

    if not raw_nii.exists():
        for attempt in range(3):
            run([
                "dcm2niix",
                "-z", "n",  # no gzip
                "-s", "y",
                "-a", "y",
                "-b", "y",
                "-v", "0",
                "-f", f"vol_{volume_idx:05d}",
                "-o", str(raw_dir),
                str(dicom_path),
            ])

            produced = sorted(raw_dir.glob(f"vol_{volume_idx:05d}*.nii*"))
            if produced:
                raw_nii = produced[0]
                break
            log.warning("[DICOM] vol %05d conversion retry %d/3", volume_idx, attempt + 1)
            time.sleep(0.2)
        else:
            log.error(f"[DICOM] vol {volume_idx:05d} FAILED (no output)")
            return None, None

    log_step("DICOM", volume_idx, start_t=t0)

    return raw_nii, volume_timestamp


def run_worker_local_volreg(ref_epi: Path, raw_nii: Path, mc_nii: Path, volume_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volreg = RtpVolreg(regmode='heptic')
    volreg.ignore_init = 0
    volreg.save_proc = False
    volreg.set_ref_vol(str(ref_epi))
    img = nib.load(str(raw_nii))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))
    volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    nib.save(nib.Nifti1Image(mc_data, img.affine), str(mc_nii))
    try:
        motion_vec = np.asarray(volreg._motion[volume_idx - 1]).astype(float)
    except Exception:
        motion_vec = np.zeros(6, dtype=float)
    return mc_data, img.affine, motion_vec


def compute_stage(cfg: RTSessionConfig, dicom_path: Path, scan: int, volume_idx: int, attempt: int = 1) -> ResultEnvelope:
    raw_nii, volume_timestamp = prepare_volume_input(cfg, dicom_path, volume_idx)
    if raw_nii is None or volume_timestamp is None:
        raise RuntimeError("prepare_volume_input failed")
    return ResultEnvelope(
        scan=scan,
        volume_idx=volume_idx,
        dicom_path=dicom_path,
        volume_timestamp=volume_timestamp,
        success=True,
        attempts=attempt,
        raw_nii=raw_nii,
    )


def commit_stage(cfg: RTSessionConfig, handler: "DICOMHandler", env: ResultEnvelope) -> bool:
    if not env.success:
        log.error("[COMMIT] scan=%s failed in compute: %s", env.scan, env.error)
        return False
    if env.raw_nii is None:
        log.error("[COMMIT] scan=%s missing raw input.", env.scan)
        return False
    return process_volume(
        cfg,
        handler,
        env.dicom_path,
        env.volume_idx,
        raw_nii=env.raw_nii,
        volume_timestamp=env.volume_timestamp,
    )


def process_volume(
    cfg: RTSessionConfig,
    handler: "DICOMHandler",
    dicom_path: Path,
    volume_idx: int,
    raw_nii: Optional[Path] = None,
    volume_timestamp: Optional[float] = None,
):
    """
    For each incoming DICOM:
      1) DICOM -> raw NIfTI (single volume)
      2) Fieldmap unwarp of raw volume
      3) Motion correction of unwarped volume -> mc NIfTI
      4) Space selection: EPI passthrough, EPI->T1, or EPI->T1->MNI
    """

    if raw_nii is None:
        raw_nii, prepared_timestamp = prepare_volume_input(cfg, dicom_path, volume_idx)
        if raw_nii is None or prepared_timestamp is None:
            return False
        volume_timestamp = prepared_timestamp
    elif volume_timestamp is None:
        volume_timestamp = time.time()

    # Unwarp raw BOLD in calibration geometry before estimating motion.
    uw_t0 = time.time()
    raw_img = nib.load(str(raw_nii))
    validate_bold(raw_img, cfg.fmap_dir, str(REGRESSOR_SETTINGS.epi_phase_encoding).upper(), raw_nii)
    ref_img = nib.load(str(cfg.rt_motion_ref_epi))
    require_same_grid(raw_img, ref_img)
    raw_data = np.asarray(raw_img.dataobj, dtype=np.float32)
    unwarp_dir = cfg.rt_unwarp_dir
    local_unwarped = unwarp_dir / f"vol_{volume_idx:05d}_uw_native.nii"
    corrected = None
    if handler.pyhysco_applier is not None:
        try:
            corrected = handler.pyhysco_applier.apply_volume(raw_data)
        except Exception:
            log.exception("[FMAP] Preloaded apply failed; using file-based PyHySCO.")
    if corrected is None:
        if not unwarp_volume(raw_nii, local_unwarped, cfg):
            return False
        corrected = np.asarray(nib.load(str(local_unwarped)).dataobj, dtype=np.float32).copy()
    img = nib.Nifti1Image(corrected, raw_img.affine)
    if bool(getattr(REGRESSOR_SETTINGS, "save_intermediate_unwarped", True)):
        nib.save(img, str(local_unwarped))
    log_step("FMAP", volume_idx, start_t=uw_t0)

    # ---------- 2) Motion correction (RtpVolreg) ----------
    t0 = time.time()
    mc_dir = cfg.rt_mc_dir
    mc_nii = mc_dir / f"vol_{volume_idx:05d}_mc.nii"

    # MC input is now corrected BOLD.
    data = np.asanyarray(img.dataobj).astype(np.float32)

    # Create a temporary NIfTI for RtpVolreg to work on
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))

    # Run RTPSpy volreg (works in-place on tmp_img.dataobj)
    handler.volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)

    # Extract corrected data and save with a FRESH header
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    mc_img = nib.Nifti1Image(mc_data, ref_img.affine)  # fixed corrected reference grid
    nib.save(mc_img, str(mc_nii))

    # One on-disk product for fully corrected BOLD, shared by all consumers.
    mc_unwarped_nii = mc_nii
    mc_unwarped_img = mc_img

    # ----- 2b) MOTION + FD (ONLINE) -----
    # RtpVolreg exposes [roll pitch yaw dS dL dP]: rotations in degrees,
    # followed by translations in mm.
    try:
        motion_vec = np.asarray(handler.volreg._motion[volume_idx - 1]).astype(float)  # shape (6,)
    except Exception as e:
        log.error(f"[MC] Could not read motion for vol {volume_idx:05d}: {e}")
        motion_vec = np.zeros(6, dtype=float)

    # Save raw motion parameters (AFNI-style 1D)
    append_motion(handler.motion_file, motion_vec)

    # Compute delta motion relative to previous volume
    if handler.prev_motion is None:
        delta = np.zeros_like(motion_vec)
    else:
        delta = motion_vec - handler.prev_motion
    handler.prev_motion = motion_vec.copy()

    # Framewise displacement: translation plus radius-scaled rotation.
    fd_value = float(fd_from_rtpspy_delta(delta, handler.brain_radius_mm))

    # Optionally emulate pre_trial_scan_num behavior:
    if volume_idx <= handler.pre_trial_scans:
        fd_to_save = float("nan")
    else:
        fd_to_save = fd_value

    append_fd(handler.fd_file, volume_idx, fd_to_save)

    log_step(
        "MC",
        volume_idx,
        extra=f"FD={fd_to_save:.4f}" if np.isfinite(fd_to_save) else "FD=NaN (pre-trial)",
        start_t=t0,
    )

    # ---------- 2b.5) CENSOR FLAGS (FD + DVARS) ----------
    fd_censor = 0
    dvars_censor = 0

    # --- FD censor: FD > thr plus +1 TR ---
    if REGRESSOR_SETTINGS.enable_fd_censor_reg:
        hit_fd = np.isfinite(fd_to_save) and (fd_value > REGRESSOR_SETTINGS.fd_thr)
        if handler.censor_next_fd:
            fd_censor = 1
            handler.censor_next_fd = False
        if hit_fd:
            fd_censor = 1
            if REGRESSOR_SETTINGS.censor_plus1:
                handler.censor_next_fd = True

    # --- DVARS censor: raw DVARS robust_z > thr plus +1 TR ---
    if REGRESSOR_SETTINGS.enable_dvars_censor_reg and handler.dvars_mask is not None:
        if handler.prev_mc_for_dvars is not None:
            dvars_val = compute_dvars(handler.prev_mc_for_dvars, mc_data, handler.dvars_mask)

            # robust z against prior DVARS history (exclude current)
            z = robust_z(dvars_val, handler.dvars_hist)

            # update history after computing z
            handler.dvars_hist.append(dvars_val)

            # save to a file
            append_dvars(
                handler.cfg.rt_work_dir / "dvars_rt.csv",
                volume_idx,
                fd_to_save,
                dvars_val,
                z
            )
            handler.last_dvars_val = dvars_val
            handler.last_dvars_z = z

            # optional: ignore first few DVARS for robust stats stability
            enough = (len(handler.dvars_hist) >= REGRESSOR_SETTINGS.dvars_warmup)
            hit_dvars = enough and np.isfinite(z) and (z > REGRESSOR_SETTINGS.dvars_thr_robust_z)

            if handler.censor_next_dvars:
                dvars_censor = 1
                handler.censor_next_dvars = False
            if hit_dvars:
                dvars_censor = 1
                if REGRESSOR_SETTINGS.censor_plus1:
                    handler.censor_next_dvars = True
        else:
            # first timepoint: no DVARS
            handler.dvars_hist.append(0.0)
            handler.last_dvars_val = float("nan")
            handler.last_dvars_z = float("nan")

    # update prev for next DVARS
    handler.prev_mc_for_dvars = mc_data.copy()


    # ---------- 2d) Motion regression (RTPS_py) ----------
    reg_t0 = time.time()
    mc_for_warp = mc_unwarped_nii
    if handler.motion_regressor is not None:
        handler.proc_src.proc_data = np.asanyarray(mc_unwarped_img.dataobj)
        cleaned, reg_ready = handler.motion_regressor.apply(
            mc_unwarped_img,
            volume_idx,
            fd_censor=fd_censor,
            dvars_censor=dvars_censor,
        )
        reg_dir = cfg.rt_reg_dir
        reg_nii = reg_dir / f"vol_{volume_idx:05d}_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, mc_unwarped_img.affine), str(reg_nii))
        mc_for_warp = reg_nii
        log_step("REG", volume_idx, "motion", start_t=reg_t0)
    else:
        cleaned = handler.voxel_normalizer.apply(np.asanyarray(mc_unwarped_img.dataobj))
        reg_ready = True
        reg_dir = cfg.rt_reg_dir
        reg_nii = reg_dir / f"vol_{volume_idx:05d}_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, mc_unwarped_img.affine), str(reg_nii))
        mc_for_warp = reg_nii
        log_step("REG", volume_idx, "skipped (voxel-normalized)", start_t=reg_t0)

    if handler.motion_regressor is not None:
        reg_names, reg_row = handler.motion_regressor.get_regressors(volume_idx)
        if reg_names and reg_row is not None:
            append_regressors(handler.cfg.rt_work_dir / "regressors_rt.csv", volume_idx, reg_names, reg_row)

    biopac_missing = False
    if handler.biopac_receiver is not None:
        biopac_missing = handler.biopac_receiver.was_missing(volume_idx)
    append_regression_status(
        handler.cfg.rt_work_dir / "regression_status_rt.csv",
        volume_idx,
        fd_to_save,
        handler.last_dvars_val,
        handler.last_dvars_z,
        fd_censor,
        dvars_censor,
        reg_ready,
        biopac_missing,
    )
    if REGRESSOR_SETTINGS.biopac_timelag and handler.biopac_receiver is not None:
        trigger_ts = handler.biopac_receiver.get_trigger_timestamp(volume_idx)
        if trigger_ts is not None:
            timelag = volume_timestamp - trigger_ts
            handler._biopac_timelag_sum += timelag
            handler._biopac_timelag_count += 1
            avg_timelag = handler._biopac_timelag_sum / handler._biopac_timelag_count
            append_biopac_timelag(
                handler._biopac_timelag_path,
                volume_idx,
                trigger_ts,
                volume_timestamp,
                timelag,
                avg_timelag,
            )

    # ---------- 3) Space handling (EPI passthrough, EPI→T1, or EPI→T1→MNI) ----------
    t0 = time.time()
    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    score_input_nii = mc_for_warp
    score_input_orig_nii: Optional[Path] = None
    if cfg.enable_original_score and mc_unwarped_nii.exists():
        score_input_orig_nii = mc_unwarped_nii

    if analysis_space == "epi":
        log_step("TRANS", volume_idx, "skipped (EPI space)", start_t=t0)
    elif analysis_space == "t1":
        t1_dir = cfg.rt_work_dir / "t1"
        t1_dir.mkdir(parents=True, exist_ok=True)
        t1_nii = t1_dir / f"vol_{volume_idx:05d}_t1.nii"

        epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
        decoder_template = resolve_decoder_template(cfg)
        t1_ref = decoder_template

        if not decoder_template.exists():
            log.error(f"Decoder template not found at {decoder_template}")
            return False

        if not epi2t1.exists():
            log.error(f"Missing EPI→T1 transform in {cfg.trans_dir}")
            return False

        if handler.gpu_resampler is not None:
            handler.gpu_resampler.resample_nifti_to_nifti(mc_for_warp, t1_nii)
        else:
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", str(mc_for_warp),
                "-r", str(t1_ref),
                "-o", str(t1_nii),
                "-t", str(epi2t1),
                "-n", "Linear",
                "--float", "1",
            ]
            run(cmd)
        score_input_nii = t1_nii

        # Optional: also save pre-denoise / pre-normalization score source.
        if cfg.enable_original_score:
            t1_orig_nii = t1_dir / f"vol_{volume_idx:05d}_t1_orig.nii"
            if handler.gpu_resampler is not None:
                handler.gpu_resampler.resample_nifti_to_nifti(mc_unwarped_nii, t1_orig_nii)
            else:
                cmd_orig = [
                    "antsApplyTransforms",
                    "-d", "3",
                    "-i", str(mc_unwarped_nii),
                    "-r", str(t1_ref),
                    "-o", str(t1_orig_nii),
                    "-t", str(epi2t1),
                    "-n", "Linear",
                    "--float", "1",
                ]
                run(cmd_orig)
            score_input_orig_nii = t1_orig_nii
        log_step("TRANS", volume_idx, "warp→T1", start_t=t0)
    elif analysis_space == "mni":
        mni_dir = cfg.rt_mni_dir
        mni_nii = mni_dir / f"vol_{volume_idx:05d}_mni.nii"

        warp_t1_mni = cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii"
        epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
        decoder_template = resolve_decoder_template(cfg)

        if not decoder_template.exists():
            log.error(f"Decoder template not found at {decoder_template}")
            return False

        if not (warp_t1_mni.exists() and epi2t1.exists()):
            log.error(f"Missing transforms in {cfg.trans_dir}")
            return False

        if handler.gpu_resampler is not None:
            handler.gpu_resampler.resample_nifti_to_nifti(mc_for_warp, mni_nii)
        else:
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", str(mc_for_warp),
                "-r", str(decoder_template),
                "-o", str(mni_nii),
                "-t", str(warp_t1_mni),
                "-t", str(epi2t1),
                "-n", "Linear",
                "--float", "1",
            ]
            run(cmd)
        score_input_nii = mni_nii

        # Optional: also save pre-denoise / pre-normalization score source.
        if cfg.enable_original_score:
            mni_orig_nii = mni_dir / f"vol_{volume_idx:05d}_mni_orig.nii"
            if handler.gpu_resampler is not None:
                handler.gpu_resampler.resample_nifti_to_nifti(mc_unwarped_nii, mni_orig_nii)
            else:
                cmd_orig = [
                    "antsApplyTransforms",
                    "-d", "3",
                    "-i", str(mc_unwarped_nii),
                    "-r", str(decoder_template),
                    "-o", str(mni_orig_nii),
                    "-t", str(warp_t1_mni),
                    "-t", str(epi2t1),
                    "-n", "Linear",
                    "--float", "1",
                ]
                run(cmd_orig)
            score_input_orig_nii = mni_orig_nii
        log_step("TRANS", volume_idx, "warp→MNI", start_t=t0)
    else:
        log.error(
            "Unsupported analysis_space=%r. Expected 'epi', 't1', or 'mni'.",
            REGRESSOR_SETTINGS.analysis_space,
        )
        return False

    # Final image operation, after spatial transforms and before all scoring.
    smoother = getattr(handler, "smoother", None)
    if smoother is not None:
        smooth_t0 = time.time()
        kind = smoother.output_kind
        folder = cfg.rt_work_dir / kind
        score_input_nii = smooth_file(smoother, score_input_nii,
                                      folder / f"vol_{volume_idx:05d}_{kind}.nii")
        if score_input_orig_nii is not None:
            score_input_orig_nii = smooth_file(smoother, score_input_orig_nii,
                folder / f"vol_{volume_idx:05d}_{kind}_orig.nii")
        log_step("SMOOTH" if kind == "smooth" else "MASK", volume_idx, start_t=smooth_t0)

    stream_kind = str(getattr(REGRESSOR_SETTINGS, "volume_stream_kind", "unwarped")).lower()
    stream_paths = {
        "raw": raw_nii,
        "mc": mc_nii,
        "unwarped": mc_nii,  # compatibility alias for fully corrected BOLD
        "reg": reg_nii,
        "score_input": score_input_nii,
    }
    handler.volume_streamer.publish(volume_idx, stream_paths.get(stream_kind))

    if not cfg.enable_scoring:
        if handler.score_queue is not None:
            try:
                handler.score_queue.put_nowait(
                    {
                        "kind": "tick",
                        "volume_idx": volume_idx,
                        "watchdog_timestamp": volume_timestamp,
                    }
                )
            except Exception as exc:
                log.error(f"[SCORE] Failed to enqueue volume {volume_idx:05d}: {exc}")
        return True

    # ---------- 4) Decoder scoring ----------
    t0 = time.time()
    try:
        # Load the warped volume (decoder/ROI space)
        score_img = nib.load(str(score_input_nii))
        score_data = np.asanyarray(score_img.dataobj)
        original_score = None
        if score_input_orig_nii is not None:
            score_orig_img = nib.load(str(score_input_orig_nii))
            score_orig_data = np.asanyarray(score_orig_img.dataobj)
            original_score = handler.scorer.score_from_array(score_orig_data)

        # Only accumulate baseline from *denoised* volumes
        if reg_ready and handler.scorer.baseline_count < handler.scorer.n_baseline:
            handler.scorer.accumulate_baseline(score_data)
            if handler.scorer.baseline_count == handler.scorer.n_baseline:
                handler.scorer.finalize_baseline()

        # Always compute raw; z will be NaN until baseline_ready
        raw_score = handler.scorer.score_from_array(score_data)
        event_type = handler.score_event_tracker.for_volume(volume_idx)
        analysis_timestamp = time.time()
        timestamp = append_score(
            cfg.rt_work_dir / "scores.csv",
            volume_idx,
            raw_score,
            original_score=original_score,
            reg_ready=reg_ready,
            timestamp=analysis_timestamp,
            event_type=event_type,
        )
        z_score = None
        if handler.reference_score_stats is not None:
            stats = handler.reference_score_stats
            z_score = (raw_score - stats["mean"]) / stats["std"]
            append_score_z(
                cfg.rt_work_dir / "scores_z.csv",
                volume_idx,
                timestamp,
                raw_score,
                z_score,
                stats,
                reg_ready=reg_ready,
            )
        if handler.score_queue is not None:
            try:
                payload = {
                    "kind": "score",
                    "volume_idx": volume_idx,
                    "timestamp": timestamp,
                    "analysis_timestamp": analysis_timestamp,
                    "watchdog_timestamp": volume_timestamp,
                    "score_raw": raw_score,
                    "score_original": original_score,
                    "reg_ready": reg_ready,
                    "event_type": event_type,
                }
                if z_score is not None:
                    payload["score_z"] = z_score
                handler.score_queue.put_nowait(payload)
            except Exception as exc:
                log.error(f"[SCORE] Failed to enqueue score for vol {volume_idx:05d}: {exc}")

        if z_score is None:
            extra = f"raw={raw_score:.4f}"
        else:
            extra = f"raw={raw_score:.4f} z={z_score:.4f}"

        log_step("SCORE", volume_idx, extra, start_t=t0)

    except Exception as e:
        log.error(f"[SCORE] Failed scoring vol {volume_idx:05d}: {e}")
        return False

    return True






def unwarp_volume(raw_nii: Path, out_nii: Path, cfg: RTSessionConfig):
    if str(REGRESSOR_SETTINGS.fieldmap_method).lower() != "pyhysco":
        raise ValueError("unwarp -> MC supports PyHySCO only.")
    pe = str(REGRESSOR_SETTINGS.epi_phase_encoding).upper()
    spec = validate_bold(nib.load(str(raw_nii)), cfg.fmap_dir, pe, raw_nii)
    apply_pyhysco_fieldmap(
        epi_path=raw_nii, fieldmap_path=_preferred_pyhysco_fieldmap(cfg.fmap_dir),
        out_path=out_nii, phase_encoding_direction=spec["phase_encoding_axis"],
        polarity=1 if pe == "AP" else -1,
    )
    return True


def validate_run_provenance(cfg):
    """Do not mix old MC-first outputs or a different field/reference in a run."""
    folder = cfg.rt_work_dir
    marker = folder / "preprocessing_order.json"
    payload = {"order": PIPELINE_ORDER, "calibration": load_calibration(cfg.fmap_dir),
               "field": source_identity(_preferred_pyhysco_fieldmap(cfg.fmap_dir)),
               "reference": source_identity(cfg.rt_motion_ref_epi),
               "epi_phase_encoding": str(REGRESSOR_SETTINGS.epi_phase_encoding).upper()}
    fwhm = float(getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0))
    if not np.isfinite(fwhm) or fwhm < 0:
        raise ValueError("smoothing_fwhm_mm must be finite and >= 0.")
    mask_type = getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain")
    if final_output_kind(fwhm, mask_type) is not None:
        space = str(REGRESSOR_SETTINGS.analysis_space).lower()
        reference = cfg.rt_unwarped_analysis_ref_epi if space == "epi" else resolve_decoder_template(cfg)
        payload["final_masking"] = {"fwhm_mm": fwhm, "space": space,
            **mask_provenance(cfg, mask_type, space, reference,
                getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
                getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"))}
    if marker.exists():
        if json.loads(marker.read_text()) != payload:
            raise ValueError("Run preprocessing/calibration/reference changed. Use a fresh run output directory.")
    elif any((folder / name).exists() and any((folder / name).glob("*.nii*"))
             for name in ("mc", "unwarped", "reg", "t1", "mni", "smooth", "masked")):
        raise ValueError("Legacy processed run detected. Use a fresh output directory for unwarp -> MC replay.")
    marker.write_text(json.dumps(payload, indent=2))


# ---------- Main ----------

def run_rt_pipeline(cfg: RTSessionConfig, score_queue: Optional[object] = None, stop_event: Optional[object] = None):
    """Run the real-time pipeline until Ctrl-C or, when `stop_event` (a multiprocessing
    Event, see rt_session) is given, until it is set; either way the handler's
    cleanup runs. With stop_event=None behavior is unchanged."""
    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    if analysis_space not in {"epi", "t1", "mni"}:
        raise ValueError(
            f"Unsupported analysis_space={REGRESSOR_SETTINGS.analysis_space!r}. "
            "Use 'epi', 't1', or 'mni'."
        )
    if cfg.enable_scoring and analysis_space in {"epi", "t1"} and cfg.decoder_template is None:
        raise ValueError(
            "EPI/T1-space scoring requires --decoder-template pointing to a decoder in the selected space."
        )
    decoder_template = resolve_decoder_template(cfg)

    ensure_fieldmap_only(cfg)
    validate_run_provenance(cfg)

    cfg.reference_score_stats = load_reference_score_stats(cfg, cfg.reference_score_run)
    write_session_metadata(cfg, decoder_template)
    # Process existing DICOMs first (offline-style), but only for this run
    existing = []
    for path in sorted(cfg.incoming_dir.glob("*.dcm")):
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            continue
        _, run_id, _ = parsed
        if run_id == int(cfg.run):
            existing.append(path)
    defer_biopac = bool(existing) or REGRESSOR_SETTINGS.biopac_start_online_only
    event_handler = DICOMHandler(
        cfg,
        score_queue=score_queue,
        start_biopac=not defer_biopac,
    )
    observer = Observer()
    observer.schedule(event_handler, str(cfg.incoming_dir), recursive=False)

    observer.start()
    if existing:
        print(f"[RT] Found {len(existing)} existing DICOMs — processing offline first…")
        for f in existing:
            event_handler.enqueue_path(Path(f))
        while event_handler.submit_pending():
            continue

    if defer_biopac:
        event_handler.enable_online_mode()

    print("[RT] Switching to online mode.")
    try:
        event_handler.enable_online_mode()
        while stop_event is None or not stop_event.is_set():
            while event_handler.submit_pending():
                continue
            time.sleep(0.2)
        print("[RT] Stop requested; shutting down.")
        observer.stop()
    except KeyboardInterrupt:
        observer.stop()
    finally:
        event_handler.stop()
    observer.join()

def compute_dvars(prev_vol: np.ndarray, cur_vol: np.ndarray, mask: np.ndarray) -> float:
    diff = (cur_vol - prev_vol)
    if mask is not None:
        diff = diff[mask]
    diff = diff.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))

def robust_z(x: float, history: list[float]) -> float:
    # robust z using median and MAD from history
    if len(history) < 5:
        return float("nan")
    med = float(np.median(history))
    mad = float(np.median(np.abs(np.asarray(history) - med)))
    if mad < 1e-8:
        return float("nan")
    return float((x - med) / (1.4826 * mad))

def append_dvars(qc_path: Path, volume_idx: int, fd: float, dvars: float, dvars_z: float):
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    exists = qc_path.exists()
    with open(qc_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", "fd", "dvars", "robust_z"])
        w.writerow([volume_idx, fd, dvars, dvars_z])


def append_regressors(reg_path: Path, volume_idx: int, reg_names: list[str], reg_row: np.ndarray):
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    exists = reg_path.exists()
    with open(reg_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", *reg_names])
        w.writerow([volume_idx, *reg_row.tolist()])


def append_regression_status(
    qc_path: Path,
    volume_idx: int,
    fd: float,
    dvars: float,
    dvars_z: float,
    fd_censor: int,
    dvars_censor: int,
    reg_ready: bool,
    biopac_missing: bool,
):
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    exists = qc_path.exists()
    with open(qc_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "volume_idx",
                    "fd",
                    "dvars",
                    "dvars_z",
                    "fd_censor",
                    "dvars_censor",
                    "reg_ready",
                    "biopac_missing",
                ]
            )
        w.writerow([volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, int(reg_ready), int(biopac_missing)])


def append_biopac_timelag(
    path: Path,
    volume_idx: int,
    trigger_timestamp: float,
    volume_timestamp: float,
    timelag_s: float,
    avg_timelag_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "volume_idx",
                    "trigger_timestamp",
                    "volume_timestamp",
                    "timelag_s",
                    "avg_timelag_s",
                ]
            )
        w.writerow(
            [
                volume_idx,
                f"{trigger_timestamp:.6f}",
                f"{volume_timestamp:.6f}",
                f"{timelag_s:.6f}",
                f"{avg_timelag_s:.6f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Real-time fMRI watcher pipeline")
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4 (matches 000004 in DICOM name)")
    parser.add_argument(
        "--ap-block",
        type=int,
        default=None,
        help="AP block to use for b0",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        default=None,
        help="PA block to use for b0",
    )
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--incoming-root",
        required=False,
        default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya",
        help="Folder where scanner writes DICOMs in real-time.",
    )
    parser.add_argument(
        "--base-data",
        required=False,
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder (same as offline pipeline).",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    parser.add_argument(
        "--decoder-roi-txt",
        required=False,
        help="Optional ROI_DECODER-style text file used for scoring mask/weights. If omitted, scoring uses decoder NIfTI nonzero voxels.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable decoder scoring (still runs motion correction + warps).",
    )
    parser.add_argument(
        "--enable-original-score",
        action="store_true",
        help="Also compute score_original from a separately warped pre-denoise/pre-normalization volume.",
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP.",
    )
    parser.add_argument(
        "--biopac-host",
        default=REGRESSOR_SETTINGS.biopac_host,
        help="Host to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-port",
        type=int,
        default=REGRESSOR_SETTINGS.biopac_port,
        help="Port to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-timeout",
        type=float,
        default=REGRESSOR_SETTINGS.biopac_timeout,
        help="Seconds to wait for physio regressors before zero-fill.",
    )
    parser.add_argument(
        "--biopac-phys-reg",
        default=REGRESSOR_SETTINGS.biopac_phys_reg,
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
        help="Physio regressor family to expect from BIOPAC stream.",
    )
    parser.add_argument(
        "--biopac-mode",
        default=REGRESSOR_SETTINGS.biopac_mode,
        choices=["tcp", "file"],
        help="BIOPAC input mode: tcp (listen on socket) or file (tail CSV).",
    )
    parser.add_argument(
        "--biopac-file",
        default=None,
        help="Path to BIOPAC regressors CSV when using --biopac-mode=file.",
    )
    parser.add_argument(
        "--biopac-poll",
        type=float,
        default=REGRESSOR_SETTINGS.biopac_poll_interval,
        help="Polling interval (seconds) for file-backed BIOPAC buffer.",
    )
    parser.add_argument(
        "--biopac-handshake",
        action="store_true",
        default=REGRESSOR_SETTINGS.biopac_handshake,
        help="Send a handshake with TR to the BIOPAC streamer.",
    )
    parser.add_argument(
        "--biopac-start-online",
        action="store_true",
        default=REGRESSOR_SETTINGS.biopac_start_online_only,
        help="Defer BIOPAC receiver start until after offline DICOM processing.",
    )
    parser.add_argument(
        "--biopac-timelag",
        action="store_true",
        default=REGRESSOR_SETTINGS.biopac_timelag,
        help="Log per-volume trigger-to-volume timelag and running average.",
    )
    parser.add_argument(
        "--analysis-space",
        choices=["epi", "t1", "mni"],
        default=REGRESSOR_SETTINGS.analysis_space,
        help="Space for scoring/output volumes: epi (native EPI), t1 (apply EPI->T1), or mni (apply EPI->T1->MNI).",
    )
    parser.add_argument(
        "--voxel-norm-ref-volumes",
        type=int,
        default=REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
        help="When motion regression is disabled, number of initial volumes used to estimate voxel-wise reference mean.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=REGRESSOR_SETTINGS.max_workers,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--rt-max-scan-length",
        type=int,
        default=REGRESSOR_SETTINGS.rt_max_scan_length,
        help="Maximum TR count preallocated by RTPSpy regression.",
    )
    parser.add_argument(
        "--pipeline-engine",
        choices=["parallel_ordered", "legacy"],
        default=REGRESSOR_SETTINGS.pipeline_engine,
        help="Pipeline execution engine.",
    )
    parser.add_argument(
        "--commit-wait-timeout-s",
        type=float,
        default=REGRESSOR_SETTINGS.commit_wait_timeout_s,
        help="Seconds to wait before force-advancing ordered commit for stalled scans.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=REGRESSOR_SETTINGS.max_retries,
        help="Maximum retries per DICOM if processing fails.",
    )
    parser.add_argument(
        "--volume-streamer",
        action="store_true",
        default=None,
        help="Open a small best-effort local QC window showing selected live NIfTI volumes.",
    )
    parser.add_argument(
        "--volume-stream-kind",
        choices=["raw", "mc", "unwarped", "reg", "score_input"],
        default=None,
        help="Which per-volume NIfTI to show in the QC window. Defaults to settings-file/global config.",
    )
    parser.add_argument(
        "--volume-stream-every-n",
        type=int,
        default=None,
        help="Render only every Nth volume in the QC window to reduce system load. Defaults to settings-file/global config.",
    )

    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    args = parser.parse_args()

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))

    REGRESSOR_SETTINGS.enable_biopac_physio = args.biopac_enable
    REGRESSOR_SETTINGS.biopac_host = args.biopac_host
    REGRESSOR_SETTINGS.biopac_port = args.biopac_port
    REGRESSOR_SETTINGS.biopac_timeout = args.biopac_timeout
    REGRESSOR_SETTINGS.biopac_phys_reg = args.biopac_phys_reg
    REGRESSOR_SETTINGS.biopac_handshake = args.biopac_handshake
    REGRESSOR_SETTINGS.biopac_start_online_only = args.biopac_start_online
    REGRESSOR_SETTINGS.biopac_mode = args.biopac_mode
    REGRESSOR_SETTINGS.biopac_file = Path(args.biopac_file) if args.biopac_file else None
    REGRESSOR_SETTINGS.biopac_poll_interval = args.biopac_poll
    REGRESSOR_SETTINGS.biopac_timelag = args.biopac_timelag
    REGRESSOR_SETTINGS.analysis_space = args.analysis_space
    REGRESSOR_SETTINGS.voxel_norm_ref_volumes = max(1, int(args.voxel_norm_ref_volumes))
    REGRESSOR_SETTINGS.max_workers = args.max_workers
    REGRESSOR_SETTINGS.rt_max_scan_length = max(1, int(args.rt_max_scan_length))
    REGRESSOR_SETTINGS.pipeline_engine = args.pipeline_engine
    REGRESSOR_SETTINGS.commit_wait_timeout_s = max(0.1, float(args.commit_wait_timeout_s))
    REGRESSOR_SETTINGS.max_retries = args.max_retries
    if args.volume_streamer is not None:
        REGRESSOR_SETTINGS.enable_volume_streamer = args.volume_streamer
    if args.volume_stream_kind is not None:
        REGRESSOR_SETTINGS.volume_stream_kind = args.volume_stream_kind
    if args.volume_stream_every_n is not None:
        REGRESSOR_SETTINGS.volume_stream_every_n = max(1, int(args.volume_stream_every_n))

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
        decoder_roi_txt=Path(args.decoder_roi_txt) if args.decoder_roi_txt else None,
        reference_score_run=args.reference_score_run,
        enable_scoring=not args.no_score,
        enable_original_score=args.enable_original_score,
        ap_block=args.ap_block,
        pa_block=args.pa_block,
    )

    if not cfg.incoming_dir.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {cfg.incoming_dir}")

    run_rt_pipeline(cfg)


if __name__ == "__main__":
    main()
