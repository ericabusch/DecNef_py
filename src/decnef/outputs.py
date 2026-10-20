"""Run output files: the per-run CSVs and session_metadata.json.

Columns are unchanged from the original pipeline (docs/architecture.md lists
them); the writers are the original functions with explicit paths.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def _append(path: Path, header: list, row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def append_score(csv_path: Path, volume_idx: int, raw_score: float, original_score: Optional[float] = None,
                 reg_ready: Optional[bool] = None, timestamp: Optional[float] = None, event_type: Optional[str] = None) -> float:
    if timestamp is None:
        timestamp = time.time()
    _append(csv_path, ["volume_idx", "timestamp", "score_raw", "score_original", "reg_ready", "event_type"],
            [volume_idx, timestamp, raw_score, original_score, int(reg_ready) if reg_ready is not None else "", event_type or ""])
    return timestamp


def append_score_z(csv_path: Path, volume_idx: int, timestamp: float, raw_score: float, z_score: float, ref_stats: dict,
                   reg_ready: Optional[bool] = None) -> None:
    _append(csv_path, ["volume_idx", "timestamp", "score_raw", "score_z", "ref_run", "ref_mean", "ref_std", "ref_n",
                       "ref_used_reg_ready", "reg_ready"],
            [volume_idx, timestamp, raw_score, z_score, ref_stats["run"], ref_stats["mean"], ref_stats["std"], ref_stats["n"],
             int(ref_stats.get("used_reg_ready", False)), int(reg_ready) if reg_ready is not None else ""])


def append_motion(motion_path: Path, motion_vec: np.ndarray) -> None:
    """One 6-parameter row per volume, AFNI 1D style (no header)."""
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(motion_path, "a") as f:
        f.write(" ".join(f"{x:.6f}" for x in motion_vec) + "\n")


def append_fd(fd_path: Path, volume_idx: int, fd_value: float) -> None:
    _append(fd_path, ["volume_idx", "fd"], [volume_idx, fd_value])


def append_dvars(qc_path: Path, volume_idx: int, fd: float, dvars: float, dvars_z: float) -> None:
    _append(qc_path, ["volume_idx", "fd", "dvars", "robust_z"], [volume_idx, fd, dvars, dvars_z])


def append_regressors(reg_path: Path, volume_idx: int, reg_names: list, reg_row: np.ndarray) -> None:
    _append(reg_path, ["volume_idx", *reg_names], [volume_idx, *reg_row.tolist()])


def append_regression_status(qc_path: Path, volume_idx: int, fd: float, dvars: float, dvars_z: float, fd_censor: int,
                             dvars_censor: int, reg_ready: bool, biopac_missing: bool) -> None:
    _append(qc_path, ["volume_idx", "fd", "dvars", "dvars_z", "fd_censor", "dvars_censor", "reg_ready", "biopac_missing"],
            [volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, int(reg_ready), int(biopac_missing)])


def append_biopac_timelag(path: Path, volume_idx: int, trigger_timestamp: float, volume_timestamp: float,
                          timelag_s: float, avg_timelag_s: float) -> None:
    _append(path, ["volume_idx", "trigger_timestamp", "volume_timestamp", "timelag_s", "avg_timelag_s"],
            [volume_idx, f"{trigger_timestamp:.6f}", f"{volume_timestamp:.6f}", f"{timelag_s:.6f}", f"{avg_timelag_s:.6f}"])


# ----------------------------------------------------------------------------- session_metadata.json
def read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_metadata(path: Path, payload: dict) -> None:
    """Shallow-merge `payload` into the file, written atomically."""
    data = read_metadata(path)
    data.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def session_metadata_payload(settings, layout, decoder_template: Optional[Path], reference_score_stats: Optional[dict],
                             preprocessing_order: str) -> dict:
    """The pipeline's own keys (same names as before; the full settings tree is added under 'settings')."""
    p, ph = settings.pipeline, settings.physio
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject": layout.sub,
        "day": layout.day,
        "run": layout.run,
        "incoming_root": str(layout.incoming_root),
        "base_data": str(layout.base_data),
        "decoder_template": str(decoder_template) if decoder_template else None,
        "decoder_roi_txt": str(settings.decoder.roi_txt) if settings.decoder.roi_txt else None,
        "reference_score_run": settings.decoder.reference_score_run,
        "reference_score_stats": reference_score_stats,
        "enable_original_score": settings.decoder.enable_original_score,
        "preprocessing_order": preprocessing_order,
        "smoothing_fwhm_mm": float(p.smoothing_fwhm_mm),
        "analysis_mask": p.mask.analysis_mask,
        "analysis_mask_file": str(p.mask.file) if p.mask.file else None,
        "analysis_mask_space": p.mask.space,
        "fieldmap_dir": str(layout.fmap_dir),
        "motion_reference": str(layout.motion_ref_epi),
        "tr": p.TR,
        "regression": {
            "enable_motion_regression": p.motion.enable_regression,
            "mot_reg": p.motion.mot_reg,
            "max_poly_order": p.motion.max_poly_order,
            "use_gs": p.nuisance.use_gs,
            "use_wm": p.nuisance.use_wm,
            "use_vent": p.nuisance.use_vent,
            "enable_fd_censor_reg": p.censor.fd.enable,
            "fd_thr": p.censor.fd.thr,
            "enable_dvars_censor_reg": p.censor.dvars.enable,
            "dvars_thr_robust_z": p.censor.dvars.thr_robust_z,
            "censor_plus1": p.censor.plus1,
            "dvars_warmup": p.censor.dvars.warmup,
            "dvars_mask_source": p.censor.dvars.mask_source,
            "analysis_space": p.analysis_space,
            "voxel_norm_ref_volumes": p.voxel_norm_ref_volumes,
            "rt_max_scan_length": p.engine.rt_max_scan_length,
            "commit_wait_timeout_s": p.engine.commit_wait_timeout_s,
        },
        "biopac": {
            "enabled": ph.enable, "phys_reg": ph.phys_reg, "mode": ph.mode,
            "file": str(ph.file) if ph.file else None, "host": ph.host, "port": ph.port, "timeout": ph.timeout,
            "handshake": ph.handshake, "start_online_only": ph.start_online_only, "poll_interval": ph.poll_interval,
        },
        "settings": settings.to_dict(),
    }


# ----------------------------------------------------------------------------- z-score reference
def load_reference_score_stats(ref_run_dir: Path, run_id: str, log=None) -> Optional[dict]:
    """Mean/std of a previous run's reg_ready scores (the first reg_ready sample is
    dropped: RTP regression can still show a transient there). Same rules as before."""
    scores_path = ref_run_dir / "scores.csv"
    if not scores_path.exists():
        if log:
            log.warning("reference scores.csv not found at %s", scores_path)
        return None

    reg_ready_map = None
    reg_status_path = ref_run_dir / "regression_status_rt.csv"
    if reg_status_path.exists():
        reg_ready_map = {}
        with open(reg_status_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "volume_idx" in reader.fieldnames and "reg_ready" in reader.fieldnames:
                for row in reader:
                    try:
                        reg_ready_map[int(row["volume_idx"])] = bool(int(row["reg_ready"]))
                    except (TypeError, ValueError):
                        continue
            else:
                reg_ready_map = None

    scores, scores_all, reg_ready_scores = [], [], []
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
                reg_ready_scores.append((vol, raw))

    used_reg_ready = (reg_ready_map is not None) or has_score_reg_ready
    skipped_first_reg_ready = False
    if reg_ready_scores:
        scores = [raw for _, raw in reg_ready_scores[1:]]
        skipped_first_reg_ready = True
    if used_reg_ready and not scores:
        scores = scores_all
        used_reg_ready = False
    if len(scores) < 2:
        return None
    values = np.asarray(scores, dtype=float)
    mean, std = float(np.nanmean(values)), float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        return None
    return {"run": str(run_id), "mean": mean, "std": std, "n": int(len(scores)),
            "used_reg_ready": used_reg_ready, "skipped_first_reg_ready": skipped_first_reg_ready}
