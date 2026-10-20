"""The real-time pipeline: build the run state, watch the incoming folder,
process every volume in order, publish score events.

    run_pipeline(settings, layout, score_queue=None, stop_event=None)

Runs until `stop_event` is set (or Ctrl-C). Volumes of this run that are
already in the incoming folder are processed first ("offline backlog"), then
the watcher goes online. Every volume's outcome is reported on the score
queue: kind "score" (scoring on), "tick" (scoring off) or "error".
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from watchdog.observers import Observer

from . import outputs
from .calibration import PIPELINE_ORDER, fieldmap_path, fieldmap_ready, load_calibration, native_paths, source_identity
from .config import PHYS_REG_COUNTS, Settings
from .decoder import DecoderScorer
from .engine import OrderedEngine, ResultEnvelope
from .events import StageSchedule
from .ingest import IncomingWatcher, convert_dicom, list_run_dicoms
from .layout import RunLayout
from .masks import final_output_kind, mask_provenance
from .physio import FileBuffer, Receiver, ReceiverConfig
from .resampler import init_gpu_resampler
from .rtpspy.rtp_volreg import RtpVolreg
from .smoothing import prepare_smoother
from .stages import (MotionRegressor, RunState, VoxelNormalizer, censor_stage, motion_stage, regression_stage,
                     score_stage, smooth_stage, status_stage, transform_stage, unwarp_stage)
from .tools import require, run
from .viewer import ViewerConfig, ViewerHandle

log = logging.getLogger(__name__)
POLL_INTERVAL_S = 0.2


class PipelineError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- prerequisites
def check_prerequisites(settings: Settings, layout: RunLayout) -> Path:
    """Fail at start, not at volume 1. Returns the decoder template path (None when scoring is off)."""
    p = settings.pipeline
    tools = ["dcm2niix"]
    if p.analysis_space != "epi" and not p.resampler.use_gpu:
        tools.append("antsApplyTransforms")
    require(*tools)
    if p.fieldmap.method != "pyhysco":
        raise PipelineError("unwarp -> MC supports fieldmap.method=pyhysco only")
    if layout.incoming_root is None or not Path(layout.incoming_root).is_dir():
        raise PipelineError(f"incoming folder does not exist: {layout.incoming_root}")
    if not layout.motion_ref_epi.exists():
        raise PipelineError(f"motion/analysis reference not found at {layout.motion_ref_epi}; run the offline preparation first")
    if not fieldmap_ready(layout.fmap_dir):
        raise PipelineError(f"fieldmap calibration missing or incomplete in {layout.fmap_dir}/{native_paths(layout.fmap_dir)[0].name}; "
                            "build it with the offline preparation")
    load_calibration(layout.fmap_dir)   # raises when stale
    decoder = None
    if settings.decoder.enable_scoring:
        decoder = settings.decoder.template
        if decoder is None and p.analysis_space != "epi":
            raise PipelineError("decoder.template is required for t1/mni analysis spaces (it is the resampling reference)")
        if decoder is not None and not Path(decoder).exists():
            raise PipelineError(f"decoder template not found: {decoder}")
    elif p.analysis_space != "epi":
        if settings.decoder.template is None or not Path(settings.decoder.template).exists():
            raise PipelineError("decoder.template (the resampling reference grid) is required for t1/mni spaces even without scoring")
        decoder = settings.decoder.template
    return Path(decoder) if decoder else None


def validate_run_provenance(settings: Settings, layout: RunLayout, decoder_template: Optional[Path]) -> None:
    """Refuse to mix outputs made under different settings / calibration / reference in one run folder."""
    p = settings.pipeline
    marker = layout.provenance_json
    payload = {"order": PIPELINE_ORDER, "calibration": load_calibration(layout.fmap_dir),
               "field": source_identity(fieldmap_path(layout.fmap_dir)),
               "reference": source_identity(layout.motion_ref_epi),
               "epi_phase_encoding": p.fieldmap.epi_phase_encoding}
    if final_output_kind(p.smoothing_fwhm_mm, p.mask.analysis_mask) is not None:
        reference = layout.motion_ref_epi if p.analysis_space == "epi" else decoder_template
        payload["final_masking"] = {"fwhm_mm": p.smoothing_fwhm_mm, "space": p.analysis_space,
                                    **mask_provenance(layout, p.mask.analysis_mask, p.analysis_space, reference,
                                                      p.mask.file, p.mask.space)}
    if marker.exists():
        if json.loads(marker.read_text()) != payload:
            raise PipelineError("run preprocessing/calibration/reference changed; use a fresh run output directory")
    elif any((layout.run_dir / name).exists() and any((layout.run_dir / name).glob("*.nii*"))
             for name in ("mc", "unwarped", "reg", "t1", "mni", "smooth", "masked")):
        raise PipelineError("run directory already holds processed volumes; use a fresh run id")
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2))


# ----------------------------------------------------------------------------- state
def build_state(settings: Settings, layout: RunLayout, decoder_template: Optional[Path], start_physio: bool = True) -> RunState:
    """Everything that is loaded once per run (ex DICOMHandler.__init__)."""
    p = settings.pipeline
    layout.create_run_dirs()
    ref_img = nib.load(str(layout.motion_ref_epi))
    ref_mask = np.asanyarray(nib.load(str(layout.motion_ref_mask)).dataobj) > 0.5 if layout.motion_ref_mask.exists() else None

    # z-score reference run
    reference_score_stats = None
    if settings.decoder.reference_score_run:
        ref = layout.other_run(settings.decoder.reference_score_run)
        reference_score_stats = outputs.load_reference_score_stats(ref.run_dir, settings.decoder.reference_score_run, log=log)
        if reference_score_stats is None:
            raise PipelineError(f"reference run {settings.decoder.reference_score_run}: no usable scores.csv for z-scoring")
        log.info("[SCORE] reference run %s (mean=%.4f, std=%.4f, n=%s)", ref.run, reference_score_stats["mean"],
                 reference_score_stats["std"], reference_score_stats["n"])

    # unwarp
    applier = None
    field = fieldmap_path(layout.fmap_dir)
    if p.fieldmap.preloaded:
        from .unwarp import PreloadedApplier

        applier = PreloadedApplier(prototype_vol_path=native_paths(layout.fmap_dir)[0] / "AP_mean.nii", fieldmap_path=field,
                                   phase_encoding_direction=load_calibration(layout.fmap_dir)["phase_encoding_axis"],
                                   polarity=1 if p.fieldmap.epi_phase_encoding == "AP" else -1, device=p.fieldmap.device,
                                   dtype=torch.float32, interpolation_backend=p.fieldmap.backend)

    # motion correction
    volreg = RtpVolreg(regmode=p.motion.volreg_mode, max_scan_length=p.engine.rt_max_scan_length)
    volreg.ignore_init = 0
    volreg.save_proc = False
    volreg.set_ref_vol(str(layout.motion_ref_epi))

    # DVARS mask
    dvars_mask = None
    if p.censor.dvars.enable:
        if ref_mask is None:
            raise PipelineError(f"DVARS censoring needs the reference mask at {layout.motion_ref_mask}")
        dvars_mask = ref_mask

    # physio (in-process receiver or file buffer written by the listener process)
    physio = None
    rtp_physio = None
    phys_reg = "None"
    ph = settings.physio
    if ph.enable:
        if not p.motion.enable_regression:
            raise PipelineError("physio regressors need pipeline.motion.enable_regression=true")
        n_reg = PHYS_REG_COUNTS[ph.phys_reg]
        if ph.mode == "file":
            if ph.file is None:
                raise PipelineError("physio.mode=file requires physio.file")
            rtp_physio = FileBuffer(ph.file, timeout=ph.timeout, expected_regressors=n_reg, poll_interval=ph.poll_interval)
        else:
            physio = Receiver(ReceiverConfig(host=ph.host, port=ph.port, timeout=ph.timeout if start_physio else 0.0,
                                             expected_regressors=n_reg, handshake_tr=p.TR if ph.handshake else None,
                                             subject=layout.sub, day=layout.day, run=layout.run,
                                             output_path=layout.biopac_regressors_csv))
            if start_physio:
                physio.start()
            rtp_physio = physio
        phys_reg = ph.phys_reg

    # regression
    motion_regressor = voxel_normalizer = None
    if p.motion.enable_regression:
        def mask_or_none(path, enabled):
            return path if (enabled and path.exists()) else None

        motion_regressor = MotionRegressor(
            volreg, reg_mask=layout.motion_ref_mask if layout.motion_ref_mask.exists() else None,
            gs_mask=mask_or_none(layout.motion_ref_mask, p.nuisance.use_gs), wm_mask=mask_or_none(layout.wm_mask, p.nuisance.use_wm),
            vent_mask=mask_or_none(layout.vent_mask, p.nuisance.use_vent), mot_reg=p.motion.mot_reg,
            max_poly_order=p.motion.max_poly_order, TR=p.TR, max_scan_length=p.engine.rt_max_scan_length,
            norm_ref_volumes=p.voxel_norm_ref_volumes, phys_reg=phys_reg, rtp_physio=rtp_physio)
        for name, enabled, path in (("WM", p.nuisance.use_wm, layout.wm_mask), ("Vent", p.nuisance.use_vent, layout.vent_mask)):
            if enabled and not path.exists():
                log.warning("[REG] %s mask missing at %s; regressor skipped", name, path)
    else:
        voxel_normalizer = VoxelNormalizer(ref_volumes=p.voxel_norm_ref_volumes, brain_mask=ref_mask)

    # transform + final stage + scoring
    gpu_resampler = init_gpu_resampler(settings, layout, decoder_template) if decoder_template else None
    final_reference = layout.motion_ref_epi if p.analysis_space == "epi" else decoder_template
    smoother = prepare_smoother(layout, p.smoothing_fwhm_mm, p.analysis_space, final_reference, run,
                                mask_type=p.mask.analysis_mask, custom_file=p.mask.file, custom_space=p.mask.space)
    scorer = None
    if settings.decoder.enable_scoring:
        scorer = DecoderScorer(decoder_template, roi_txt=settings.decoder.roi_txt)
        log.info("[SCORE] decoder: %s", scorer.source)

    state = RunState(settings=settings, layout=layout, decoder_template=decoder_template, ref_affine=ref_img.affine,
                     volreg=volreg, applier=applier, fieldmap_path=field, motion_regressor=motion_regressor,
                     voxel_normalizer=voxel_normalizer, gpu_resampler=gpu_resampler, smoother=smoother, scorer=scorer,
                     score_tracker=StageSchedule.from_config(settings.stages, p.skip_first_trs), reference_score_stats=reference_score_stats,
                     physio=physio, dvars_mask=dvars_mask)
    if motion_regressor is not None:
        motion_regressor._regress.mask_src_proc = state.proc_src
    return state


# ----------------------------------------------------------------------------- per volume
def process_volume(state: RunState, raw_nii: Path, volume_idx: int, volume_timestamp: float) -> dict:
    """All ordered stages for one volume. Returns the queue message (kind/source added by the caller)."""
    img = unwarp_stage(state, raw_nii, volume_idx)
    mc_img, mc_nii, fd_value, fd_to_save = motion_stage(state, img, volume_idx)
    fd_censor, dvars_censor = censor_stage(state, np.asanyarray(mc_img.dataobj), volume_idx, fd_value, fd_to_save)
    reg_nii, reg_ready = regression_stage(state, mc_img, volume_idx, fd_censor, dvars_censor)
    status_stage(state, volume_idx, fd_to_save, fd_censor, dvars_censor, reg_ready, volume_timestamp)
    score_input, score_input_orig = transform_stage(state, reg_nii, mc_nii, volume_idx)
    score_input, score_input_orig = smooth_stage(state, score_input, score_input_orig, volume_idx)
    if state.scorer is None:
        return {"volume_idx": volume_idx, "watchdog_timestamp": volume_timestamp, "reg_ready": reg_ready,
                "score_input": str(score_input), "kind": "tick"}
    message = score_stage(state, score_input, score_input_orig, volume_idx, reg_ready)
    message["watchdog_timestamp"] = volume_timestamp
    message["kind"] = "score"
    return message


def stream_path(state: RunState, kind: str, raw_nii: Path, volume_idx: int, score_input: Optional[Path]) -> Optional[Path]:
    lay = state.layout
    return {"raw": raw_nii, "mc": lay.mc_nii(volume_idx), "reg": lay.reg_nii(volume_idx), "score_input": score_input}.get(kind)


# ----------------------------------------------------------------------------- the run
def run_pipeline(settings: Settings, layout: RunLayout, score_queue=None, stop_event=None) -> dict:
    """Run until stop_event is set or Ctrl-C. Returns counts {'committed', 'failed'}."""
    decoder_template = check_prerequisites(settings, layout)
    validate_run_provenance(settings, layout, decoder_template)
    existing = list_run_dicoms(layout.incoming_root, int(layout.run))
    defer_physio = bool(existing) or settings.physio.start_online_only
    state = build_state(settings, layout, decoder_template, start_physio=not defer_physio)
    outputs.write_metadata(layout.metadata_json,
                           outputs.session_metadata_payload(settings, layout, decoder_template, state.reference_score_stats, PIPELINE_ORDER))
    if settings.stages is not None:
        outputs.write_metadata(layout.metadata_json, {"stages": state.score_tracker.to_metadata()})
    viewer = ViewerHandle(ViewerConfig.from_settings(settings.qc))
    viewer.start()

    def put(message: dict) -> None:
        if score_queue is None:
            return
        message.setdefault("source", "pipeline")
        try:
            score_queue.put_nowait(message)
        except Exception as exc:
            log.error("[QUEUE] dropped %s for vol %s: %s", message.get("kind"), message.get("volume_idx"), exc)

    def compute(path: Path, scan: int, volume_idx: int) -> ResultEnvelope:
        ts = time.time()
        raw_nii = convert_dicom(path, layout.raw_dir, volume_idx)
        return ResultEnvelope(scan=scan, volume_idx=volume_idx, dicom_path=path, volume_timestamp=ts, success=True, raw_nii=raw_nii)

    def commit(env: ResultEnvelope) -> None:
        if not env.success:
            log.error("[COMMIT] scan %s failed in compute: %s", env.scan, env.error)
            put({"kind": "error", "volume_idx": env.volume_idx, "timestamp": time.time(), "error": env.error})
            return
        try:
            message = process_volume(state, env.raw_nii, env.volume_idx, env.volume_timestamp)
        except Exception as exc:
            log.exception("[COMMIT] volume %05d failed", env.volume_idx)
            put({"kind": "error", "volume_idx": env.volume_idx, "timestamp": time.time(), "error": f"{type(exc).__name__}: {exc}"})
            env.success = False
            return
        viewer.publish(env.volume_idx, stream_path(state, settings.qc.stream_kind, env.raw_nii, env.volume_idx,
                                                   Path(message["score_input"]) if "score_input" in message else None))
        message.pop("score_input", None)
        put(message)

    def on_first_online_volume() -> None:
        if state.physio is None:
            return
        if defer_physio:
            state.physio.config.timeout = settings.physio.timeout
            state.physio.start()
        state.physio.send_run_start()

    engine = OrderedEngine(compute, commit, max_workers=settings.pipeline.engine.max_workers,
                           commit_wait_timeout_s=settings.pipeline.engine.commit_wait_timeout_s,
                           max_retries=settings.pipeline.engine.max_retries)
    watcher = IncomingWatcher(int(layout.run), on_first_online_volume=on_first_online_volume)
    observer = Observer()
    observer.schedule(watcher, str(layout.incoming_root), recursive=False)
    observer.start()

    def pump() -> bool:
        moved = False
        while True:
            item = watcher.next_pending()
            if item is None:
                break
            engine.note_pending(watcher.pending_scans)
            engine.submit(item[1], item[0])
            moved = True
        engine.note_pending(watcher.pending_scans)
        engine.drain()
        return moved

    try:
        if existing:
            log.info("[RT] %d volumes of run %s already in %s; processing them first", len(existing), layout.run, layout.incoming_root)
            for path in existing:
                watcher.enqueue(path)
            pump()
        watcher.set_online()
        log.info("[RT] online: watching %s for run %s", layout.incoming_root, layout.run)
        while stop_event is None or not stop_event.is_set():
            if not pump():
                time.sleep(POLL_INTERVAL_S)
        log.info("[RT] stop requested")
    except KeyboardInterrupt:
        log.info("[RT] interrupted")
    finally:
        observer.stop()
        engine.shutdown(wait=False)
        if state.physio is not None:
            state.physio.send_run_end()
            state.physio.stop()
        viewer.stop()
        observer.join(timeout=5)
    return {"committed": engine.committed, "failed": engine.failed}
