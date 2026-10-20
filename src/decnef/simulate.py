"""Simulation: run a launcher end to end without a scanner.

Two levels, selected with `start_session(cfg, simulate=SimulationConfig(level=...))`:

- ``scores``: a fake pipeline process emits one ScoreEvent per TR with the
  correct stage labels and reg_ready behaviour, and writes
  session_metadata.json and scores.csv. No external tools, no phantom.
- ``scanner``: synthetic Siemens-mosaic DICOMs are written into an incoming
  folder at the TR (or a recorded run's DICOMs are replayed) and fed through
  the full real pipeline, so the display receives real events and the run
  folder holds every real output file. Needs ``dcm2niix`` on PATH (and ANTs
  only for t1/mni spaces without the GPU resampler).

Outputs never go into real subject data: everything lives under
``SimulationConfig.out_dir`` (a fresh temp dir by default). A reused out_dir
keeps its phantom; only the run folder is regenerated. See docs/session_api.md
"Simulation".
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import nibabel as nib
import numpy as np

from .config import PHYS_REG_COUNTS, Settings
from .events import StageSchedule
from .ingest import parse_dicom_name
from .layout import RunLayout
from .tools import which

log = logging.getLogger(__name__)
LEVELS = ("scores", "scanner")


class SimulationError(RuntimeError):
    pass


@dataclass
class SimulationConfig:
    level: str = "scores"                     # "scores" | "scanner"
    n_volumes: int = 60
    tr: Optional[float] = None                # seconds between volumes; default settings.pipeline.TR
    out_dir: Optional[Path] = None            # None: fresh temp dir; an existing dir is reused (phantom kept)
    sub: str = "sim"                          # phantom subject / day under out_dir/data
    day: str = "1"
    # phantom and synthetic scanner output (scanner level)
    grid: tuple = (32, 32, 20)
    voxel_mm: float = 3.0
    motion_amp_mm: float = 0.5                # slow rigid jitter amplitude; a 4x spike every 15th volume feeds the censoring
    drift: float = 0.001                      # fractional intensity drift per volume
    noise_sd: float = 5.0                     # additive Gaussian noise (phantom intensities ~1000)
    signal_amp: float = 0.02                  # fractional signal injected in the decoder's positive voxels when "on"
    signal_stages: tuple = ("scans",)         # stage labels during which the signal is on; no stages: 10-on/10-off blocks
    seed: int = 0
    from_dicoms: Optional[Path] = None        # scanner level from a recorded run instead of the phantom (= replay)
    source_run: Optional[int] = None          # run field of the recorded DICOM names
    # scores level
    reg_ready_from: Optional[int] = None      # first volume with reg_ready=True; default: RtpRegress's warm-up rule
    baseline_score: float = 0.0
    score_noise_sd: float = 1.0

    def __post_init__(self):
        if self.level not in LEVELS:
            raise SimulationError(f"level must be one of {LEVELS}, got {self.level!r}")
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
        if self.from_dicoms is not None:
            self.from_dicoms = Path(self.from_dicoms)
            if self.source_run is None:
                raise SimulationError("from_dicoms needs source_run (the run field of the recorded DICOM names)")

    def phantom_params(self) -> dict:
        return {"grid": list(self.grid), "voxel_mm": self.voxel_mm, "seed": self.seed}

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("out_dir", "from_dicoms"):
            if d[k] is not None:
                d[k] = str(d[k])
        return d


@dataclass
class SimPaths:
    out_dir: Path
    base_data: Path
    incoming: Path
    layout: RunLayout
    decoder: Optional[Path]
    phantom_manifest: Path

    @property
    def run_dir(self) -> Path:
        return self.layout.run_dir


# ----------------------------------------------------------------------------- paths
def resolve_sim_paths(settings: Settings, run: str, sim: SimulationConfig, ap_block=None, pa_block=None) -> SimPaths:
    """Where a simulation lives. A fresh temp dir unless `sim.out_dir` is given (then reused)."""
    out_dir = sim.out_dir if sim.out_dir is not None else Path(tempfile.mkdtemp(prefix="decnef_sim_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    base_data = out_dir / "data"
    incoming = out_dir / "incoming"
    layout = RunLayout(base_data, sim.sub, sim.day, str(run), incoming_root=incoming, ap_block=ap_block, pa_block=pa_block)
    return SimPaths(out_dir=out_dir, base_data=base_data, incoming=incoming, layout=layout, decoder=out_dir / "decoder.nii",
                    phantom_manifest=out_dir / "phantom.json")


def simulation_settings(settings: Settings, paths: SimPaths, sim: SimulationConfig) -> Settings:
    """A copy of the settings pointing into the simulation directory."""
    s = settings.copy()
    s.paths.base_data = paths.base_data
    s.paths.incoming_root = paths.incoming
    if sim.level == "scanner" and sim.from_dicoms is None:
        s.decoder.template = paths.decoder
        s.decoder.roi_txt = None
    if sim.tr is not None:
        s.pipeline.TR = float(sim.tr)
    return s


def reset_run_dir(paths: SimPaths) -> None:
    """Simulation output is disposable: a reused out_dir gets a fresh run folder (the phantom is kept)."""
    if paths.run_dir.exists():
        shutil.rmtree(paths.run_dir)
    paths.incoming.mkdir(parents=True, exist_ok=True)
    for stale in paths.incoming.glob(f"*_{int(paths.layout.run):06d}_*.dcm"):
        stale.unlink()


# ----------------------------------------------------------------------------- stage gating
def signal_schedule(settings: Settings, sim: SimulationConfig) -> Callable[[int], bool]:
    """volume_idx -> is the injected signal on? Stage-gated when settings.stages is set, else 10-on/10-off blocks
    after skip_first_trs + voxel_norm_ref_volumes."""
    sched = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)
    if sched is not None and sched.cycle:
        on = set(sim.signal_stages)
        return lambda v: sched.label_for_volume(v) in on
    start = settings.pipeline.skip_first_trs + settings.pipeline.voxel_norm_ref_volumes
    return lambda v: v > start and ((v - start - 1) // 10) % 2 == 1


def reg_ready_from(settings: Settings, sim: SimulationConfig) -> int:
    """RtpRegress waits max(voxel_norm_ref_volumes - 1, n_regressors + 1) volumes before regressing
    (rtpspy/rtp_regress.py set_wait_num); the first reg_ready volume is one after that."""
    if sim.reg_ready_from is not None:
        return int(sim.reg_ready_from)
    p = settings.pipeline
    if not p.motion.enable_regression:
        return 1
    n_reg = 6 + int(p.nuisance.use_wm) + int(p.nuisance.use_vent) + int(p.nuisance.use_gs)
    if settings.physio.enable:
        n_reg += PHYS_REG_COUNTS[settings.physio.phys_reg]
    n_reg += 1   # polynomial baseline term
    return max(p.voxel_norm_ref_volumes - 1, n_reg + 1) + 1


# ----------------------------------------------------------------------------- level 1: scores
def scores_pipeline_main(settings_dict: dict, layout: RunLayout, sim_dict: dict, score_queue, stop_event) -> None:
    """Fake pipeline process: one ScoreEvent per TR with stage labels and reg_ready, scores.csv and
    session_metadata.json as the real pipeline writes them; exits after n_volumes (-> 'end')."""
    from . import outputs
    from .config import from_dict

    settings = from_dict(settings_dict)
    sim = SimulationConfig(**{k: v for k, v in sim_dict.items() if k in SimulationConfig.__dataclass_fields__})
    layout.create_run_dirs()
    sched = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)
    on = signal_schedule(settings, sim)
    ready_from = reg_ready_from(settings, sim)
    tr = float(settings.pipeline.TR)
    outputs.write_metadata(layout.metadata_json, outputs.session_metadata_payload(settings, layout, settings.decoder.template, None, "simulated_scores"))
    outputs.write_metadata(layout.metadata_json, {"simulation": {"level": "scores", **sim.to_dict()},
                                                  **({"stages": sched.to_metadata()} if sched else {})})
    rng = np.random.default_rng(sim.seed)
    t_next = time.time()
    for v in range(1, sim.n_volumes + 1):
        if stop_event.is_set():
            return
        now = time.time()
        if t_next > now:
            time.sleep(min(t_next - now, 0.05))
            continue_sleep = t_next - time.time()
            while continue_sleep > 0 and not stop_event.is_set():
                time.sleep(min(continue_sleep, 0.05))
                continue_sleep = t_next - time.time()
        watchdog_ts = time.time()
        reg_ready = v >= ready_from
        stage = sched.label_for_volume(v) if sched else None
        raw = sim.baseline_score + (sim.signal_amp * 100.0 if on(v) else 0.0) + rng.normal(0, sim.score_noise_sd)
        analysis_ts = time.time()
        ts = outputs.append_score(layout.scores_csv, v, raw, None, reg_ready, analysis_ts, stage)
        outputs.append_regression_status(layout.regression_status_csv, v, 0.0, 0.0, float("nan"), 0, 0, reg_ready, False)
        if score_queue is not None:
            try:
                score_queue.put_nowait({"kind": "score", "source": "pipeline", "volume_idx": v, "timestamp": ts,
                                        "analysis_timestamp": analysis_ts, "watchdog_timestamp": watchdog_ts,
                                        "score_raw": raw, "score_original": None, "reg_ready": reg_ready, "stage": stage,
                                        "simulated": True})
            except Exception as exc:
                log.error("[SIM] dropped score for volume %s: %s", v, exc)
        t_next += tr


# ----------------------------------------------------------------------------- level 2: phantom
def _ellipsoid(grid, center, radii):
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in grid], indexing="ij")
    return ((x - center[0]) ** 2 / radii[0] ** 2 + (y - center[1]) ** 2 / radii[1] ** 2 + (z - center[2]) ** 2 / radii[2] ** 2) < 1.0


def phantom_base(sim: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """(intensity volume, brain mask) on the DICOM/NIfTI grid: a smooth ellipsoid 'brain' (~1000) with a
    brighter core, background ~20."""
    nx, ny, nz = sim.grid
    c = ((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2)
    brain = _ellipsoid(sim.grid, c, (nx * 0.42, ny * 0.42, nz * 0.45))
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in sim.grid], indexing="ij")
    core = np.exp(-(((x - c[0]) / (nx * 0.25)) ** 2 + ((y - c[1]) / (ny * 0.25)) ** 2))
    base = np.where(brain, 1000.0 + 300.0 * core, 20.0).astype(np.float32)
    return base, brain


def write_mosaic_dicom(volume_xyz: np.ndarray, voxel_mm: float, path: Path, run: int, scan: int, tr_s: float,
                       series_uid: str, study_uid: str) -> Path:
    """One Siemens-style mosaic DICOM per volume (what the scanner writes and what dcm2niix expects
    for a single-file 3D volume). `volume_xyz` is (nx, ny, nz); intensities are stored as uint16."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    nx, ny, nz = volume_xyz.shape
    tiles = int(np.ceil(np.sqrt(nz)))
    mosaic = np.zeros((tiles * ny, tiles * nx), dtype=np.uint16)
    data = np.clip(np.rint(volume_xyz), 0, 65535).astype(np.uint16)
    for k in range(nz):
        r, c = divmod(k, tiles)
        mosaic[r * ny:(r + 1) * ny, c * nx:(c + 1) * nx] = data[:, :, k].T   # DICOM rows = y, columns = x
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    if int(pydicom.__version__.split(".")[0]) < 3:       # pydicom 2.x needs these; 3.x derives them from the transfer syntax
        ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.Modality = "MR"
    ds.Manufacturer = "SIEMENS"
    ds.ManufacturerModelName = "decnef simulator"
    ds.PatientName, ds.PatientID = "SIM^PHANTOM", "sim"
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = ds.ContentDate = "20260101"
    ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = ds.ContentTime = "120000"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND", "MOSAIC"]
    ds.SeriesNumber, ds.AcquisitionNumber, ds.InstanceNumber = int(run), int(scan), int(scan)
    ds.SeriesDescription, ds.ProtocolName = "sim_bold", "sim_bold"
    ds.RepetitionTime, ds.EchoTime, ds.FlipAngle = float(tr_s * 1000.0), 30.0, 70.0
    ds.MagneticFieldStrength, ds.ScanningSequence, ds.SequenceVariant, ds.ScanOptions, ds.MRAcquisitionType = 3.0, "EP", "SK", "FS", "2D"
    ds.SliceThickness = ds.SpacingBetweenSlices = float(voxel_mm)
    ds.PixelSpacing = [float(voxel_mm), float(voxel_mm)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-(nx - 1) * voxel_mm / 2, -(ny - 1) * voxel_mm / 2, -(nz - 1) * voxel_mm / 2]
    ds.SliceLocation = ds.ImagePositionPatient[2]
    ds.PatientPosition = "HFS"
    ds.Rows, ds.Columns = mosaic.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = mosaic.tobytes()
    ds.add_new((0x0019, 0x0010), "LO", "SIEMENS MR HEADER")          # private creator for (0019,10xx)
    ds.add_new((0x0019, 0x100A), "US", int(nz))                      # NumberOfImagesInMosaic
    ds.add_new((0x0019, 0x100C), "IS", "0")                          # b-value
    try:
        ds.save_as(str(path), enforce_file_format=True)      # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)     # pydicom 2.x
    return path


def _convert_with_dcm2niix(dicom_path: Path, out_dir: Path, name: str) -> Path:
    from .tools import run

    run(["dcm2niix", "-z", "n", "-s", "y", "-a", "y", "-b", "y", "-v", "0", "-f", name, "-o", str(out_dir), str(dicom_path)])
    produced = sorted(out_dir.glob(f"{name}*.nii*"))
    if not produced:
        raise SimulationError(f"dcm2niix produced no NIfTI from the simulated DICOM {dicom_path}")
    return produced[0]


@dataclass
class Phantom:
    base: np.ndarray            # reference intensities on the NIfTI grid dcm2niix produces
    affine: np.ndarray
    brain: np.ndarray
    decoder_pos: np.ndarray     # voxels where the injected signal is added
    manifest: dict
    dicom_from_nifti: Callable[[np.ndarray], np.ndarray]   # NIfTI-grid array -> array in the writer's (x,y,z) convention


def _writer_orientation(converted: np.ndarray, source: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """dcm2niix may flip axes between the DICOM pixel layout and the NIfTI array; find the flips so
    that volumes designed on the NIfTI grid can be written back as DICOMs that convert to themselves."""
    for flips in ((), (0,), (1,), (0, 1), (2,), (0, 2), (1, 2), (0, 1, 2)):
        cand = np.flip(source, axis=flips) if flips else source
        if np.allclose(cand, converted, atol=1.0):
            return lambda arr, f=flips: (np.flip(arr, axis=f) if f else arr)
    raise SimulationError("could not match dcm2niix's output orientation to the simulated DICOM (unexpected axis permutation)")


def require_scanner_tools(settings: Settings) -> None:
    """Fail early with a clear message: dcm2niix always; ANTs for t1/mni without the GPU resampler."""
    if which("dcm2niix") is None:
        raise SimulationError("scanner-level simulation needs dcm2niix on PATH (conda-forge: dcm2niix)")
    p = settings.pipeline
    if p.analysis_space != "epi" and not p.resampler.use_gpu and which("antsApplyTransforms") is None:
        raise SimulationError(f"analysis_space={p.analysis_space} needs antsApplyTransforms on PATH, or set "
                              "pipeline.resampler.use_gpu=true (the simulator provides a sampling grid), or use analysis_space=epi")


def make_phantom(settings: Settings, sim: SimulationConfig, paths: SimPaths) -> Phantom:
    """Everything the real pipeline needs for the phantom subject, generated through dcm2niix so that
    the reference grid is exactly the grid the simulated DICOMs convert to. Reused when
    `out_dir/phantom.json` matches `sim.phantom_params()` (and the files exist)."""
    from . import calibration
    from .pipeline import PipelineError, check_prerequisites

    require_scanner_tools(settings)
    p = settings.pipeline
    lay = paths.layout
    paths.incoming.mkdir(parents=True, exist_ok=True)
    params = {**sim.phantom_params(), "analysis_space": p.analysis_space, "use_gpu": p.resampler.use_gpu}
    manifest = paths.phantom_manifest
    reuse = manifest.exists() and json.loads(manifest.read_text()).get("params") == params and lay.motion_ref_epi.exists() \
        and paths.decoder.exists() and calibration.fieldmap_ready(lay.fmap_dir)
    base_src, brain_src = phantom_base(sim)
    work = paths.out_dir / "phantom_work"
    shutil.rmtree(work, ignore_errors=True)       # stale conversions from another grid must not be picked up
    work.mkdir(parents=True)
    study_uid, series_uid = _uids(sim)
    proto_dcm = write_mosaic_dicom(base_src, sim.voxel_mm, work / "proto.dcm", 0, 1, p.TR, series_uid, study_uid)
    proto_nii = _convert_with_dcm2niix(proto_dcm, work, "proto")
    img = nib.load(str(proto_nii))
    base = np.asarray(img.dataobj, dtype=np.float32)
    if base.ndim == 4:
        base = base[..., 0]
    to_writer = _writer_orientation(base, base_src)
    brain = base > 500.0
    x, y, z = np.meshgrid(*[np.arange(s, dtype=float) for s in base.shape], indexing="ij")
    cx = (base.shape[0] - 1) / 2
    decoder_pos = brain & (x > cx + 1)
    if not reuse:
        lay.trans_dir.mkdir(parents=True, exist_ok=True)
        lay.anat_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(base, img.affine), lay.trans_dir / "epi_unwarped_mean.nii")
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), img.affine), lay.trans_dir / "epi_mask_mean.nii")
        wm = brain & (np.abs(x - cx) < base.shape[0] * 0.1) & (np.abs(y - (base.shape[1] - 1) / 2) < base.shape[1] * 0.1)
        vent = brain & (z <= base.shape[2] * 0.2) & (np.abs(x - cx) < base.shape[0] * 0.06)
        nib.save(nib.Nifti1Image(wm.astype(np.uint8), img.affine), lay.wm_mask)
        nib.save(nib.Nifti1Image(vent.astype(np.uint8), img.affine), lay.vent_mask)
        # fieldmap calibration: AP/PA = the reference, a small smooth field on PyHySCO's staggered grid (PE axis 'j')
        fm = lay.fmap_dir
        shutil.rmtree(fm, ignore_errors=True)
        fm.mkdir(parents=True)
        for name, pe in (("AP", "j"), ("PA", "j-")):
            nib.save(nib.Nifti1Image(base, img.affine), fm / f"{name}.nii")
            (fm / f"{name}.json").write_text(json.dumps({"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05}))
        native, field, man = calibration.native_paths(fm)
        native.mkdir()
        for name in ("AP", "PA"):
            calibration.save_native_mean(fm / f"{name}.nii", native / f"{name}_mean.nii")
        fshape = (base.shape[0], base.shape[1] + 1, base.shape[2])
        fx, fy, fz = np.meshgrid(*[np.arange(s, dtype=float) for s in fshape], indexing="ij")
        fbrain = _ellipsoid(fshape, (cx, base.shape[1] / 2, (base.shape[2] - 1) / 2), (base.shape[0] * 0.42, base.shape[1] * 0.42, base.shape[2] * 0.45))
        nib.save(nib.Nifti1Image((0.3 * np.sin(fx / 4.0) * fbrain).astype(np.float32), img.affine), field)
        man.write_text(json.dumps(calibration.calibration_spec(fm / "AP.nii", fm / "PA.nii")))
        # decoder: + on one side, - on the other, mean-free over the brain, scaled so the injected signal reads ~1
        dec = np.zeros(base.shape, dtype=np.float32)
        dec[decoder_pos] = 1.0 / max(1, decoder_pos.sum())
        neg = brain & ~decoder_pos
        dec[neg] = -1.0 / max(1, neg.sum())
        if p.analysis_space != "epi" and p.resampler.use_gpu:
            _write_identity_gpu_grid(lay, base.shape, img.affine, p.analysis_space)
        nib.save(nib.Nifti1Image(dec, img.affine), paths.decoder)
        manifest.write_text(json.dumps({"params": params, "grid_nifti": list(base.shape), "created": time.time()}, indent=2))
        log.info("[SIM] phantom written under %s", paths.out_dir)
    else:
        log.info("[SIM] reusing phantom under %s", paths.out_dir)
    sim_settings = simulation_settings(settings, paths, sim)
    try:
        check_prerequisites(sim_settings, lay)
    except PipelineError as exc:
        raise SimulationError(f"phantom does not satisfy the pipeline prerequisites: {exc}") from exc
    return Phantom(base=base, affine=img.affine, brain=brain, decoder_pos=decoder_pos,
                   manifest=json.loads(manifest.read_text()), dicom_from_nifti=to_writer)


def _write_identity_gpu_grid(lay: RunLayout, shape, affine, space: str) -> None:
    """A cached sampling grid that maps the 'decoder' grid 1:1 onto the EPI grid, plus the transform files
    the resampler checks for, so t1/mni spaces run without ANTs."""
    nx, ny, nz = shape
    lay.anat_dir.mkdir(parents=True, exist_ok=True)
    lay.t1_to_mni_warp.write_bytes(b"simulated")
    lay.epi2t1_composite.write_bytes(b"simulated")
    out = lay.gpu_grids_dir / f"epi_to_{space}"
    out.mkdir(parents=True, exist_ok=True)
    mx, my, mz = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    g = np.stack([2 * mx / max(nx - 1, 1) - 1, 2 * my / max(ny - 1, 1) - 1, 2 * mz / max(nz - 1, 1) - 1], axis=-1)
    grid = np.transpose(g, (2, 1, 0, 3))[None].astype(np.float32)
    np.savez_compressed(out / "sampling_grid.npz", grid=grid, moving_shape=np.asarray(shape, dtype=np.int32),
                        fixed_shape=np.asarray(shape, dtype=np.int32), fixed_affine=np.asarray(affine, dtype=np.float64))


def _uids(sim: SimulationConfig) -> tuple[str, str]:
    root = "1.2.826.0.1.3680043.10.1234"
    return f"{root}.{sim.seed}.1", f"{root}.{sim.seed}.2"


def synthesize_volume(ph: Phantom, v: int, sim: SimulationConfig, signal_on: bool) -> np.ndarray:
    """Volume v on the NIfTI grid: drift, rigid jitter (a spike every 15th volume), noise, and the
    injected signal in the decoder's positive voxels."""
    from scipy.ndimage import shift as nd_shift

    rng = np.random.default_rng(sim.seed * 100003 + v)
    amp = sim.motion_amp_mm / sim.voxel_mm
    jitter = np.array([np.sin(v / 7.0), np.cos(v / 11.0), np.sin(v / 13.0)]) * amp + rng.normal(0, 0.1 * amp, 3)
    if v % 15 == 0:
        jitter = jitter + np.array([4 * amp, -3 * amp, 0.0])
    data = ph.base * (1.0 + sim.drift * v)
    if signal_on:
        data = data.copy()
        data[ph.decoder_pos] *= 1.0 + sim.signal_amp
    moved = nd_shift(data, jitter, order=1, mode="nearest")
    moved += rng.normal(0, sim.noise_sd, moved.shape)
    return np.clip(moved, 0, None).astype(np.float32)


# ----------------------------------------------------------------------------- feeding the incoming folder
def feed_at_tr(items: Iterable[tuple[str, Union[Path, Callable[[Path], None]]]], incoming: Path, tr: float,
               stop_event: threading.Event, on_item: Optional[Callable[[str], None]] = None) -> int:
    """Deliver each item into `incoming` at the TR: `src` is either a file to copy or a function that
    writes the file. Files are staged on the same filesystem and moved in with os.replace, so the
    watcher only ever sees complete files. Returns the number delivered."""
    staging = incoming.parent / (incoming.name + "_staging")
    staging.mkdir(parents=True, exist_ok=True)
    t_next = time.time()
    n = 0
    for dst_name, src in items:
        if stop_event.is_set():
            break
        tmp = staging / dst_name
        if callable(src):
            src(tmp)
        else:
            shutil.copyfile(src, tmp)
        while not stop_event.is_set():
            remaining = t_next - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.05))
        if stop_event.is_set():
            tmp.unlink(missing_ok=True)
            break
        os.replace(tmp, incoming / dst_name)
        t_next += tr
        n += 1
        if on_item is not None:
            on_item(dst_name)
    return n


def recorded_items(dicom_dir: Path, source_run: int, run: int, max_volumes: Optional[int] = None):
    files = []
    for p in Path(dicom_dir).iterdir():
        parsed = parse_dicom_name(p.name)
        if parsed is not None and parsed[1] == int(source_run):
            files.append((parsed[2], parsed[0], p))
    files.sort()
    if not files:
        raise SimulationError(f"no DICOMs for run {source_run} in {dicom_dir}")
    if max_volumes:
        files = files[:max_volumes]
    return [(f"{series:03d}_{int(run):06d}_{scan:06d}.dcm", path) for scan, series, path in files]


def phantom_items(ph: Phantom, settings: Settings, sim: SimulationConfig, run: int):
    on = signal_schedule(settings, sim)
    study_uid, series_uid = _uids(sim)
    tr = float(settings.pipeline.TR)

    def writer(v):
        def _write(path: Path):
            vol = synthesize_volume(ph, v, sim, on(v))
            write_mosaic_dicom(ph.dicom_from_nifti(vol), sim.voxel_mm, path, run, v, tr, series_uid, study_uid)
        return _write

    return ((f"001_{int(run):06d}_{v:06d}.dcm", writer(v)) for v in range(1, sim.n_volumes + 1))


class ScannerFeeder(threading.Thread):
    """Writes the simulated scanner output into the incoming folder at the TR (daemon thread in the launcher)."""

    def __init__(self, items, incoming: Path, tr: float, start_delay_s: float = 1.0):
        super().__init__(name="scanner-feeder", daemon=True)
        self.items, self.incoming, self.tr, self.start_delay_s = items, Path(incoming), float(tr), float(start_delay_s)
        self.stop_event = threading.Event()
        self.delivered = 0

    def run(self) -> None:
        deadline = time.time() + self.start_delay_s
        while time.time() < deadline and not self.stop_event.is_set():
            time.sleep(0.05)
        self.delivered = feed_at_tr(self.items, self.incoming, self.tr, self.stop_event)

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.join(timeout=timeout)


# ----------------------------------------------------------------------------- summary
def summarize_run(run_dir: Path, settings: Settings, sim: SimulationConfig) -> dict:
    """Counts and the recoverability of the injected signal (scores in 'on' volumes vs 'off')."""
    out = {"run_dir": str(run_dir)}
    scores = run_dir / "scores.csv"
    if not scores.exists():
        return out
    on = signal_schedule(settings, sim)
    rows = list(csv.DictReader(open(scores)))
    vals = {int(r["volume_idx"]): float(r["score_raw"]) for r in rows if r["score_raw"] not in ("", "nan")}
    ready = {int(r["volume_idx"]) for r in rows if r.get("reg_ready") == "1"}
    on_vals = [s for v, s in vals.items() if on(v) and v in ready]
    off_vals = [s for v, s in vals.items() if not on(v) and v in ready]
    out.update({"n_scores": len(vals), "n_reg_ready": len(ready), "mean_on": float(np.mean(on_vals)) if on_vals else None,
                "mean_off": float(np.mean(off_vals)) if off_vals else None})
    if on_vals and off_vals:
        x = np.array([1.0 if on(v) else 0.0 for v in sorted(vals) if v in ready])
        y = np.array([vals[v] for v in sorted(vals) if v in ready])
        out["corr_with_signal"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return out
