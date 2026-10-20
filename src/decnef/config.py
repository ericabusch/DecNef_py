"""Settings: one dataclass tree loaded from one YAML (or JSON) file.

    settings = load_settings("decnef.yaml", overrides=["pipeline.TR=1.4"])
    settings.pipeline.censor.fd.thr

Rules: every value lives here (no module globals, no in-code defaults scattered
through the pipeline); paths and the decoder have **no defaults** (missing =
error); unknown keys are errors; `migrate_legacy(old)` translates a
RegressorSettings-style dict (the old rt_settings.json) into this layout.
Field-by-field documentation, including each field's old name: docs/config.md.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    pass


# ----------------------------------------------------------------------------- sections
@dataclass
class PathsConfig:
    base_data: Path                     # root of sub-<sub>/<day>/... (old --base-data)
    incoming_root: Optional[Path] = None  # folder the scanner writes DICOMs into (old --incoming-root); may be given on the CLI


@dataclass
class DecoderConfig:
    template: Optional[Path] = None     # decoder weights (NIfTI) and the t1/mni resampling reference; required when scoring
    roi_txt: Optional[Path] = None      # ATR-style ROI_DECODER.txt (preferred over the NIfTI when given)
    reference_score_run: Optional[str] = None   # run whose scores.csv gives z-score mean/std (old --rs)
    enable_scoring: bool = True
    enable_original_score: bool = False  # also score the un-regressed volume (score_original)


@dataclass
class MotionConfig:
    enable_regression: bool = True      # old enable_motion_regression
    mot_reg: str = "mot6"
    max_poly_order: float = math.inf
    brain_radius_mm: float = 50.0       # FD radius (was hardcoded)
    volreg_mode: str = "heptic"         # RtpVolreg regmode (was hardcoded)


@dataclass
class FdCensorConfig:
    enable: bool = True                 # old enable_fd_censor_reg
    thr: float = 0.3                    # old fd_thr


@dataclass
class DvarsCensorConfig:
    enable: bool = True                 # old enable_dvars_censor_reg
    thr_robust_z: float = 3.0           # old dvars_thr_robust_z
    warmup: int = 20                    # old dvars_warmup
    mask_source: str = "ref_mask"       # old dvars_mask_source


@dataclass
class CensorConfig:
    fd: FdCensorConfig = field(default_factory=FdCensorConfig)
    dvars: DvarsCensorConfig = field(default_factory=DvarsCensorConfig)
    plus1: bool = True                  # old censor_plus1: also censor the volume after a spike


@dataclass
class NuisanceConfig:
    use_gs: bool = False
    use_wm: bool = True
    use_vent: bool = True


@dataclass
class MaskConfig:
    analysis_mask: str = "whole_brain"  # whole_brain | cortical_gm | custom
    file: Optional[Path] = None         # old analysis_mask_file
    space: str = "final"                # old analysis_mask_space: final | epi | t1 | mni


@dataclass
class FieldmapConfig:
    method: str = "pyhysco"             # pyhysco | ants
    epi_phase_encoding: str = "PA"      # AP | PA
    preloaded: bool = True              # old use_preloaded_pyhysco
    device: str = "cuda"                # old pyhysco_device (CPU fallback when unavailable)
    backend: str = "grid_sample"        # old pyhysco_backend: grid_sample | gather1d
    save_intermediate_unwarped: bool = True


@dataclass
class ResamplerConfig:
    use_gpu: bool = False               # old use_gpu_resampler
    device: str = "cuda"                # old gpu_resampler_device


@dataclass
class EngineConfig:
    max_workers: int = 6
    commit_wait_timeout_s: float = 1.0
    max_retries: int = 3
    rt_max_scan_length: int = 2000


@dataclass
class PipelineConfig:
    TR: float                           # required: seconds per volume
    analysis_space: str = "mni"         # epi | t1 | mni
    voxel_norm_ref_volumes: int = 1     # warm-up window for voxel normalisation / regression readiness
    skip_first_trs: int = 0             # volumes excluded from FD/censoring and from stage labels
    smoothing_fwhm_mm: float = 0.0
    motion: MotionConfig = field(default_factory=MotionConfig)
    censor: CensorConfig = field(default_factory=CensorConfig)
    nuisance: NuisanceConfig = field(default_factory=NuisanceConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    fieldmap: FieldmapConfig = field(default_factory=FieldmapConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    truncate_t1_to_epi_fov: bool = True    # used by the rs_pca plugin's T1 reference
    truncate_t1_padding_vox: int = 2


@dataclass
class PhysioConfig:
    enable: bool = False                # old enable_biopac_physio (no longer defaults to on)
    host: str = "0.0.0.0"
    port: int = 15000
    timeout: float = 0.3
    phys_reg: str = "RICOR8"            # RICOR8 | RVT5 | RVT+RICOR13
    handshake: bool = True
    start_online_only: bool = False
    mode: str = "tcp"                   # tcp | file
    file: Optional[Path] = None
    poll_interval: float = 0.05
    listener: bool = False              # run the receiver in its own process writing the CSV
    timelag: bool = False               # old biopac_timelag


@dataclass
class QcConfig:
    volume_viewer: bool = False         # old enable_volume_streamer
    stream_kind: str = "mc"             # raw | mc | reg | score_input ("unwarped" alias removed; mc is the corrected volume)
    every_n: int = 1
    max_queue: int = 2
    window_title: str = "RT volume QC"


@dataclass
class StagesConfig:
    # volumes <= pipeline.skip_first_trs get pre_label (one setting for pipeline and labels)
    baseline_trs: int = 0
    cycle: list = field(default_factory=list)   # [[name, n_trs], ...]
    n_cycles: Optional[int] = None
    pre_label: str = "background"
    baseline_label: str = "background"
    post_label: str = "post_task"


@dataclass
class Settings:
    paths: PathsConfig
    pipeline: PipelineConfig
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    physio: PhysioConfig = field(default_factory=PhysioConfig)
    qc: QcConfig = field(default_factory=QcConfig)
    stages: Optional[StagesConfig] = None

    def to_dict(self) -> dict:
        return _to_plain(self)

    def copy(self) -> "Settings":
        return from_dict(self.to_dict())


PHYS_REG_COUNTS = {"RICOR8": 8, "RVT5": 5, "RVT+RICOR13": 13}
_CHOICES = {
    "pipeline.analysis_space": {"epi", "t1", "mni"},
    "pipeline.mask.analysis_mask": {"whole_brain", "cortical_gm", "custom"},
    "pipeline.mask.space": {"final", "epi", "t1", "mni"},
    "pipeline.fieldmap.method": {"pyhysco", "ants"},
    "pipeline.fieldmap.epi_phase_encoding": {"AP", "PA"},
    "pipeline.fieldmap.backend": {"grid_sample", "gather1d"},
    "physio.phys_reg": set(PHYS_REG_COUNTS),
    "physio.mode": {"tcp", "file"},
    "qc.stream_kind": {"raw", "mc", "reg", "score_input"},
}


# ----------------------------------------------------------------------------- build / validate
def _unwrap_optional(tp):
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp, True
    return tp, False


def _coerce(value, tp, where: str):
    tp, optional = _unwrap_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where}: value required")
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return _build(tp, value, where)
    if tp is Path:
        return Path(str(value)).expanduser()
    if tp is float:
        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
            return math.inf
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or (isinstance(value, float) and not float(value).is_integer()):
            raise ConfigError(f"{where}: expected an integer")
        return int(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return value.lower() in {"true", "yes", "1"}
        raise ConfigError(f"{where}: expected true/false")
    if tp is str:
        return str(value)
    if tp is list or get_origin(tp) is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        return [list(v) if isinstance(v, (list, tuple)) else v for v in value]
    return value


def _build(cls, data: dict, where: str = ""):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    kwargs = {}
    for f in fields(cls):
        path = f"{where}.{f.name}" if where else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], path)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{path}: required")
    return cls(**kwargs)


def _validate(s: Settings) -> Settings:
    def check_choice(dotted, value):
        allowed = _CHOICES[dotted]
        if value not in allowed:
            raise ConfigError(f"{dotted}: {value!r} not in {sorted(allowed)}")

    s.pipeline.analysis_space = s.pipeline.analysis_space.lower()
    s.pipeline.mask.analysis_mask = s.pipeline.mask.analysis_mask.lower()
    s.pipeline.mask.space = s.pipeline.mask.space.lower()
    s.pipeline.fieldmap.method = s.pipeline.fieldmap.method.lower()
    s.pipeline.fieldmap.epi_phase_encoding = s.pipeline.fieldmap.epi_phase_encoding.upper()
    s.pipeline.fieldmap.backend = s.pipeline.fieldmap.backend.lower()
    s.qc.stream_kind = s.qc.stream_kind.lower()
    for dotted in _CHOICES:
        obj = s
        for part in dotted.split("."):
            obj = getattr(obj, part)
        check_choice(dotted, obj)
    p = s.pipeline
    if not (p.TR > 0):
        raise ConfigError("pipeline.TR must be > 0")
    if not math.isfinite(p.smoothing_fwhm_mm) or p.smoothing_fwhm_mm < 0:
        raise ConfigError("pipeline.smoothing_fwhm_mm must be finite and >= 0")
    if p.voxel_norm_ref_volumes < 1:
        raise ConfigError("pipeline.voxel_norm_ref_volumes must be >= 1")
    if p.skip_first_trs < 0:
        raise ConfigError("pipeline.skip_first_trs must be >= 0")
    if p.engine.max_workers < 1 or p.engine.rt_max_scan_length < 1 or p.engine.max_retries < 0:
        raise ConfigError("pipeline.engine: max_workers/rt_max_scan_length must be >= 1, max_retries >= 0")
    if p.mask.analysis_mask == "custom" and p.mask.file is None:
        raise ConfigError("pipeline.mask.file is required with analysis_mask=custom")
    if s.decoder.enable_scoring and s.decoder.template is None and s.decoder.roi_txt is None:
        raise ConfigError("decoder.template (or decoder.roi_txt) is required when decoder.enable_scoring is true")
    if s.physio.listener and (not s.physio.enable or s.physio.mode != "file"):
        raise ConfigError("physio.listener requires physio.enable=true and physio.mode=file")
    if s.qc.every_n < 1 or s.qc.max_queue < 1:
        raise ConfigError("qc.every_n and qc.max_queue must be >= 1")
    if s.stages is not None:
        for item in s.stages.cycle:
            if not (isinstance(item, (list, tuple)) and len(item) == 2 and int(item[1]) >= 0):
                raise ConfigError("stages.cycle items must be [name, n_trs] with n_trs >= 0")
    return s


def from_dict(data: dict) -> Settings:
    return _validate(_build(Settings, dict(data)))


def _to_plain(obj):
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------- files and overrides
def read_config_file(path: Union[str, Path]) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def write_config_file(path: Union[str, Path], settings: Settings) -> None:
    p = Path(path)
    data = settings.to_dict()
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _parse_override_value(text: str):
    t = text.strip()
    low = t.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    if low in {"inf", "infinity"}:
        return "inf"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    if t.startswith("["):
        return json.loads(t)
    return t


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """`a.b.c=value` strings (CLI --set) applied to the raw dict before validation."""
    out = json.loads(json.dumps(data))  # deep copy of plain data
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r}: expected key.path=value")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {part} is not a section")
        node[parts[-1]] = _parse_override_value(value)
    return out


def load_settings(path: Union[str, Path], overrides: Optional[list[str]] = None, **cli: Any) -> Settings:
    """Read the file, apply `--set` overrides, then keyword overrides for the
    few CLI values (incoming_root=..., etc. given as dotted keys), validate."""
    data = apply_overrides(read_config_file(path), list(overrides or []) + [f"{k}={v}" for k, v in cli.items() if v is not None])
    return from_dict(data)


# ----------------------------------------------------------------------------- legacy translation
_LEGACY_MAP = {
    # RegressorSettings key -> dotted new key
    "TR": "pipeline.TR",
    "analysis_space": "pipeline.analysis_space",
    "voxel_norm_ref_volumes": "pipeline.voxel_norm_ref_volumes",
    "skip_first_trs": "pipeline.skip_first_trs",
    "smoothing_fwhm_mm": "pipeline.smoothing_fwhm_mm",
    "enable_motion_regression": "pipeline.motion.enable_regression",
    "mot_reg": "pipeline.motion.mot_reg",
    "max_poly_order": "pipeline.motion.max_poly_order",
    "enable_fd_censor_reg": "pipeline.censor.fd.enable",
    "fd_thr": "pipeline.censor.fd.thr",
    "enable_dvars_censor_reg": "pipeline.censor.dvars.enable",
    "dvars_thr_robust_z": "pipeline.censor.dvars.thr_robust_z",
    "dvars_warmup": "pipeline.censor.dvars.warmup",
    "dvars_mask_source": "pipeline.censor.dvars.mask_source",
    "censor_plus1": "pipeline.censor.plus1",
    "use_gs": "pipeline.nuisance.use_gs",
    "use_wm": "pipeline.nuisance.use_wm",
    "use_vent": "pipeline.nuisance.use_vent",
    "analysis_mask": "pipeline.mask.analysis_mask",
    "analysis_mask_file": "pipeline.mask.file",
    "analysis_mask_space": "pipeline.mask.space",
    "fieldmap_method": "pipeline.fieldmap.method",
    "epi_phase_encoding": "pipeline.fieldmap.epi_phase_encoding",
    "use_preloaded_pyhysco": "pipeline.fieldmap.preloaded",
    "pyhysco_device": "pipeline.fieldmap.device",
    "pyhysco_backend": "pipeline.fieldmap.backend",
    "save_intermediate_unwarped": "pipeline.fieldmap.save_intermediate_unwarped",
    "use_gpu_resampler": "pipeline.resampler.use_gpu",
    "gpu_resampler_device": "pipeline.resampler.device",
    "max_workers": "pipeline.engine.max_workers",
    "commit_wait_timeout_s": "pipeline.engine.commit_wait_timeout_s",
    "max_retries": "pipeline.engine.max_retries",
    "rt_max_scan_length": "pipeline.engine.rt_max_scan_length",
    "truncate_t1_to_epi_fov": "pipeline.truncate_t1_to_epi_fov",
    "truncate_t1_padding_vox": "pipeline.truncate_t1_padding_vox",
    "enable_biopac_physio": "physio.enable",
    "biopac_host": "physio.host",
    "biopac_port": "physio.port",
    "biopac_timeout": "physio.timeout",
    "biopac_phys_reg": "physio.phys_reg",
    "biopac_handshake": "physio.handshake",
    "biopac_start_online_only": "physio.start_online_only",
    "biopac_mode": "physio.mode",
    "biopac_file": "physio.file",
    "biopac_poll_interval": "physio.poll_interval",
    "biopac_timelag": "physio.timelag",
    "enable_volume_streamer": "qc.volume_viewer",
    "volume_stream_kind": "qc.stream_kind",
    "volume_stream_every_n": "qc.every_n",
    "volume_stream_max_queue": "qc.max_queue",
    "volume_stream_window_title": "qc.window_title",
}
_LEGACY_DROPPED = {"pipeline_engine"}   # only the ordered engine exists now


def migrate_legacy(old: dict, *, base_data=None, incoming_root=None, decoder_template=None, unknown: str = "skip") -> dict:
    """Translate an old rt_settings.json dict into the new layout (as a plain dict,
    ready for `from_dict` after the required paths are filled in).
    Keys the old code never read ("_comment_*", keys absent from RegressorSettings)
    are skipped with a warning, as the old loader silently ignored them; pass
    unknown="error" to make them an error instead."""
    import logging

    out: dict = {"paths": {}, "pipeline": {}, "decoder": {}, "physio": {}, "qc": {}}
    skipped = []
    for key, value in old.items():
        if key in _LEGACY_DROPPED or key.startswith("_"):   # "_comment_*" keys in the old files
            continue
        if key not in _LEGACY_MAP:
            if unknown == "error":
                raise ConfigError(f"legacy key {key!r} has no equivalent")
            skipped.append(key)
            continue
        node = out
        parts = _LEGACY_MAP[key].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if "volume_stream_kind" in old and str(old["volume_stream_kind"]).lower() == "unwarped":
        out["qc"]["stream_kind"] = "mc"   # the old alias for the corrected volume
    if base_data is not None:
        out["paths"]["base_data"] = str(base_data)
    if incoming_root is not None:
        out["paths"]["incoming_root"] = str(incoming_root)
    if decoder_template is not None:
        out["decoder"]["template"] = str(decoder_template)
    if skipped:
        logging.getLogger(__name__).warning("legacy keys with no equivalent were skipped (the old loader ignored them too): %s", skipped)
    return out
