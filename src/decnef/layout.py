"""RunLayout: every path of one subject / day / run, in one place.

    base_data/sub-<sub>/
        anat/                          T1 and derived (shared across days)
        <day>/fmap[/pair-apNNN_paMMM]/ AP/PA fieldmap pair (+ native_unwarp_v1/ calibration)
        <day>/func/trans/              offline preparation outputs (references, transforms, masks)
        <day>/func/<run>/              this run's outputs (raw/ unwarped/ mc/ reg/ t1|mni/ smooth|masked/, CSVs, metadata)

Properties never create directories; call `create_run_dirs()` once at run start.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def prefer_uncompressed(path_nii: Path) -> Path:
    """Return `x.nii` if it exists, else `x.nii.gz` if that exists, else `x.nii`."""
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    return gz if gz.exists() else path_nii


@dataclass(frozen=True)
class RunLayout:
    base_data: Path
    sub: str
    day: str
    run: str
    incoming_root: Optional[Path] = None
    ap_block: Optional[int] = None
    pa_block: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "base_data", Path(self.base_data))
        if self.incoming_root is not None:
            object.__setattr__(self, "incoming_root", Path(self.incoming_root))
        if (self.ap_block is None) != (self.pa_block is None):
            raise ValueError("set both ap_block and pa_block, or neither")

    # ---- subject / day
    @property
    def subject_root(self) -> Path:
        return self.base_data / f"sub-{self.sub}"

    @property
    def anat_dir(self) -> Path:
        return self.subject_root / "anat"

    @property
    def day_root(self) -> Path:
        return self.subject_root / str(self.day)

    @property
    def fmap_dir(self) -> Path:
        if self.ap_block is None:
            return self.day_root / "fmap"
        return self.day_root / "fmap" / f"pair-ap{int(self.ap_block):03d}_pa{int(self.pa_block):03d}"

    @property
    def trans_dir(self) -> Path:
        return self.day_root / "func" / "trans"

    # ---- offline preparation outputs the real-time run reads
    @property
    def motion_ref_epi(self) -> Path:
        """Unwarped mean EPI: motion-correction target and EPI->T1/MNI reference."""
        return prefer_uncompressed(self.trans_dir / "epi_unwarped_mean.nii")

    @property
    def motion_ref_mask(self) -> Path:
        return prefer_uncompressed(self.trans_dir / "epi_mask_mean.nii")

    @property
    def wm_mask(self) -> Path:
        return self.trans_dir / "rt_WM_mask.nii"

    @property
    def vent_mask(self) -> Path:
        return self.trans_dir / "rt_Vent_mask.nii"

    @property
    def epi2t1_composite(self) -> Path:
        return self.trans_dir / "epi2t1_Composite.h5"

    @property
    def epi2t1_inverse_composite(self) -> Path:
        return self.trans_dir / "epi2t1_InverseComposite.h5"

    @property
    def t1_to_mni_warp(self) -> Path:
        return self.anat_dir / "warp_T1_to_MNI_synth.nii"

    @property
    def gpu_grids_dir(self) -> Path:
        return self.trans_dir / "gpu_grids"

    # ---- this run
    @property
    def run_dir(self) -> Path:
        return self.day_root / "func" / str(self.run)

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "raw"

    @property
    def unwarped_dir(self) -> Path:
        return self.run_dir / "unwarped"

    @property
    def mc_dir(self) -> Path:
        return self.run_dir / "mc"

    @property
    def reg_dir(self) -> Path:
        return self.run_dir / "reg"

    def space_dir(self, space: str) -> Path:
        """t1/ or mni/ (epi-space scoring reads reg/ directly)."""
        return self.run_dir / space

    def final_dir(self, kind: str) -> Path:
        """smooth/ or masked/ (final masking + smoothing stage)."""
        return self.run_dir / kind

    STAGE_DIRS = ("raw", "unwarped", "mc", "reg", "t1", "mni", "smooth", "masked")

    def create_run_dirs(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "unwarped", "mc", "reg"):
            (self.run_dir / name).mkdir(exist_ok=True)
        return self.run_dir

    # ---- per-volume files
    @staticmethod
    def vol_name(volume_idx: int, suffix: str) -> str:
        return f"vol_{int(volume_idx):05d}{suffix}.nii"

    def raw_nii(self, v: int) -> Path:
        return self.raw_dir / self.vol_name(v, "")

    def unwarped_nii(self, v: int) -> Path:
        return self.unwarped_dir / self.vol_name(v, "_uw_native")

    def mc_nii(self, v: int) -> Path:
        return self.mc_dir / self.vol_name(v, "_mc")

    def reg_nii(self, v: int) -> Path:
        return self.reg_dir / self.vol_name(v, "_reg")

    def space_nii(self, space: str, v: int, original: bool = False) -> Path:
        return self.space_dir(space) / self.vol_name(v, f"_{space}" + ("_orig" if original else ""))

    def final_nii(self, kind: str, v: int, original: bool = False) -> Path:
        return self.final_dir(kind) / self.vol_name(v, f"_{kind}" + ("_orig" if original else ""))

    # ---- run-level files
    @property
    def metadata_json(self) -> Path:
        return self.run_dir / "session_metadata.json"

    @property
    def provenance_json(self) -> Path:
        return self.run_dir / "preprocessing_order.json"

    @property
    def scores_csv(self) -> Path:
        return self.run_dir / "scores.csv"

    @property
    def scores_z_csv(self) -> Path:
        return self.run_dir / "scores_z.csv"

    @property
    def motion_1d(self) -> Path:
        return self.run_dir / "motion_rt.1D"

    @property
    def fd_csv(self) -> Path:
        return self.run_dir / "fd_rt.csv"

    @property
    def dvars_csv(self) -> Path:
        return self.run_dir / "dvars_rt.csv"

    @property
    def regressors_csv(self) -> Path:
        return self.run_dir / "regressors_rt.csv"

    @property
    def regression_status_csv(self) -> Path:
        return self.run_dir / "regression_status_rt.csv"

    @property
    def biopac_timelag_csv(self) -> Path:
        return self.run_dir / "biopac_timelag.csv"

    @property
    def biopac_regressors_csv(self) -> Path:
        return self.run_dir / "biopac_regressors_rx.csv"

    def other_run(self, run: str) -> "RunLayout":
        """Same subject/day, another run (e.g. the z-score reference run)."""
        return RunLayout(self.base_data, self.sub, self.day, str(run), self.incoming_root, self.ap_block, self.pa_block)
