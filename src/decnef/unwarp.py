"""PyHySCO fieldmap application (unwarping) for one raw BOLD volume at a time.

`PreloadedApplier` is built once per run from a prototype volume and the
calibration's field; `apply_volume` then unwarps each incoming volume with a
fixed sampling grid and Jacobian. Numerics unchanged from `pyhysco_apply.py`.
Until the vendored-code phase, PyHySCO itself is imported from the repository's
`fmri_rt_preproc/PyHySCO-main/src` (see `PYHYSCO_SRC`).
"""
from __future__ import annotations

from pathlib import Path
import logging
import time
import sys
from uuid import uuid4
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PYHYSCO_SRC = Path(__file__).resolve().parents[2] / "fmri_rt_preproc" / "PyHySCO-main" / "src"   # TODO(phase 4): a real dependency
if str(PYHYSCO_SRC) not in sys.path:
    sys.path.append(str(PYHYSCO_SRC))

from EPI_MRI.EPIMRIDistortionCorrection import DataObject, EPIMRIDistortionCorrection
from EPI_MRI.utils import m_plus

log = logging.getLogger(__name__)


def _inverse_permutation(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _load_internal_fieldmap(fieldmap_path: Path, data_obj: DataObject) -> torch.Tensor:
    """
    Load a PyHySCO fieldmap saved with original orientation and convert it to
    PyHySCO's internal orientation.
    """
    field_img = nib.load(str(fieldmap_path))
    field = torch.tensor(np.asarray(field_img.dataobj), dtype=data_obj.dtype, device=data_obj.device)

    expected_external_shape = tuple(m_plus(data_obj.m).cpu().numpy()[data_obj.p])
    if tuple(field.shape) != expected_external_shape:
        raise ValueError(
            "PyHySCO fieldmap shape does not match target EPI geometry. "
            f"Expected {expected_external_shape}, got {tuple(field.shape)}."
        )

    internal_perm = _inverse_permutation(data_obj.p)
    return field.permute(internal_perm).contiguous()


def _to_internal_volume(
    volume_xyz: np.ndarray,
    data_obj: DataObject,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an external-orientation volume to PyHySCO's internal orientation."""
    internal_perm = _inverse_permutation(data_obj.p)
    vol_t = torch.as_tensor(volume_xyz, dtype=dtype, device=data_obj.device)
    return vol_t.permute(internal_perm).contiguous()


def select_device(device: str | None) -> str:
    """
    Mirror gpu_ants_like_resampler device-selection behavior.

    - default device string is "cuda"
    - if explicit/default device is "cuda" but CUDA is unavailable, fall back to "cpu"
    - otherwise pass through device as-is
    """
    requested = "cuda" if device is None else str(device)
    return requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu"


class PreloadedApplier:
    """
    Reusable PyHySCO fieldmap applier for fixed geometry.

    This avoids reloading/initializing DataObject and EPIMRIDistortionCorrection for
    every single volume.
    """

    def __init__(
        self,
        prototype_vol_path: Path,
        fieldmap_path: Path,
        phase_encoding_direction: int = 1,
        polarity: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        interpolation_backend: str = "grid_sample",
    ):
        requested_device = str(device)
        if requested_device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested for PreloadedApplier but unavailable; falling back to CPU.")
            requested_device = "cpu"
        if dtype != torch.float32:
            log.warning("PreloadedApplier currently expects float32 runtime math; overriding dtype=%s -> float32.", dtype)
            dtype = torch.float32
        if interpolation_backend not in {"grid_sample", "gather1d"}:
            raise ValueError(
                f"Unsupported interpolation_backend={interpolation_backend!r}. "
                "Use 'grid_sample' or 'gather1d'."
            )

        self.data_obj = DataObject(
            str(prototype_vol_path),
            str(prototype_vol_path),
            phase_encoding_direction=phase_encoding_direction,
            do_normalize=False,
            dtype=dtype,
            device=requested_device,
        )
        self.corr_obj = EPIMRIDistortionCorrection(self.data_obj, alpha=1.0, beta=0.0)
        self.phase_encoding_direction = int(phase_encoding_direction)
        self.polarity = int(polarity)
        self.fieldmap_path = Path(fieldmap_path)
        self.prototype_vol_path = Path(prototype_vol_path)

        proto_img = nib.load(str(self.prototype_vol_path))
        field_img = nib.load(str(self.fieldmap_path))
        self.prototype_image_shape = tuple(int(v) for v in proto_img.shape[:3])
        self.fieldmap_image_shape = tuple(int(v) for v in field_img.shape[:3])

        b = _load_internal_fieldmap(self.fieldmap_path, self.data_obj)
        if polarity < 0:
            b = -b
        self.fieldmap_internal = b.contiguous()
        self.device = str(self.data_obj.device)
        self.interpolation_backend = interpolation_backend
        self.external_input_shape = tuple(int(v) for v in self.data_obj.m.cpu().numpy()[self.data_obj.p])
        self.external_fieldmap_shape = tuple(int(v) for v in m_plus(self.data_obj.m).cpu().numpy()[self.data_obj.p])
        self.internal_shape = tuple(int(v) for v in self.data_obj.m.tolist())
        self.data_obj_p = tuple(int(v) for v in self.data_obj.p)
        self._internal_perm = _inverse_permutation(self.data_obj.p)
        self._external_perm = list(self.data_obj.p)
        self._init_fast_apply_cache()
        log.info(
            (
                "[PyHySCO:init] prototype_image_shape=%s fieldmap_image_shape=%s data_obj.m=%s "
                "data_obj.p=%s external_input_shape=%s external_fieldmap_shape=%s "
                "fieldmap_internal.shape=%s jac.shape=%s grid.shape=%s"
            ),
            self.prototype_image_shape,
            self.fieldmap_image_shape,
            self.internal_shape,
            self.data_obj_p,
            self.external_input_shape,
            self.external_fieldmap_shape,
            tuple(self.fieldmap_internal.shape),
            tuple(self.jac.shape),
            tuple(self.grid.shape),
        )

    @staticmethod
    def _idx_to_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(idx)
        return 2.0 * idx / float(size - 1) - 1.0

    def _init_fast_apply_cache(self) -> None:
        """
        Precompute a fixed sampling grid and Jacobian for fast per-volume apply.

        For fixed fieldmap b:
            TI(xc) = I(xc + bc) * (1 + dbc)
        where bc, dbc, and therefore xc+bc/Jac are constant across incoming volumes.
        """
        m = self.internal_shape
        if len(m) != 3:
            raise ValueError(f"Fast fieldmap apply currently expects 3D volumes, got shape rank={len(m)}.")

        # Precompute deformation and Jacobian terms from fixed fieldmap.
        bc = self.corr_obj.A.mat_mul(self.fieldmap_internal)
        dbc = self.corr_obj.D.mat_mul(self.fieldmap_internal)
        jac = (1.0 + dbc).reshape(m).contiguous()
        xt = (self.corr_obj.xc + bc).reshape(m)

        # Convert physical coordinates to interpolation indices along distortion dim.
        x_idx = (xt - self.data_obj.omega[-2]) / self.data_obj.h[-1] - 0.5
        x_idx = torch.clamp(x_idx, 0, m[-1] - 1)

        # Identity coordinates for non-distortion axes.
        z_idx = torch.arange(m[0], device=self.data_obj.device, dtype=self.data_obj.dtype).view(m[0], 1, 1).expand(m)
        y_idx = torch.arange(m[1], device=self.data_obj.device, dtype=self.data_obj.dtype).view(1, m[1], 1).expand(m)

        x_norm = self._idx_to_norm(x_idx, m[2]).float()
        y_norm = self._idx_to_norm(y_idx, m[1]).float()
        z_norm = self._idx_to_norm(z_idx, m[0]).float()

        # grid_sample expects [x, y, z] components in the last dimension.
        self.grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).unsqueeze(0).contiguous().to(self.data_obj.device)
        self.jac = jac.float().contiguous().to(self.data_obj.device)
        self.distortion_dim_internal = 2  # matches x_idx construction along m[2]/h[-1]/omega[-2]

        if self.interpolation_backend == "gather1d":
            x0 = torch.floor(x_idx).long().clamp(0, m[self.distortion_dim_internal] - 1)
            x1 = (x0 + 1).clamp(0, m[self.distortion_dim_internal] - 1)
            w = (x_idx - x0.float()).float().contiguous()
            self.x0 = x0.contiguous().to(self.data_obj.device)
            self.x1 = x1.contiguous().to(self.data_obj.device)
            self.w = w.to(self.data_obj.device)

    def _validate_external_volume_shape(self, shape: tuple[int, ...]) -> None:
        if tuple(shape) != self.external_input_shape:
            raise ValueError(
                "Input volume shape mismatch for PreloadedApplier. "
                f"Expected external XYZ shape {self.external_input_shape}, got {tuple(shape)}."
            )

    def _validate_internal_volume_shape(self, shape: tuple[int, ...]) -> None:
        if tuple(shape) != self.internal_shape:
            raise ValueError(
                "Internal PyHySCO volume shape mismatch. "
                f"Expected internal shape {self.internal_shape}, got {tuple(shape)}."
            )

    def _apply_internal_tensor(self, vol_internal: torch.Tensor) -> torch.Tensor:
        self._validate_internal_volume_shape(tuple(vol_internal.shape))
        src = vol_internal.float().contiguous().unsqueeze(0).unsqueeze(0)
        if self.interpolation_backend == "grid_sample":
            sampled = F.grid_sample(
                src,
                self.grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, 0]
        else:
            v0 = torch.gather(vol_internal.float().contiguous(), dim=self.distortion_dim_internal, index=self.x0)
            v1 = torch.gather(vol_internal.float().contiguous(), dim=self.distortion_dim_internal, index=self.x1)
            sampled = (1.0 - self.w) * v0 + self.w * v1
        return (sampled * self.jac).contiguous()

    def apply_volume_tensor_from_tensor(
        self,
        volume_xyz_t: torch.Tensor,
        already_external: bool = True,
    ) -> torch.Tensor:
        """
        Apply fixed-field PyHySCO correction to a tensor that can already be on-device.
        Returns corrected tensor in external XYZ orientation and keeps it on device.
        """
        with torch.inference_mode():
            vol_t = volume_xyz_t.to(device=self.data_obj.device, dtype=torch.float32)
            if already_external:
                self._validate_external_volume_shape(tuple(vol_t.shape))
                vol_internal = vol_t.permute(self._internal_perm).contiguous()
            else:
                self._validate_internal_volume_shape(tuple(vol_t.shape))
                vol_internal = vol_t.contiguous()

            corr_internal = self._apply_internal_tensor(vol_internal)
            return corr_internal.permute(self._external_perm).contiguous()

    def apply_volume_tensor(self, volume_xyz: np.ndarray) -> torch.Tensor:
        """
        Apply fixed-field PyHySCO correction to an external XYZ NumPy volume and
        return corrected tensor in external XYZ orientation on applier device.
        """
        self._validate_external_volume_shape(tuple(volume_xyz.shape))
        with torch.inference_mode():
            vol_t = torch.as_tensor(volume_xyz, dtype=torch.float32, device=self.data_obj.device)
            return self.apply_volume_tensor_from_tensor(vol_t, already_external=True)

    def apply_volume(self, volume_xyz: np.ndarray) -> np.ndarray:
        """
        Apply cached fieldmap/correction objects to one external XYZ 3D volume.
        Returns CPU np.float32.
        """
        self._validate_external_volume_shape(tuple(volume_xyz.shape))
        with torch.inference_mode():
            corr_vol = self.apply_volume_tensor(volume_xyz)
            return corr_vol.detach().cpu().numpy().astype(np.float32, copy=False)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1).astype(np.float64)
    b1 = b.reshape(-1).astype(np.float64)
    if a1.size == 0:
        return float("nan")
    a1 -= a1.mean()
    b1 -= b1.mean()
    denom = np.sqrt((a1 * a1).sum() * (b1 * b1).sum())
    if denom == 0:
        return float("nan")
    return float((a1 * b1).sum() / denom)


def validate_against_existing_pyhysco(
    applier: PreloadedApplier,
    epi_path: Path,
    out_reference_path: Optional[Path] = None,
    n_volumes: int = 5,
) -> dict:
    """
    Validate persistent runtime applier against the file-based wrapper output.
    If out_reference_path is not provided, a temporary reference output is created.
    """
    epi_img = nib.load(str(epi_path))
    epi = np.asarray(epi_img.dataobj)
    if epi.ndim == 3:
        epi_4d = epi[..., np.newaxis]
    elif epi.ndim == 4:
        epi_4d = epi
    else:
        raise ValueError(f"Expected 3D or 4D EPI, got shape {epi.shape}")

    n_use = min(max(1, int(n_volumes)), epi_4d.shape[-1])
    use_tmp_ref = out_reference_path is None
    ref_path = out_reference_path or epi_path.parent / f".__pyhysco_validate_ref_{uuid4().hex}.nii.gz"
    try:
        apply_fieldmap_file(
            epi_path=epi_path,
            fieldmap_path=Path(applier.fieldmap_path),
            out_path=ref_path,
            phase_encoding_direction=applier.phase_encoding_direction,
            polarity=applier.polarity,
            device=applier.device,
            dtype=torch.float32,
        )
        ref = np.asarray(nib.load(str(ref_path)).dataobj)
        if ref.ndim == 3:
            ref = ref[..., np.newaxis]
        pred = np.zeros_like(ref[..., :n_use], dtype=np.float32)
        for t in range(n_use):
            pred[..., t] = applier.apply_volume(epi_4d[..., t])
        target = ref[..., :n_use].astype(np.float32, copy=False)
        diff = pred - target
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        corr = _pearson_corr(pred, target)
        return {
            "mae": mae,
            "rmse": rmse,
            "max_abs": max_abs,
            "pearson_r": corr,
            "shape": tuple(pred.shape),
            "dtype": str(pred.dtype),
            "device": applier.device,
            "n_volumes": n_use,
        }
    finally:
        if use_tmp_ref:
            ref_path.unlink(missing_ok=True)


def benchmark_applier(
    applier: PreloadedApplier,
    epi_path: Path,
    n_iter: int = 50,
    warmup: int = 5,
) -> dict:
    """Benchmark per-volume apply timing for persistent applier."""
    epi = np.asarray(nib.load(str(epi_path)).dataobj)
    if epi.ndim == 4:
        vol = epi[..., 0]
    elif epi.ndim == 3:
        vol = epi
    else:
        raise ValueError(f"Expected 3D/4D EPI for benchmark, got shape {epi.shape}")

    use_cuda = applier.device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(max(0, int(warmup))):
        _ = applier.apply_volume_tensor(vol)
    if use_cuda:
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(max(1, int(n_iter))):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = applier.apply_volume_tensor(vol)
        if use_cuda:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "n_iter": int(arr.size),
        "warmup": int(warmup),
        "device": applier.device,
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def apply_fieldmap_file(
    epi_path: Path,
    fieldmap_path: Path,
    out_path: Path,
    phase_encoding_direction: int = 1,
    polarity: int = 1,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """
    Apply a pre-estimated PyHySCO fieldmap to a 3D or 4D EPI.

    WARNING:
      This function is a backward-compatible file-based wrapper and incurs setup
      overhead when called repeatedly. For real-time usage, instantiate
      PreloadedApplier once per run/session and call apply_volume() or
      apply_volume_tensor() per incoming 3D volume.

    Parameters
    ----------
    epi_path:
        Input distorted 3D or 4D EPI.
    fieldmap_path:
        Fieldmap saved by PyHySCO (`*-EstFieldMap.nii.gz`).
    out_path:
        Output corrected 4D EPI.
    phase_encoding_direction:
        PyHySCO phase encoding axis index (1-based).
    polarity:
        +1 applies correction in the same direction as I1, -1 as I2.
    device:
        Torch device; defaults to cuda if available.
    dtype:
        Torch dtype.
    """
    device = select_device(device)

    epi_img = nib.load(str(epi_path))
    epi = np.asarray(epi_img.dataobj)
    if epi.ndim == 3:
        epi_work = epi[..., np.newaxis]
        squeeze_output = True
    elif epi.ndim == 4:
        epi_work = epi
        squeeze_output = False
    else:
        raise ValueError(f"Expected 3D or 4D EPI, got shape {epi.shape}")

    corrected = np.zeros_like(epi_work, dtype=np.float32)
    run_tag = uuid4().hex
    proto_path = epi_path.parent / f".__pyhysco_tmp_{epi_path.stem}_{run_tag}_proto.nii.gz"
    try:
        nib.save(nib.Nifti1Image(epi_work[..., 0], epi_img.affine, epi_img.header), str(proto_path))
        applier = PreloadedApplier(
            prototype_vol_path=proto_path,
            fieldmap_path=fieldmap_path,
            phase_encoding_direction=phase_encoding_direction,
            polarity=polarity,
            device=device,
            dtype=dtype,
        )
        log.info(
            "Applying PyHySCO fieldmap on device=%s (cuda_available=%s, mps_available=%s)",
            applier.device,
            torch.cuda.is_available(),
            torch.backends.mps.is_available(),
        )

        with torch.inference_mode():
            for t in range(epi_work.shape[-1]):
                corrected[..., t] = applier.apply_volume(epi_work[..., t])
    finally:
        proto_path.unlink(missing_ok=True)

    out_data = corrected[..., 0] if squeeze_output else corrected
    nib.save(nib.Nifti1Image(out_data, epi_img.affine, epi_img.header), str(out_path))
    return out_path
