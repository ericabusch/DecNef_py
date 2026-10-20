"""GPU resampling to the decoder grid (ANTs-like pull resampling with a cached sampling grid).

`precompute_sampling_grid` maps the moving grid's voxel coordinates through the
ANTs transforms once (with antsApplyTransforms) and stores a grid_sample lookup;
`GpuResampler` then warps every volume in one `grid_sample` call. Numerics
unchanged from `gpu_ants_like_resampler.py`. `init_gpu_resampler` is the
run-start setup (ex `rt_pipeline.maybe_init_gpu_resampler`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .tools import run

log = logging.getLogger(__name__)


def _torch_load_compat(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(str(path), map_location="cpu")


def make_voxel_coordinate_images(moving_ref: Path, out_dir: Path) -> dict[str, Path]:
    """Scalar NIfTIs on the moving grid whose values are the voxel x/y/z indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(moving_ref))
    shape = img.shape[:3]
    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    z = np.arange(shape[2], dtype=np.float32)[None, None, :]
    paths = {"x": out_dir / "coord_x_epi_vox.nii", "y": out_dir / "coord_y_epi_vox.nii", "z": out_dir / "coord_z_epi_vox.nii"}
    for axis, arr in (("x", np.broadcast_to(x, shape)), ("y", np.broadcast_to(y, shape)), ("z", np.broadcast_to(z, shape))):
        nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header), str(paths[axis]))
    return paths


def precompute_sampling_grid(moving_ref: Path, fixed_ref: Path, transforms: Sequence[Path], out_dir: Path,
                             grid_name: str = "sampling_grid.npz") -> Path:
    """Build the fixed-grid lookup for grid_sample that reproduces ANTs pull-resampling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_paths = make_voxel_coordinate_images(moving_ref, out_dir)
    warped_paths: dict[str, Path] = {}
    for axis, coord_img in coord_paths.items():
        warped_img = out_dir / f"coord_{axis}_in_fixed.nii"
        if not warped_img.exists():
            cmd = ["antsApplyTransforms", "-d", "3", "-i", str(coord_img), "-r", str(fixed_ref), "-o", str(warped_img)]
            for transform in transforms:
                cmd.extend(["-t", str(transform)])
            cmd.extend(["-n", "Linear", "--float", "1"])
            run(cmd)
        warped_paths[axis] = warped_img

    moving_img = nib.load(str(moving_ref))
    fixed_img = nib.load(str(fixed_ref))
    moving_shape = np.array(moving_img.shape[:3], dtype=np.float32)
    cx = np.asanyarray(nib.load(str(warped_paths["x"])).dataobj).astype(np.float32)
    cy = np.asanyarray(nib.load(str(warped_paths["y"])).dataobj).astype(np.float32)
    cz = np.asanyarray(nib.load(str(warped_paths["z"])).dataobj).astype(np.float32)
    gx = 2.0 * cx / max(float(moving_shape[0] - 1), 1.0) - 1.0
    gy = 2.0 * cy / max(float(moving_shape[1] - 1), 1.0) - 1.0
    gz = 2.0 * cz / max(float(moving_shape[2] - 1), 1.0) - 1.0
    # nibabel uses [X,Y,Z], grid_sample expects [Z,Y,X] layout and [x,y,z] components.
    grid_xyz = np.stack([gx, gy, gz], axis=-1)
    grid_zyx = np.transpose(grid_xyz, (2, 1, 0, 3))
    grid = torch.from_numpy(grid_zyx[None, ...]).float()
    grid_path = out_dir / grid_name
    np.savez_compressed(str(grid_path), grid=grid.numpy(),
                        moving_shape=np.asarray(moving_img.shape[:3], dtype=np.int32),
                        fixed_shape=np.asarray(fixed_img.shape[:3], dtype=np.int32),
                        fixed_affine=fixed_img.affine.astype(np.float64))
    return grid_path


class GpuResampler:
    def __init__(self, grid_path: Path, device: str = "cuda", mode: str = "bilinear", padding_mode: str = "zeros"):
        payload = self._load_payload(grid_path)
        selected_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(selected_device)
        self.mode = mode
        self.padding_mode = padding_mode
        self.grid = payload["grid"].to(self.device, non_blocking=True)
        self.moving_shape = tuple(payload["moving_shape"])
        self.fixed_shape = tuple(payload["fixed_shape"])
        self.fixed_affine = np.asarray(payload["fixed_affine"])
        self._input = torch.empty((1, 1, self.moving_shape[2], self.moving_shape[1], self.moving_shape[0]),
                                  dtype=torch.float32, device=self.device)

    @staticmethod
    def _load_payload(grid_path: Path) -> dict:
        if grid_path.suffix == ".npz":
            with np.load(str(grid_path)) as npz:
                return {"grid": torch.from_numpy(npz["grid"]).float(),
                        "moving_shape": tuple(int(v) for v in npz["moving_shape"].tolist()),
                        "fixed_shape": tuple(int(v) for v in npz["fixed_shape"].tolist()),
                        "fixed_affine": np.asarray(npz["fixed_affine"], dtype=np.float64)}
        payload = _torch_load_compat(grid_path)   # older .pt payloads
        payload["fixed_affine"] = np.asarray(payload["fixed_affine"], dtype=np.float64)
        return payload

    @torch.inference_mode()
    def resample_array(self, data_xyz: np.ndarray) -> np.ndarray:
        if data_xyz.shape[:3] != self.moving_shape:
            raise ValueError(f"Input shape {data_xyz.shape[:3]} does not match {self.moving_shape}.")
        src_zyx = np.transpose(data_xyz.astype(np.float32, copy=False), (2, 1, 0))
        self._input[0, 0].copy_(torch.from_numpy(src_zyx).to(self.device, non_blocking=True))
        out = F.grid_sample(self._input, self.grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)
        return np.transpose(out[0, 0].detach().cpu().numpy(), (2, 1, 0)).astype(np.float32, copy=False)

    def resample_nifti_to_nifti(self, in_path: Path, out_path: Path) -> Path:
        out_data = self.resample_array(np.asanyarray(nib.load(str(in_path)).dataobj))
        nib.save(nib.Nifti1Image(out_data, self.fixed_affine), str(out_path))
        return out_path


def init_gpu_resampler(settings, layout, fixed_ref: Path) -> Optional[GpuResampler]:
    """Build (or load the cached) sampling grid for the configured analysis space; None when disabled."""
    rs = settings.pipeline.resampler
    if not rs.use_gpu:
        return None
    space = settings.pipeline.analysis_space
    if space not in {"t1", "mni"}:
        log.info("[GPU] disabled: analysis_space=%s needs no warp", space)
        return None
    transforms = [layout.epi2t1_composite] if space == "t1" else [layout.t1_to_mni_warp, layout.epi2t1_composite]
    out_dir = layout.gpu_grids_dir / f"epi_to_{space}"
    moving_ref = layout.motion_ref_epi
    missing = [p for p in [fixed_ref, *transforms, moving_ref] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"GPU resampler inputs missing: {missing}")

    grid_npz, grid_pt = out_dir / "sampling_grid.npz", out_dir / "sampling_grid.pt"
    grid_path = grid_npz if grid_npz.exists() else grid_pt
    stale = [grid_npz, grid_pt] + [out_dir / f"coord_{a}_in_fixed.nii" for a in "xyz"]
    if grid_path.exists():
        try:
            fixed_img = nib.load(str(fixed_ref))
            payload = GpuResampler._load_payload(grid_path)
            if tuple(payload["fixed_shape"]) != fixed_img.shape[:3] or not np.allclose(payload["fixed_affine"], fixed_img.affine, atol=1e-3):
                log.warning("[GPU] cached sampling grid does not match the fixed reference; recomputing %s", out_dir)
                for p in stale:
                    p.unlink(missing_ok=True)
                grid_path = grid_npz
        except Exception as exc:
            log.warning("[GPU] could not validate cached sampling grid %s (%s); recomputing", grid_path, exc)
            for p in stale:
                p.unlink(missing_ok=True)
            grid_path = grid_npz
    if not grid_path.exists():
        log.info("[GPU] precomputing sampling grid at %s", out_dir)
        grid_path = precompute_sampling_grid(moving_ref=moving_ref, fixed_ref=fixed_ref, transforms=transforms, out_dir=out_dir)
    resampler = GpuResampler(grid_path=grid_path, device=rs.device, mode="bilinear")
    log.info("[GPU] resampler ready (%s, device=%s)", grid_path, resampler.device)
    return resampler
