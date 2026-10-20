"""Decoder scoring: score = dot(volume[mask], weights) + bias.

Two file formats (docs/architecture.md "Decoder files"):
- ROI text (ATR / export_decoder_roi_txt format): line 1 `nx ny nz`, line 2 `N`,
  then N lines `x y z weight template` (1-based voxel indices), then `0 0 0 bias 0`.
- NIfTI weights: non-zero, non-NaN voxels are the mask and the weights; bias 0.
The decoder must be on the grid of the volumes it scores; `check_shape` makes a
mismatch an error instead of a numpy indexing failure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


class DecoderScorer:
    def __init__(self, decoder_path: Optional[Path], roi_txt: Optional[Path] = None):
        self.decoder_path = Path(decoder_path) if decoder_path is not None else None
        self.roi_txt = Path(roi_txt) if roi_txt is not None else None
        self.mask: np.ndarray
        self.w: np.ndarray
        self.bias = 0.0
        self.template: Optional[np.ndarray] = None
        self.dims: tuple
        if self.roi_txt is not None:
            if not self.roi_txt.exists():
                raise FileNotFoundError(f"decoder ROI file not found: {self.roi_txt}")
            self._load_from_roi()
        elif self.decoder_path is not None:
            if not self.decoder_path.exists():
                raise FileNotFoundError(f"decoder file not found: {self.decoder_path}")
            self._load_from_decoder_nii()
        else:
            raise ValueError("a decoder NIfTI or ROI text file is required")

    @property
    def source(self) -> Path:
        return self.roi_txt if self.roi_txt is not None else self.decoder_path

    def _load_from_roi(self) -> None:
        lines = [ln.strip() for ln in open(self.roi_txt, "r") if ln.strip()]
        if len(lines) < 3:
            raise ValueError(f"ROI file seems too short: {self.roi_txt}")
        nx, ny, nz = map(int, lines[0].split()[:3])
        self.dims = (nx, ny, nz)
        n_vox = int(lines[1].split()[0])
        if len(lines) < 2 + n_vox + 1:
            raise ValueError(f"ROI file {self.roi_txt} inconsistent: expected {n_vox} voxel lines but found {len(lines) - 2}")
        coords, weights, template_vals = [], [], []
        for ln in lines[2 : 2 + n_vox]:
            parts = ln.split()
            if len(parts) < 5:
                continue
            coords.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1))
            weights.append(float(parts[3]))
            template_vals.append(float(parts[4]))
        mask = np.zeros(self.dims, dtype=bool)
        for x, y, z in coords:
            mask[x, y, z] = True
        bias_line = lines[2 + n_vox].split()
        bias = 0.0
        if len(bias_line) >= 4:
            try:
                bias = float(bias_line[3])
            except ValueError:
                bias = 0.0
        self.mask = mask
        self.w = np.asarray(weights, dtype=np.float32)
        self.bias = bias
        self.template = np.asarray(template_vals, dtype=np.float32)

    def _load_from_decoder_nii(self) -> None:
        dec = np.asanyarray(nib.load(str(self.decoder_path)).dataobj).astype(np.float32)
        self.dims = dec.shape
        self.mask = (np.abs(dec) > 0) & ~np.isnan(dec)
        self.w = dec[self.mask].ravel()
        self.bias = 0.0

    def check_shape(self, shape: tuple) -> None:
        if tuple(shape) != tuple(self.dims):
            raise ValueError(f"volume shape {tuple(shape)} does not match decoder grid {tuple(self.dims)} ({self.source})")

    def score_from_array(self, vol_arr: np.ndarray) -> float:
        vol = np.asanyarray(vol_arr, dtype=np.float32)
        self.check_shape(vol.shape)
        vox = vol[self.mask].ravel()
        return float(np.dot(vox, self.w) + self.bias)

    def score_volume(self, vol_path: Path) -> float:
        return self.score_from_array(np.asanyarray(nib.load(str(vol_path)).dataobj))
