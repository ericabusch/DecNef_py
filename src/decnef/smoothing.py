"""Final masking + normalized Gaussian smoothing (unchanged numerics from `volume_smooth.py`)."""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from .masks import final_output_kind, prepare_analysis_mask


class MaskedGaussianSmoother:
    """Fixed binary mask; float64 normalized convolution, zero outside the mask.

    Buffers are reused, so call only from the ordered processing stage. The
    returned array belongs to this instance and is overwritten by the next call.
    """

    def __init__(self, mask, affine, fwhm_mm):
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.affine = np.asarray(affine, dtype=float).copy()
        fwhm = float(fwhm_mm)
        if not np.isfinite(fwhm) or fwhm < 0:
            raise ValueError("FWHM must be finite and nonnegative.")
        if self.mask.ndim != 3 or not self.mask.any():
            raise ValueError("Smoothing mask must be a nonempty 3D binary mask.")
        sizes = np.linalg.norm(self.affine[:3, :3], axis=0)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Invalid smoothing voxel sizes.")
        self.output_kind = "smooth" if fwhm > 0 else "masked"
        self.output = np.zeros(self.mask.shape, dtype=np.float64)
        if fwhm == 0:
            return  # Mask-only processing: no Gaussian setup or filtering.
        self.sigma = fwhm / (np.sqrt(8 * np.log(2)) * sizes)
        density = self._filter(self.mask.astype(np.float64))
        self.bad = density[self.mask] < 1e-6
        self.denominator = np.where(self.bad, 1., density[self.mask])
        self.buffer = np.zeros(self.mask.shape, dtype=np.float64)
        self.filtered = np.empty_like(self.buffer)

    def _filter(self, data, output=None):
        return gaussian_filter(data, sigma=self.sigma, output=output,
                               mode="constant", cval=0., truncate=4.)

    def apply(self, data):
        data = np.asarray(data)
        if data.shape != self.mask.shape:
            raise ValueError("Smoothing data and mask shapes differ.")
        values = data[self.mask]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite values inside smoothing mask.")
        if self.output_kind == "masked":
            self.output[self.mask] = values
            return self.output
        self.buffer[self.mask] = values
        self._filter(self.buffer, output=self.filtered)
        result = self.filtered[self.mask] / self.denominator
        result[self.bad] = values[self.bad]
        self.output[self.mask] = result
        return self.output


def prepare_smoother(layout, fwhm_mm, analysis_space, final_reference, run_command,
                     mask_type="whole_brain", custom_file=None, custom_space="final"):
    """Prepare the experiment mask once (into run_dir/<smooth|masked>/); None when no final stage."""
    kind = final_output_kind(fwhm_mm, mask_type)
    if kind is None:
        return None
    folder = layout.final_dir(kind)
    folder.mkdir(parents=True, exist_ok=True)
    mask, affine = prepare_analysis_mask(layout, mask_type, analysis_space, final_reference,
                                        folder, run_command, custom_file, custom_space)
    processor = MaskedGaussianSmoother(mask, affine, fwhm_mm)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(folder / "mask.nii"))
    return processor


def smooth_file(smoother, source, destination):
    """Save atomically: consumers poll for complete per-volume files."""
    img = nib.load(str(source))
    if img.shape != smoother.mask.shape or not np.allclose(img.affine, smoother.affine, atol=1e-4, rtol=0):
        raise ValueError("Final BOLD and smoothing mask grids differ.")
    data = smoother.apply(np.asarray(img.dataobj))
    destination = Path(destination)
    temporary = destination.with_name("." + destination.name)
    try:
        nib.save(nib.Nifti1Image(data, img.affine), str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
