"""Fieldmap calibration for raw-BOLD unwarping (no AP/PA registration).

A calibration lives in `<fmap_dir>/native_unwarp_v1/`: `AP_mean.nii`,
`PA_mean.nii`, the PyHySCO field `pyhysco_native-EstFieldMap.nii` and
`calibration.json` (the spec below). The real-time run validates every raw
volume against it (same grid, same phase-encoding polarity, same readout).
Unchanged from the original `native_fieldmap.py`. `build_native_calibration`
is the original preparation step (`FMRIRealtimePreprocessor._prepare_fieldmap`
+ `_run_ap_pa_pyhysco`): raw AP/PA averaged without motion correction, then
the PyHySCO estimation script with the same parameters.
"""
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

PIPELINE_ORDER = "unwarp_then_mc_v1"
NATIVE_DIR = "native_unwarp_v1"


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def find_source(folder, name):
    for suffix in (".nii", ".nii.gz"):
        path = Path(folder) / (name + suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing raw {name}.nii[.gz] in {folder}; derived AP/PA means are not inputs.")


def sidecar(path):
    path = Path(path)
    name = path.name.removesuffix(".gz").removesuffix(".nii") + ".json"
    meta = path.with_name(name)
    return json.loads(meta.read_text()) if meta.exists() else {}


def calibration_spec(ap, pa):
    ap_meta, pa_meta = sidecar(ap), sidecar(pa)
    pe = [m.get("PhaseEncodingDirection") for m in (ap_meta, pa_meta)]
    known = [v for v in pe if v]
    if any(v not in ("i", "i-", "j", "j-", "k", "k-") for v in known):
        raise ValueError(f"Invalid PhaseEncodingDirection: {pe}")
    if len(known) == 2 and (pe[0][0] != pe[1][0] or pe[0] == pe[1]):
        raise ValueError("AP and PA must have opposite signs on the same voxel axis.")
    readouts = [m.get("TotalReadoutTime") for m in (ap_meta, pa_meta)]
    if all(v is not None for v in readouts) and not np.isclose(*readouts, rtol=1e-4, atol=1e-6):
        raise ValueError("This PyHySCO path requires matching AP/PA readout times.")
    # Existing PA protocol used dimension 2. Metadata overrides this legacy default.
    axis = "ijk".index(known[0][0]) + 1 if known else 2
    return {"order": PIPELINE_ORDER, "motion_correction": "none",
            "ap": source_identity(ap), "pa": source_identity(pa),
            "phase_encoding_axis": axis, "ap_pe": pe[0], "pa_pe": pe[1],
            "readout_time": next((v for v in readouts if v is not None), None)}


def require_same_grid(image, reference):
    if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("Raw BOLD/calibration/reference grids differ; header relabeling is not registration.")


def save_native_mean(source, out):
    image = nib.load(str(source))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(axis=3)
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D/4D calibration: {source}")
    nib.save(nib.Nifti1Image(data, image.affine), str(out))


def native_paths(pair_dir):
    folder = Path(pair_dir) / NATIVE_DIR
    return folder, folder / "pyhysco_native-EstFieldMap.nii", folder / "calibration.json"


def fieldmap_path(pair_dir) -> Path:
    return native_paths(pair_dir)[1]


def fieldmap_ready(pair_dir) -> bool:
    folder, field, manifest = native_paths(pair_dir)
    return all(p.exists() for p in (field, manifest, folder / "AP_mean.nii", folder / "PA_mean.nii"))


def load_calibration(pair_dir):
    folder, field, manifest = native_paths(pair_dir)
    spec = json.loads(manifest.read_text())
    expected = calibration_spec(find_source(pair_dir, "AP"), find_source(pair_dir, "PA"))
    if spec != expected or not all(p.exists() for p in (field, folder / "AP_mean.nii", folder / "PA_mean.nii")):
        raise ValueError("Native calibration is stale or incomplete; rebuild the selected pair.")
    return spec


def can_build(pair_dir) -> bool:
    """Raw AP/PA present, so the native calibration can be built here."""
    try:
        find_source(pair_dir, "AP"), find_source(pair_dir, "PA")
        return True
    except FileNotFoundError:
        return False


def build_native_calibration(pair_dir, max_iter: int = 50, log_=log) -> dict:
    """Build (or refresh) `<pair_dir>/native_unwarp_v1/` from the raw AP/PA in `pair_dir`: means without
    motion correction, then PyHySCO (GaussNewton, alpha=300, beta=1e-4, jac, single precision - the
    script's defaults, as in the original preparation). A complete, current calibration is kept.
    Several minutes on CPU for a 7T volume; seconds on a GPU. Returns the spec."""
    from .unwarp import PYHYSCO_SRC   # vendored PyHySCO; lazy (imports torch)

    pair_dir = Path(pair_dir)
    ap, pa = find_source(pair_dir, "AP"), find_source(pair_dir, "PA")
    spec = calibration_spec(ap, pa)
    if spec["ap_pe"] is None and spec["pa_pe"] is None:
        log_.warning("[FMAP] AP/PA metadata missing: using legacy voxel phase-encoding axis 2")
    folder, field, manifest = native_paths(pair_dir)
    folder.mkdir(parents=True, exist_ok=True)
    ap_mean, pa_mean = folder / "AP_mean.nii", folder / "PA_mean.nii"
    if manifest.exists() and all(p.exists() for p in (field, ap_mean, pa_mean)) and json.loads(manifest.read_text()) == spec:
        return spec
    if manifest.exists():
        manifest.unlink()                      # an interrupted build must never look complete
    require_same_grid(nib.load(str(ap)), nib.load(str(pa)))
    save_native_mean(ap, ap_mean)
    save_native_mean(pa, pa_mean)
    generated = folder / "pyhysco_native-EstFieldMap.nii.gz"
    for stale in (field, generated):
        if stale.exists():
            stale.unlink()
    log_.info("[FMAP] PyHySCO estimation in %s (max_iter=%d)", folder, max_iter)
    env = dict(os.environ, PYTHONPATH=f"{PYHYSCO_SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    subprocess.run([sys.executable, str(PYHYSCO_SRC / "scripts" / "pyhysco.py"), str(ap_mean), str(pa_mean),
                    str(spec["phase_encoding_axis"]), "--output_dir", str(folder / "pyhysco_native"),
                    "--max_iter", str(max_iter), "--correction", "jac"], check=True, env=env, cwd=str(folder))
    if generated.exists():
        with gzip.open(generated, "rb") as f_in, open(field, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    if not field.exists():
        raise FileNotFoundError(f"PyHySCO did not produce {field}")
    manifest.write_text(json.dumps(spec, indent=2))
    log_.info("[FMAP] native calibration built: %s", folder)
    return spec


def validate_bold(image, pair_dir, polarity, source_path=None):
    spec = load_calibration(pair_dir)
    folder, _, _ = native_paths(pair_dir)
    require_same_grid(image, nib.load(str(folder / "AP_mean.nii")))
    if polarity not in ("AP", "PA"):
        raise ValueError("epi_phase_encoding must be AP or PA")
    meta = sidecar(source_path) if source_path else {}
    expected_pe = spec["ap_pe" if polarity == "AP" else "pa_pe"]
    if meta.get("PhaseEncodingDirection") and expected_pe and meta["PhaseEncodingDirection"] != expected_pe:
        raise ValueError("BOLD phase encoding does not match the selected calibration polarity.")
    if meta.get("TotalReadoutTime") is not None and spec["readout_time"] is not None:
        if not np.isclose(meta["TotalReadoutTime"], spec["readout_time"], rtol=1e-4, atol=1e-6):
            raise ValueError("BOLD/calibration readout times differ; displacement scaling is not implemented.")
    return spec
