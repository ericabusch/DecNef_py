"""build_native_calibration = the original preparation's fieldmap step (AP/PA means + PyHySCO), used by
`decnef calibrate`, by check_prerequisites at startup (old `ensure_fieldmap_only`) and by replay staging.
PyHySCO runs for real (3 iterations on the 24x24x12 synthetic pair)."""
import json
import shutil

import pytest

import synth
from decnef import calibration as C


def _subject_without_native(tmp_path):
    lay, decoder, vols = synth.make_subject(tmp_path)
    shutil.rmtree(C.native_paths(lay.fmap_dir)[0])         # data prepared before native_unwarp_v1 existed
    return lay, decoder


def _fast(monkeypatch, *modules):
    """The same build with max_iter=3 wherever the code under test looks it up."""
    real = C.build_native_calibration
    for mod in modules:
        monkeypatch.setattr(mod, "build_native_calibration", lambda pair, max_iter=50, log_=C.log: real(pair, 3, log_))


def test_build_native_calibration_from_raw_ap_pa(tmp_path):
    lay, _ = _subject_without_native(tmp_path)
    assert not C.fieldmap_ready(lay.fmap_dir) and C.can_build(lay.fmap_dir)
    spec = C.build_native_calibration(lay.fmap_dir, max_iter=3)
    folder, field, manifest = C.native_paths(lay.fmap_dir)
    assert C.fieldmap_ready(lay.fmap_dir) and field.exists() and (folder / "AP_mean.nii").exists()
    assert json.loads(manifest.read_text()) == spec == C.load_calibration(lay.fmap_dir)
    assert spec["phase_encoding_axis"] == 2 and spec["ap"]["path"].endswith("AP.nii")
    stamp = field.stat().st_mtime_ns
    assert C.build_native_calibration(lay.fmap_dir, max_iter=3) == spec and field.stat().st_mtime_ns == stamp   # kept


def test_check_prerequisites_builds_a_missing_calibration(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    import decnef.pipeline as P
    from decnef.config import from_dict

    lay, decoder = _subject_without_native(tmp_path)
    settings = from_dict(synth.new_settings(lay, decoder))
    _fast(monkeypatch, P)
    assert P.check_prerequisites(settings, lay) == decoder
    assert C.fieldmap_ready(lay.fmap_dir)
    for name in ("AP.nii", "PA.nii"):
        (lay.fmap_dir / name).unlink()
    shutil.rmtree(C.native_paths(lay.fmap_dir)[0])
    with pytest.raises(P.PipelineError, match="no raw AP/PA"):
        P.check_prerequisites(settings, lay)


def test_replay_staging_builds_the_calibration_in_the_copy(tmp_path, monkeypatch):
    from decnef.config import from_dict
    from decnef.simulate import SimulationConfig, resolve_sim_paths, stage_subject_copy

    lay, decoder = _subject_without_native(tmp_path)
    settings = from_dict(synth.new_settings(lay, decoder))
    sim = SimulationConfig(level="scanner", out_dir=tmp_path / "sim", from_dicoms=tmp_path, source_run=1)
    paths = resolve_sim_paths(settings, "901", sim, sub="99", day="1")
    _fast(monkeypatch, C)
    stage_subject_copy(settings, paths)
    assert C.fieldmap_ready(paths.layout.fmap_dir)                     # built in the copy
    assert not C.fieldmap_ready(lay.fmap_dir)                          # the real tree untouched
    assert C.load_calibration(paths.layout.fmap_dir)["ap"]["path"].startswith(str(tmp_path / "sim"))
