"""decnef.prepare: EPI-space preparation from an incoming folder (fieldmap pair, calibration, reference,
mask, stand-in decoder), then the real pipeline on the prepared subject. Needs dcm2niix."""
import json
import multiprocessing as mp
import time

import nibabel as nib
import numpy as np
import pytest

from decnef import calibration as C
from decnef import config as CFG
from decnef.layout import RunLayout
from decnef.simulate import SimulationConfig, phantom_base, write_mosaic_dicom
from decnef.tools import which

needs_dcm2niix = pytest.mark.skipif(which("dcm2niix") is None, reason="dcm2niix not on PATH")
AP, PA, EPI = 2, 3, 4


def _incoming(tmp_path):
    """A session folder as the scanner writes it: AP (series 2), PA (3), a 14-volume EPI (4)."""
    inc = tmp_path / "incoming"
    inc.mkdir()
    base, brain = phantom_base(SimulationConfig(level="scanner", grid=(16, 16, 8)))
    rng = np.random.default_rng(0)
    for series, n in ((AP, 3), (PA, 3), (EPI, 14)):
        for scan in range(1, n + 1):
            vol = base * (1 + 0.001 * scan) + rng.normal(0, 3, base.shape)
            if series == EPI:
                vol = np.roll(vol, scan % 2, axis=0)           # a little motion
            write_mosaic_dicom(vol.astype(np.float32), 3.0, inc / f"001_{series:06d}_{scan:06d}.dcm", series, scan, 1.0,
                               f"1.2.3.{series}", "1.2.3.0")
    return inc


def _settings(tmp_path):
    return CFG.from_dict({"paths": {"base_data": str(tmp_path / "data")},
                          "pipeline": {"TR": 0.3, "voxel_norm_ref_volumes": 3, "skip_first_trs": 1, "analysis_space": "epi",
                                       "nuisance": {"use_wm": False, "use_vent": False}, "fieldmap": {"device": "cpu"}},
                          "decoder": {"template": str(tmp_path / "data/sub-phantom/1/func/trans/phantom_decoder.nii")}})


@needs_dcm2niix
def test_prepare_epi_builds_the_prerequisites_and_the_pipeline_runs(tmp_path, monkeypatch):
    from decnef.pipeline import check_prerequisites
    from decnef.prepare import prepare_epi
    from decnef.session import SessionConfig, start_session

    inc = _incoming(tmp_path)
    settings = _settings(tmp_path)
    lay = RunLayout(settings.paths.base_data, "phantom", "1", "0", incoming_root=inc, ap_block=AP, pa_block=PA)
    out = prepare_epi(settings, lay, inc, AP, PA, EPI, keep=(11, 14), max_iter=3)
    assert (lay.fmap_dir / "AP.nii.gz").exists() and (lay.fmap_dir / "PA.json").exists()
    assert C.fieldmap_ready(lay.fmap_dir) and C.load_calibration(lay.fmap_dir)
    ref, mask = nib.load(str(out["reference"])), nib.load(str(out["mask"]))
    assert ref.shape == (16, 16, 8) and mask.shape == ref.shape and 0 < np.asarray(mask.dataobj).sum() < np.prod(mask.shape)
    assert len(list((lay.trans_dir / "transfer_raw").glob("vol_*.nii"))) == 4 and (lay.trans_dir / "motion.1D").exists()
    assert out["decoder"].exists() and np.count_nonzero(np.asarray(nib.load(str(out["decoder"])).dataobj)) > 0
    assert check_prerequisites(settings, lay) == out["decoder"]
    # idempotent
    stamp = out["reference"].stat().st_mtime_ns
    prepare_epi(settings, lay, inc, AP, PA, EPI, keep=(11, 14), max_iter=3)
    assert out["reference"].stat().st_mtime_ns == stamp

    # the real pipeline on the prepared subject, fed with the EPI series (as a replay of the "NF run")
    mp.set_start_method("spawn", force=True)
    sim = SimulationConfig(level="scanner", out_dir=tmp_path / "replay", from_dicoms=inc, source_run=EPI, n_volumes=6)
    cfg = SessionConfig(settings, "phantom", "1", "901", ap_block=AP, pa_block=PA, stop_timeout_s=30.0)
    events, deadline = [], time.time() + 120
    with start_session(cfg, simulate=sim) as session:
        while len([e for e in events if e.volume_idx is not None]) < 6:
            events.extend(session.poll())
            assert time.time() < deadline, events[-3:]
            time.sleep(0.1)
    scores = [e for e in events if e.kind == "score"]
    assert [e.volume_idx for e in scores] == [1, 2, 3, 4, 5, 6] and not [e for e in events if e.kind == "error"]
    meta = json.loads((session.run_dir / "session_metadata.json").read_text())
    assert meta["settings"]["pipeline"]["analysis_space"] == "epi"
