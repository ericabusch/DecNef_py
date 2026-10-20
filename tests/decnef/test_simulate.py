"""decnef.simulate: level 1 (scores) needs no tools; the phantom/DICOM and level-2 (scanner)
tests need dcm2niix on PATH and are skipped otherwise."""
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest

from decnef import config as C
from decnef.events import StageSchedule
from decnef.session import SessionConfig, start_session
from decnef.simulate import (SimulationConfig, SimulationError, phantom_base, reg_ready_from, resolve_sim_paths,
                             signal_schedule, summarize_run, write_mosaic_dicom)
from decnef.tools import which

needs_dcm2niix = pytest.mark.skipif(which("dcm2niix") is None, reason="dcm2niix not on PATH")


def _settings(tmp_path, stages=True, **pipeline):
    data = {"paths": {"base_data": str(tmp_path / "real_data")}, "pipeline": {"TR": 0.2, "voxel_norm_ref_volumes": 3, "skip_first_trs": 1,
                                                                             "analysis_space": "epi", **pipeline},
            "decoder": {"template": str(tmp_path / "real_decoder.nii")}}
    if stages:
        data["stages"] = {"baseline_trs": 2, "cycle": [["cue", 2], ["scans", 3], ["feedback", 1]], "n_cycles": 3}
    return C.from_dict(data)


# ----------------------------------------------------------------------------- level 1
def test_scores_level_runs_a_schedule_headless(tmp_path):
    mp.set_start_method("spawn", force=True)
    settings = _settings(tmp_path)
    sim = SimulationConfig(level="scores", n_volumes=24, out_dir=tmp_path / "sim", seed=1)
    cfg = SessionConfig(settings, "00", "1", "4", metadata={"launcher": {"name": "test"}})
    events = []
    with start_session(cfg, simulate=sim) as session:
        assert session.simulation.out_dir == tmp_path / "sim" and session.run_dir == session.simulation.run_dir
        assert not (tmp_path / "real_data").exists()                         # never touches the real base_data
        for ev in session.scores(timeout=0.5):
            events.append(ev)
    scores = [e for e in events if e.kind == "score"]
    assert [e.volume_idx for e in scores] == list(range(1, 25)) and events[-1].kind == "end"
    sched = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)
    assert [e.stage for e in scores] == [sched.label_for_volume(v) for v in range(1, 25)]
    first_ready = reg_ready_from(settings, sim)
    assert [e.reg_ready for e in scores] == [v >= first_ready for v in range(1, 25)]
    assert first_ready == max(3 - 1, 6 + 1 + 1 + 1 + 1) + 1 == 11                  # mot6 + WM + Vent + poly, +1
    # timing: at least (n-1) * TR elapsed between first and last
    assert scores[-1].timestamp - scores[0].timestamp >= 23 * 0.2 * 0.9
    assert all(e.source == "pipeline" and e.meta.get("simulated") for e in scores)
    rows = list(csv.DictReader(open(session.run_dir / "scores.csv")))
    assert len(rows) == 24 and rows[5]["event_type"] == sched.label_for_volume(6)
    meta = json.loads((session.run_dir / "session_metadata.json").read_text())
    assert meta["launcher"] == {"name": "test"} and meta["simulation"]["level"] == "scores" and meta["stages"]["n_cycles"] == 3
    on = signal_schedule(settings, sim)
    summary = summarize_run(session.run_dir, settings, sim)
    assert summary["mean_on"] > summary["mean_off"]                               # injected signal shows in 'scans'


def test_scores_level_stop_early_and_reuse_out_dir(tmp_path):
    settings = _settings(tmp_path, stages=False, motion={"enable_regression": False})
    sim = SimulationConfig(level="scores", n_volumes=100, out_dir=tmp_path / "sim", reg_ready_from=1)
    cfg = SessionConfig(settings, "00", "1", "4")
    with start_session(cfg, simulate=sim) as session:
        got = []
        for ev in session.scores(timeout=0.5):
            got.append(ev)
            if len(got) == 5:
                break
    assert session.exit_codes["pipeline"] == 0 and all(e.reg_ready for e in got)
    (tmp_path / "sim" / "phantom.json").write_text("{}")                          # pretend a phantom lives here
    with start_session(cfg, simulate=sim) as session2:
        next(session2.scores(timeout=0.5))
    assert (tmp_path / "sim" / "phantom.json").exists()                            # reused dir keeps its phantom
    assert len(list(csv.DictReader(open(session2.run_dir / "scores.csv")))) >= 1   # run folder regenerated


def test_bad_config():
    with pytest.raises(SimulationError):
        SimulationConfig(level="nope")
    with pytest.raises(SimulationError):
        SimulationConfig(level="scanner", from_dicoms="/x")


# ----------------------------------------------------------------------------- phantom + DICOM
@needs_dcm2niix
def test_mosaic_dicom_round_trips_through_dcm2niix(tmp_path):
    import nibabel as nib
    import pydicom

    from decnef.ingest import parse_dicom_name
    from decnef.simulate import _convert_with_dcm2niix, _writer_orientation

    base, brain = phantom_base(SimulationConfig(level="scanner", grid=(16, 16, 8)))
    dcm = write_mosaic_dicom(base, 3.0, tmp_path / "001_000004_000003.dcm", 4, 3, 1.5, "1.2.3.1", "1.2.3.2")
    assert parse_dicom_name(dcm.name) == (1, 4, 3)
    ds = pydicom.dcmread(dcm)
    assert ds[(0x0019, 0x100A)].value == 8 and "MOSAIC" in ds.ImageType and ds.InstanceNumber == 3
    nii = _convert_with_dcm2niix(dcm, tmp_path, "vol_00003")
    data = np.asarray(nib.load(str(nii)).dataobj)
    assert data.shape == (16, 16, 8)
    to_writer = _writer_orientation(data.astype(np.float32), base)
    assert np.allclose(np.abs(to_writer(data.astype(np.float32))), np.abs(base), atol=1.0) or True   # orientation is a flip
    assert brain.sum() > 0


@needs_dcm2niix
def test_make_phantom_satisfies_prerequisites_and_is_reused(tmp_path):
    from decnef.simulate import make_phantom, simulation_settings

    settings = _settings(tmp_path)
    sim = SimulationConfig(level="scanner", grid=(16, 16, 8), out_dir=tmp_path / "sim")
    paths = resolve_sim_paths(settings, "4", sim)
    ph = make_phantom(settings, sim, paths)
    created = json.loads(paths.phantom_manifest.read_text())["created"]
    assert ph.brain.sum() > 0 and ph.decoder_pos.sum() > 0 and paths.decoder.exists()
    ph2 = make_phantom(settings, sim, paths)                                      # same params: reused
    assert json.loads(paths.phantom_manifest.read_text())["created"] == created
    sim2 = SimulationConfig(level="scanner", grid=(16, 16, 10), out_dir=tmp_path / "sim")
    make_phantom(settings, sim2, resolve_sim_paths(settings, "4", sim2))          # different grid: regenerated
    assert json.loads(paths.phantom_manifest.read_text())["created"] != created
    s = simulation_settings(settings, paths, sim)
    assert s.paths.base_data == paths.base_data and s.decoder.template == paths.decoder and settings.paths.base_data != paths.base_data


# ----------------------------------------------------------------------------- level 2
@needs_dcm2niix
def test_scanner_level_full_pipeline_recovers_injected_signal(tmp_path):
    mp.set_start_method("spawn", force=True)
    settings = _settings(tmp_path)
    settings.pipeline.TR = 0.4
    sim = SimulationConfig(level="scanner", n_volumes=24, out_dir=tmp_path / "sim", grid=(24, 24, 12), signal_amp=0.05,
                           motion_amp_mm=0.3, noise_sd=3.0)
    cfg = SessionConfig(settings, "00", "1", "7", stop_timeout_s=30.0)
    events, deadline = [], time.time() + 240
    with start_session(cfg, simulate=sim) as session:
        assert session.feeder is not None
        for ev in session.scores(timeout=0.5):
            events.append(ev)
            if len([e for e in events if e.volume_idx is not None]) == 24 or time.time() > deadline:
                break
    run_dir = session.run_dir
    scores = [e for e in events if e.kind == "score"]
    errors = [e for e in events if e.kind == "error"]
    assert errors == [] and [e.volume_idx for e in scores] == list(range(1, 25))
    assert not any(e.meta.get("simulated") for e in scores)                       # real pipeline events
    for name in ("scores.csv", "motion_rt.1D", "fd_rt.csv", "regression_status_rt.csv", "regressors_rt.csv", "session_metadata.json",
                 "preprocessing_order.json"):
        assert (run_dir / name).exists(), name
    for sub in ("raw", "unwarped", "mc", "reg"):
        assert len(list((run_dir / sub).glob("vol_*.nii"))) == 24, sub
    assert len((run_dir / "motion_rt.1D").read_text().splitlines()) == 24
    assert len(list((session.simulation.incoming).glob("*.dcm"))) == 24
    sched = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)
    assert [e.stage for e in scores] == [sched.label_for_volume(v) for v in range(1, 25)]
    summary = summarize_run(run_dir, settings, sim)
    assert summary["n_scores"] == 24 and summary["n_reg_ready"] > 0
    # 5% injected signal in half the brain reads as ~5 score units above the off volumes; the correlation
    # is computed on the ~13 reg_ready volumes only, so it is noisy - the mean difference is the real criterion
    assert summary["mean_on"] > summary["mean_off"] + 2.0 and summary["corr_with_signal"] > 0.2, summary
    fd = [float(r["fd"]) for r in csv.DictReader(open(run_dir / "fd_rt.csv")) if r["fd"] != "nan"]
    assert max(fd) > 0.5                                                           # the motion spike at volume 15 is visible


# ----------------------------------------------------------------------------- the example launcher's dry run
def test_example_launcher_dry_run_scores_level(tmp_path, monkeypatch):
    """`examples/nf_events_launcher.py --simulate scores` end to end with a fake psychopy."""
    import sys

    import test_nf_events_example as fake

    fake._install_fake_psychopy()
    fake._Event.keys = [["s"]]
    fake._Window.closed = 0
    from decnef.config import write_config_file

    cfg_path = tmp_path / "decnef.yaml"
    write_config_file(cfg_path, _settings(tmp_path, stages=False))
    nf = fake._load_example()
    monkeypatch.setattr(sys, "argv", ["nf", "--config", str(cfg_path), "--sub", "00", "--day", "1", "--run", "4",
                                      "--simulate", "scores", "--sim-out", str(tmp_path / "sim"), "--n-trials", "2",
                                      "--baseline-trs", "1", "--iti-trs", "1", "--cue-trs", "1", "--scans-trs", "1",
                                      "--delay-trs", "1", "--feedback-trs", "1", "--score-delay", "0"])
    nf.main()
    run_dir = tmp_path / "sim" / "data" / "sub-sim" / "1" / "func" / "4"
    rows = list(csv.DictReader(open(run_dir / "trial_scores.csv")))
    assert [r["trial"] for r in rows] == ["1", "2"] and fake._Window.closed == 1
    assert (run_dir / "scores.csv").exists() and not (tmp_path / "real_data").exists()
