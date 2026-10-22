"""Behaviour of the new pipeline that has no old counterpart: online ingest through
the watcher, error isolation per volume, the full session path, config
migration of the lab's real settings files, the QC viewer process."""
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import synth
from synth import AFF, N_VOL, REPO

needs_librtp = pytest.mark.skipif(not synth.librtp_loads(), reason="librtp cannot be loaded on this machine")


@needs_librtp
def test_online_ingest_processes_volumes_as_they_arrive(tmp_path, monkeypatch):
    """No backlog: DICOM names appear while the pipeline is online; order and count must hold."""
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    from decnef.layout import RunLayout

    ctx = mp.get_context("spawn")
    lay, decoder, vols = synth.make_subject(tmp_path)
    synth.stage_run(lay, "905", vols, incoming=False)       # raw volumes only, nothing in incoming yet
    settings = synth.new_settings(lay, decoder)
    lay_new = RunLayout(lay.base_data, "99", "1", "905", incoming_root=lay.incoming_root)
    state = {"dropped": 0, "t0": time.time()}

    def feeder(seen):
        # drop the next DICOM once the pipeline is up (first after 2 s), then one per 0.3 s, out of order in pairs
        due = int((time.time() - state["t0"] - 2.0) / 0.3) + 1
        while state["dropped"] < min(due, N_VOL):
            v = state["dropped"] + 1
            pair = [v + 1, v] if v % 2 == 1 and v + 1 <= N_VOL else [v]
            synth.drop_dicoms(lay, "905", pair)
            state["dropped"] += len(pair)

    seen, msgs, code = synth.run_until(ctx, synth.new_pipeline, (settings, lay_new), N_VOL, timeout=120, on_tick=feeder)
    assert (len(seen), code) == (N_VOL, 0)
    order = [m["volume_idx"] for m in msgs if m.get("kind") == "score"]
    assert order == list(range(1, N_VOL + 1))             # committed in scan order despite paired arrivals
    assert not [m for m in msgs if m.get("kind") == "error"]


@needs_librtp
def test_corrupt_volume_is_reported_and_the_run_continues(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    from decnef.layout import RunLayout

    ctx = mp.get_context("spawn")
    lay, decoder, vols = synth.make_subject(tmp_path)
    synth.stage_run(lay, "906", vols)
    bad = lay.base_data / "sub-99/1/func/906/raw/vol_00004.nii"
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), bad)   # wrong grid
    settings = synth.new_settings(lay, decoder)
    seen, msgs, code = synth.run_until(ctx, synth.new_pipeline, (settings, RunLayout(lay.base_data, "99", "1", "906", incoming_root=lay.incoming_root)), N_VOL)
    assert (len(seen), code) == (N_VOL, 0)
    kinds = {m["volume_idx"]: m["kind"] for m in msgs}
    assert kinds[4] == "error" and "grids differ" in [m for m in msgs if m["volume_idx"] == 4][0]["error"]
    assert all(kinds[v] == "score" for v in range(1, N_VOL + 1) if v != 4)
    rows = (lay.base_data / "sub-99/1/func/906/scores.csv").read_text().splitlines()
    assert len(rows) == N_VOL and all(not r.startswith("4,") for r in rows[1:])   # no row for the failed volume
    motion = (lay.base_data / "sub-99/1/func/906/motion_rt.1D").read_text().splitlines()
    assert len(motion) == N_VOL - 1                        # and no zero motion row substituted


@needs_librtp
def test_full_session_path_with_stages(tmp_path, monkeypatch):
    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    from decnef.config import from_dict
    from decnef.session import SessionConfig, start_session

    mp.set_start_method("spawn", force=True)
    lay, decoder, vols = synth.make_subject(tmp_path)
    synth.stage_run(lay, "907", vols)
    settings = from_dict(synth.new_settings(lay, decoder, **{"stages": {"baseline_trs": 2, "cycle": [["cue", 2], ["fb", 1]], "n_cycles": 2}}))
    cfg = SessionConfig(settings, "99", "1", "907", metadata={"launcher": {"name": "test"}}, stop_timeout_s=30.0)
    events, deadline = [], time.time() + 180
    with start_session(cfg) as session:
        for ev in session.scores(timeout=0.5):
            events.append(ev)
            if len([e for e in events if e.kind == "score"]) == N_VOL or time.time() > deadline:
                break
        session.stop()
        events.extend(session.poll())
    from decnef.events import StageSchedule

    scores = [e for e in events if e.kind == "score"]
    assert [e.volume_idx for e in scores] == list(range(1, N_VOL + 1))
    expected = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)
    assert [e.stage for e in scores] == [expected.label_for_volume(v) for v in range(1, N_VOL + 1)]
    assert [e.stage for e in scores][:7] == ["background", "background", "background", "cue", "cue", "fb", "cue"]
    assert events[-1].kind == "end" and session.exit_codes["pipeline"] == 0
    meta = json.loads((cfg.layout.metadata_json).read_text())
    assert meta["launcher"] == {"name": "test"} and meta["stages"]["n_cycles"] == 2 and meta["settings"]["pipeline"]["TR"] == 1.5


@pytest.mark.parametrize("path", sorted(p for p in glob.glob(str(REPO / "*_settings.json")) if "brainpong" not in p) or [None])
def test_lab_settings_files_migrate(path, tmp_path):
    if path is None:
        pytest.skip("no legacy settings files in the repo")
    from decnef.config import from_dict, migrate_legacy

    old = json.loads(Path(path).read_text())
    settings = from_dict(migrate_legacy(old, base_data="/data", incoming_root="/in", decoder_template="dec.nii"))
    assert settings.pipeline.TR == old.get("TR", 1.0)


def test_viewer_process_starts_publishes_and_stops(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from decnef.viewer import ViewerConfig, ViewerHandle

    vol = tmp_path / "v.nii"
    nib.save(nib.Nifti1Image(np.random.rand(6, 6, 4).astype(np.float32), AFF), vol)
    h = ViewerHandle(ViewerConfig(enabled=True, kind="mc", max_queue=1))
    h.start()
    pid = h._process.pid
    for v in range(1, 6):
        h.publish(v, vol)                    # queue of 1: older frames are dropped, never blocks
    h.publish(6, None)
    time.sleep(0.5)
    assert h._process.is_alive()
    h.stop()
    assert h._process is None
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive
    disabled = ViewerHandle(ViewerConfig(enabled=False))
    disabled.start(); disabled.publish(1, vol); disabled.stop()


@needs_librtp
def test_volume_arriving_during_startup_is_not_lost(tmp_path, monkeypatch):
    """A DICOM that lands after the initial folder listing but before the watcher is armed (build_state
    takes seconds) must still be processed. Seen on the scanner machine: volume 1 vanished."""
    import queue
    import threading

    synth.install_dcm2niix_stub(tmp_path, monkeypatch)
    from decnef import pipeline as P
    from decnef.layout import RunLayout

    lay, decoder, vols = synth.make_subject(tmp_path)
    synth.stage_run(lay, "907", vols, incoming=False)
    from decnef.config import from_dict

    settings = from_dict(synth.new_settings(lay, decoder))
    lay_new = RunLayout(lay.base_data, "99", "1", "907", incoming_root=lay.incoming_root)
    real_build_state = P.build_state

    def build_state_with_arrival(*a, **k):
        synth.drop_dicoms(lay, "907", [1])                  # lands in the gap
        return real_build_state(*a, **k)

    monkeypatch.setattr(P, "build_state", build_state_with_arrival)
    q, stop = queue.Queue(), threading.Event()
    t = threading.Thread(target=P.run_pipeline, args=(settings, lay_new, q, stop), daemon=True)
    t.start()
    seen, deadline = [], time.time() + 120
    while len(seen) < 3 and time.time() < deadline:
        try:
            m = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if m.get("kind") == "score":
            seen.append(m["volume_idx"])
            if seen == [1]:
                synth.drop_dicoms(lay, "907", [2, 3])        # online arrivals via the observer
    stop.set()
    t.join(timeout=30)
    assert seen == [1, 2, 3]
