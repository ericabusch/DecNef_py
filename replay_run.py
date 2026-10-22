#!/usr/bin/env python
"""Replay a recorded run's DICOMs through the real-time pipeline, headless.

Copies `SSS_RRRRRR_VVVVVV.dcm` files from --dicom-dir (those whose run field
== --run) into a temporary incoming folder in scan order, one every --tr
seconds (staged on the same filesystem, then os.replace()d in so the watcher
only ever sees complete files), with the run field rewritten to --replay-run so
the pipeline writes into a fresh run folder. Runs the pipeline with no display
and no physio, waits for the last volume's score (or --timeout), stops, and
copies scores.csv / scores_z.csv / regression_status_rt.csv / fd_rt.csv /
motion_rt.1D to --out. Compare two such folders with compare_scores.py.

    python replay_run.py --dicom-dir /data/recorded/run4 --run 4 --tr 1.4 --driver legacy \
        --sub 00086 --day 3 --base-data /SSD2/DecNef_py/data --replay-run 901 --out /tmp/replay_ref \
        --decoder-template decoders/x.nii --settings-file my_settings.json

--driver legacy: pipeline child started the way the old launchers did (no stop
event, terminate() at the end). --driver session: through rt_session.start_session.
Needs dcm2niix and the offline preparation for that subject/day (see readme_old.txt).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DICOM_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.dcm$")
OUTPUT_FILES = ["scores.csv", "scores_z.csv", "regression_status_rt.csv", "fd_rt.csv", "motion_rt.1D", "session_metadata.json"]


def list_run_dicoms(dicom_dir: Path, run: int) -> list[tuple[int, Path]]:
    out = []
    for p in dicom_dir.iterdir():
        m = DICOM_RE.match(p.name)
        if m and int(m.group(2)) == run:
            out.append((int(m.group(3)), p))
    return sorted(out)


def copy_at_tr(files: list[tuple[int, Path]], incoming: Path, replay_run: int, tr: float, stop: threading.Event) -> None:
    """Thread: move each DICOM into `incoming` every `tr` seconds, in scan order."""
    staging = incoming.parent / (incoming.name + "_staging")
    staging.mkdir(parents=True, exist_ok=True)
    t_next = time.time()
    for scan, src in files:
        if stop.is_set():
            return
        m = DICOM_RE.match(src.name)
        dst_name = f"{m.group(1)}_{replay_run:06d}_{m.group(3)}.dcm"
        tmp = staging / dst_name
        shutil.copyfile(src, tmp)
        now = time.time()
        if t_next > now:
            time.sleep(t_next - now)
        os.replace(tmp, incoming / dst_name)    # atomic within one filesystem
        t_next += tr
        print(f"[replay] {dst_name} ({scan})", flush=True)


def _legacy_pipeline_main(cfg, score_queue, settings):
    """What the old launchers' _run_pipeline_with_settings did (no stop event)."""
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings)
    run_rt_pipeline(cfg, score_queue)


def main() -> int:
    from decnef.cli import ArgFileParser   # `python replay_run.py @legacy.args`: arguments from a file, like the decnef CLI

    p = ArgFileParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, fromfile_prefix_chars="@")
    p.add_argument("--dicom-dir", type=Path, required=True, help="folder with the recorded run's DICOMs")
    p.add_argument("--run", type=int, required=True, help="run field of the recorded DICOM names")
    p.add_argument("--replay-run", type=int, required=True, help="new run id to write into (fresh func/<run> folder)")
    p.add_argument("--tr", type=float, required=True, help="seconds between DICOM arrivals")
    p.add_argument("--sub", required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--base-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="folder to copy the result CSVs into")
    p.add_argument("--driver", choices=["legacy", "session"], default="session")
    p.add_argument("--incoming", type=Path, default=None, help="temporary incoming folder (default: a new temp dir)")
    p.add_argument("--settings-file", type=Path, default=None)
    p.add_argument("--decoder-template", type=Path, default=None)
    p.add_argument("--decoder-roi-txt", type=Path, default=None)
    p.add_argument("--rs", dest="reference_score_run", default=None)
    p.add_argument("--ap-block", type=int, default=None)
    p.add_argument("--pa-block", type=int, default=None)
    p.add_argument("--max-volumes", type=int, default=None, help="replay only the first N scans")
    p.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for the last score after the last copy")
    a = p.parse_args()

    mp.set_start_method("spawn", force=True)
    files = list_run_dicoms(a.dicom_dir, a.run)
    if not files:
        raise SystemExit(f"no DICOMs for run {a.run} in {a.dicom_dir}")
    if a.max_volumes:
        files = files[: a.max_volumes]
    last_scan = files[-1][0]
    incoming = a.incoming or Path(tempfile.mkdtemp(prefix="decnef_replay_")) / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    run_dir_name = str(a.replay_run)
    print(f"[replay] {len(files)} volumes, run {a.run} -> {a.replay_run}, incoming {incoming}, driver {a.driver}")

    from rt_session import PhysioConfig, SessionConfig, prepare_session, start_session

    cfg = SessionConfig(
        sub=a.sub, day=a.day, run=run_dir_name, base_data=a.base_data, incoming_root=incoming,
        settings_file=a.settings_file, decoder_template=a.decoder_template, decoder_roi_txt=a.decoder_roi_txt,
        reference_score_run=a.reference_score_run, ap_block=a.ap_block, pa_block=a.pa_block,
        physio=PhysioConfig(enable=False), qc_viewer=False, stop_timeout_s=30.0,
    )
    prepared = prepare_session(cfg)
    run_dir = prepared["run_dir"]
    if any((run_dir / f).exists() for f in OUTPUT_FILES[:1]):
        raise SystemExit(f"{run_dir} already has scores.csv; use a new --replay-run")

    stop_copy = threading.Event()
    copier = threading.Thread(target=copy_at_tr, args=(files, incoming, a.replay_run, a.tr, stop_copy), daemon=True)

    seen_last = False
    if a.driver == "session":
        with start_session(cfg, prepared) as session:
            copier.start()
            deadline = None
            for ev in session.scores(timeout=0.1):
                if ev.kind in ("score", "tick") and ev.volume_idx == last_scan:
                    seen_last = True
                    break
                if ev.kind in ("end", "error"):
                    break
                if not copier.is_alive():
                    deadline = deadline or time.time() + a.timeout
                    if time.time() > deadline:
                        print("[replay] timeout waiting for the last volume", flush=True)
                        break
    else:
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=200)
        proc = ctx.Process(target=_legacy_pipeline_main, args=(prepared["rt_cfg"], q, prepared["settings"]))
        proc.start()
        copier.start()
        deadline = None
        try:
            while proc.is_alive():
                try:
                    m = q.get(timeout=0.1)
                except Exception:
                    m = None
                if m is not None and int(m.get("volume_idx", 0)) == last_scan:
                    seen_last = True
                    break
                if not copier.is_alive():
                    deadline = deadline or time.time() + a.timeout
                    if time.time() > deadline:
                        print("[replay] timeout waiting for the last volume", flush=True)
                        break
        finally:
            time.sleep(1.0)       # let the last commit finish its CSV rows (the old launchers had no such grace)
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
    stop_copy.set()

    a.out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in OUTPUT_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copyfile(src, a.out / name)
            copied.append(name)
    print(f"[replay] last volume {'seen' if seen_last else 'NOT seen'}; copied {copied} -> {a.out}")
    return 0 if seen_last else 1


if __name__ == "__main__":
    raise SystemExit(main())
