"""Command line: `decnef run | replay | compare | config`.

    decnef run --config decnef.yaml --sub 00086 --day 3 --run 4 [--incoming DIR] [--set key.path=value ...]
    decnef replay --config decnef.yaml --sub .. --day .. --dicom-dir DIR --run 4 --replay-run 901 --out OUT
    decnef compare REF NEW
    decnef config migrate old_settings.json --base-data .. --decoder-template .. [--out decnef.yaml]
    decnef config show --config decnef.yaml [--set ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from .config import ConfigError, load_settings, migrate_legacy, read_config_file, write_config_file, from_dict

log = logging.getLogger("decnef")


def _common(p):
    p.add_argument("--config", required=True, type=Path, help="decnef YAML/JSON settings file")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override a setting, e.g. pipeline.TR=1.4")
    p.add_argument("--sub", required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--ap-block", type=int, default=None, help="fieldmap pair (both or neither)")
    p.add_argument("--pa-block", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("watchdog").setLevel(logging.WARNING)


# ----------------------------------------------------------------------------- run
def cmd_run(a) -> int:
    from .layout import RunLayout
    from .pipeline import run_pipeline

    _logging(a.verbose)
    settings = load_settings(a.config, a.overrides, **({"paths.incoming_root": str(a.incoming)} if a.incoming else {}))
    layout = RunLayout(settings.paths.base_data, a.sub, a.day, a.run, settings.paths.incoming_root, a.ap_block, a.pa_block)
    counts = run_pipeline(settings, layout)
    log.info("done: %s", counts)
    return 0 if counts["failed"] == 0 else 1


# ----------------------------------------------------------------------------- replay
DICOM_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.dcm$")
REPLAY_FILES = ["scores.csv", "scores_z.csv", "regression_status_rt.csv", "fd_rt.csv", "motion_rt.1D", "session_metadata.json"]


def list_run_dicoms(dicom_dir: Path, run: int) -> list:
    out = []
    for p in dicom_dir.iterdir():
        m = DICOM_RE.match(p.name)
        if m and int(m.group(2)) == run:
            out.append((int(m.group(3)), p))
    return sorted(out)


def copy_at_tr(files, incoming: Path, replay_run: int, tr: float, stop: threading.Event) -> None:
    """Move each DICOM into `incoming` every `tr` seconds in scan order (staged, then os.replace)."""
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
        os.replace(tmp, incoming / dst_name)
        t_next += tr
        log.info("[replay] %s (%s)", dst_name, scan)


def cmd_replay(a) -> int:
    from .session import SessionConfig, start_session

    _logging(a.verbose)
    mp.set_start_method("spawn", force=True)
    files = list_run_dicoms(a.dicom_dir, a.run)
    if not files:
        raise SystemExit(f"no DICOMs for run {a.run} in {a.dicom_dir}")
    if a.max_volumes:
        files = files[: a.max_volumes]
    last_scan = files[-1][0]
    incoming = a.incoming or Path(tempfile.mkdtemp(prefix="decnef_replay_")) / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    settings = load_settings(a.config, a.overrides + ["physio.enable=false", "qc.volume_viewer=false"],
                             **{"paths.incoming_root": str(incoming)})
    tr = a.tr if a.tr is not None else settings.pipeline.TR
    cfg = SessionConfig(settings, a.sub, a.day, str(a.replay_run), ap_block=a.ap_block, pa_block=a.pa_block, stop_timeout_s=30.0)
    run_dir = cfg.layout.run_dir
    if (run_dir / "scores.csv").exists():
        raise SystemExit(f"{run_dir} already has scores.csv; use a new --replay-run")
    log.info("[replay] %d volumes, run %s -> %s, TR %.3f s, incoming %s", len(files), a.run, a.replay_run, tr, incoming)

    stop_copy = threading.Event()
    copier = threading.Thread(target=copy_at_tr, args=(files, incoming, a.replay_run, tr, stop_copy), daemon=True)
    seen_last = False
    with start_session(cfg) as session:
        copier.start()
        deadline = None
        for ev in session.scores(timeout=0.1):
            if ev.kind in ("score", "tick", "error") and ev.volume_idx == last_scan:
                seen_last = ev.kind != "error"
                break
            if ev.kind == "end" or (ev.kind == "error" and ev.source == "session"):
                break
            if not copier.is_alive():
                deadline = deadline or time.time() + a.timeout
                if time.time() > deadline:
                    log.error("[replay] timeout waiting for the last volume")
                    break
    stop_copy.set()
    a.out.mkdir(parents=True, exist_ok=True)
    copied = [n for n in REPLAY_FILES if (run_dir / n).exists() and shutil.copyfile(run_dir / n, a.out / n)]
    log.info("[replay] last volume %s; copied %s -> %s", "seen" if seen_last else "NOT seen", copied, a.out)
    return 0 if seen_last else 1


# ----------------------------------------------------------------------------- compare
IGNORE_COLUMNS = {"timestamp", "analysis_timestamp", "watchdog_timestamp"}


def _same(x: str, y: str, rtol: float, atol: float) -> bool:
    if x == y:
        return True
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    if math.isnan(fx) and math.isnan(fy):
        return True
    return math.isclose(fx, fy, rel_tol=rtol, abs_tol=atol)


def compare_dirs(ref: Path, new: Path, rtol: float = 0.0, atol: float = 0.0) -> list:
    problems = []
    for name in ("scores.csv", "scores_z.csv", "regression_status_rt.csv", "fd_rt.csv"):
        r, n = ref / name, new / name
        if r.exists() != n.exists():
            problems.append(f"{name}: present in only one folder")
            continue
        if not r.exists():
            continue
        with open(r, newline="") as fr, open(n, newline="") as fn:
            rr, rn = csv.DictReader(fr), csv.DictReader(fn)
            if rr.fieldnames != rn.fieldnames:
                problems.append(f"{name}: columns differ {rr.fieldnames} vs {rn.fieldnames}")
            rows_r, rows_n = list(rr), list(rn)
        if len(rows_r) != len(rows_n):
            problems.append(f"{name}: {len(rows_r)} vs {len(rows_n)} rows")
        for i, (x, y) in enumerate(zip(rows_r, rows_n)):
            for col in x:
                if col in IGNORE_COLUMNS or col not in y:
                    continue
                if not _same(x[col], y[col], rtol, atol):
                    problems.append(f"{name} row {i} ({col}): {x[col]!r} vs {y[col]!r}")
                    if len(problems) > 50:
                        return problems
    r, n = ref / "motion_rt.1D", new / "motion_rt.1D"
    if r.exists() and n.exists():
        a = [l.split() for l in r.read_text().splitlines() if l.strip()]
        b = [l.split() for l in n.read_text().splitlines() if l.strip()]
        if len(a) != len(b):
            problems.append(f"motion_rt.1D: {len(a)} vs {len(b)} rows")
        problems += [f"motion_rt.1D row {i}: {ra} vs {rb}" for i, (ra, rb) in enumerate(zip(a, b))
                     if not all(_same(x, y, rtol, atol) for x, y in zip(ra, rb))][:50]
    return problems


def cmd_compare(a) -> int:
    problems = compare_dirs(a.ref, a.new, a.rtol, a.atol)
    if problems:
        print("\n".join(problems))
        print(f"\nDIFFERENT ({len(problems)} problem(s))")
        return 1
    print("IDENTICAL (timestamp columns ignored)")
    return 0


# ----------------------------------------------------------------------------- config
def cmd_config_migrate(a) -> int:
    old = json.loads(Path(a.old).read_text())
    data = migrate_legacy(old, base_data=a.base_data, incoming_root=a.incoming, decoder_template=a.decoder_template)
    try:
        settings = from_dict(data)
    except ConfigError as exc:
        print(f"translated, but the result is not complete/valid yet: {exc}", file=sys.stderr)
        text = json.dumps(data, indent=2)
        (a.out.write_text(text + "\n") if a.out else print(text))
        return 1
    if a.out:
        write_config_file(a.out, settings)
        print(f"wrote {a.out}")
    else:
        print(json.dumps(settings.to_dict(), indent=2))
    return 0


def cmd_config_show(a) -> int:
    settings = load_settings(a.config, a.overrides)
    print(json.dumps(settings.to_dict(), indent=2))
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decnef", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the real-time pipeline (no display) for one run")
    _common(r)
    r.add_argument("--run", required=True, help="run id (the run field of the DICOM names)")
    r.add_argument("--incoming", type=Path, default=None, help="incoming DICOM folder (overrides paths.incoming_root)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="replay a recorded run's DICOMs at the TR into a fresh run (equivalence check)")
    _common(rp)
    rp.add_argument("--dicom-dir", type=Path, required=True)
    rp.add_argument("--run", type=int, required=True, help="run field of the recorded DICOM names")
    rp.add_argument("--replay-run", type=int, required=True, help="new run id to write into")
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--tr", type=float, default=None, help="seconds between arrivals (default: pipeline.TR)")
    rp.add_argument("--incoming", type=Path, default=None, help="temporary incoming folder (default: a new temp dir)")
    rp.add_argument("--max-volumes", type=int, default=None)
    rp.add_argument("--timeout", type=float, default=120.0)
    rp.set_defaults(func=cmd_replay)

    c = sub.add_parser("compare", help="compare two replay output folders (timestamps ignored)")
    c.add_argument("ref", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--rtol", type=float, default=0.0)
    c.add_argument("--atol", type=float, default=0.0)
    c.set_defaults(func=cmd_compare)

    cf = sub.add_parser("config", help="config file helpers")
    cfs = cf.add_subparsers(dest="config_cmd", required=True)
    m = cfs.add_parser("migrate", help="translate an old rt_settings.json into the new layout")
    m.add_argument("old", type=Path)
    m.add_argument("--base-data", type=Path, default=None)
    m.add_argument("--incoming", type=Path, default=None)
    m.add_argument("--decoder-template", type=Path, default=None)
    m.add_argument("--out", type=Path, default=None, help="write here (.yaml or .json); default: print")
    m.set_defaults(func=cmd_config_migrate)
    s = cfs.add_parser("show", help="print the effective settings")
    s.add_argument("--config", required=True, type=Path)
    s.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    s.set_defaults(func=cmd_config_show)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.func(a))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
