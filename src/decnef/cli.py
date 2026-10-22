"""Command line: `decnef run | replay | compare | config`.

    decnef run --config decnef.yaml --sub 00086 --day 3 --run 4 [--incoming DIR] [--set key.path=value ...]
    decnef simulate --config decnef.yaml --sub .. --day .. --run 4 --level scores|scanner [--out DIR]
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
import sys
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


# ----------------------------------------------------------------------------- simulate / replay
REPLAY_FILES = ["scores.csv", "scores_z.csv", "regression_status_rt.csv", "fd_rt.csv", "motion_rt.1D", "session_metadata.json"]


def _consume(session, n_expected, timeout_after_feed: float, feeder_alive, log_each: bool = True) -> tuple[int, bool]:
    """Headless consumer: print each event, stop after the last expected volume (or a timeout once feeding ended)."""
    seen, deadline, failed, done = set(), None, False, False
    last_event = time.time()
    # poll() rather than scores(): the deadline must fire even if the pipeline never emits anything
    while not done:
        for ev in session.poll():
            last_event = time.time()
            if ev.volume_idx is not None and ev.kind in ("score", "tick", "error"):
                seen.add(ev.volume_idx)
                if log_each:
                    extra = f"score={ev.score_raw:.4f} reg_ready={ev.reg_ready} stage={ev.stage}" if ev.kind == "score" else ev.kind
                    if ev.kind == "error":
                        extra = f"ERROR {ev.meta.get('error')}"
                    log.info("[sim] vol %05d %s", ev.volume_idx, extra)
                if len(seen) >= n_expected:
                    done = True
            if ev.kind == "end":
                done = True
            if ev.kind == "error" and ev.source == "session":
                failed = done = True
        if done:
            break
        if not feeder_alive():
            deadline = deadline or time.time() + timeout_after_feed
        if (deadline and time.time() > deadline) or time.time() - last_event > timeout_after_feed:
            log.error("[sim] timeout: %d/%d volumes seen, nothing from the pipeline for %.0f s", len(seen), n_expected, timeout_after_feed)
            failed = True
            break
        time.sleep(0.05)
    return len(seen), failed


def cmd_simulate(a) -> int:
    from .session import SessionConfig, start_session
    from .simulate import SimulationConfig, summarize_run

    _logging(a.verbose)
    mp.set_start_method("spawn", force=True)
    overrides = a.overrides + ["physio.enable=false", "qc.volume_viewer=false"] if a.level == "scanner" and a.from_dicoms else a.overrides
    settings = load_settings(a.config, overrides)
    sim = SimulationConfig(level=a.level, n_volumes=a.n_volumes, tr=a.tr, out_dir=a.out, motion_amp_mm=a.motion_mm,
                           signal_amp=a.signal, noise_sd=a.noise, seed=a.seed, from_dicoms=a.from_dicoms, source_run=a.source_run)
    cfg = SessionConfig(settings, a.sub, a.day, str(a.run), ap_block=a.ap_block, pa_block=a.pa_block, stop_timeout_s=30.0)
    with start_session(cfg, simulate=sim) as session:
        log.info("[sim] level=%s out_dir=%s run_dir=%s", a.level, session.simulation.out_dir, session.run_dir)
        n_expected = a.n_volumes if not a.from_dicoms else (a.n_volumes or 10 ** 9)
        feeder_alive = (lambda: session.feeder is not None and session.feeder.is_alive())
        n_seen, failed = _consume(session, n_expected, a.timeout, feeder_alive, log_each=not a.quiet)
    summary = summarize_run(session.run_dir, settings, sim)
    summary.update({"volumes_seen": n_seen, "out_dir": str(session.simulation.out_dir), "exit_codes": session.exit_codes})
    print(json.dumps(summary, indent=2))
    return 0 if (not failed and n_seen >= (a.n_volumes if not a.from_dicoms else 1)) else 1


def cmd_replay(a) -> int:
    """`decnef replay` = `decnef simulate --level scanner --from-dicoms DIR --source-run R`."""
    a.level, a.from_dicoms, a.source_run = "scanner", a.dicom_dir, a.run
    a.run = a.replay_run
    a.n_volumes = a.max_volumes or 0
    a.motion_mm = a.signal = a.noise = 0.0
    a.seed, a.quiet = 0, False
    return cmd_simulate(a)


# ----------------------------------------------------------------------------- compare
IGNORE_COLUMNS = {"timestamp", "analysis_timestamp", "watchdog_timestamp"}


ARGFILE_HELP = """

Arguments can come from a file: `decnef replay @my_replay.args`. One or more
arguments per line, shell-style (quotes allowed), `#` starts a comment; extra
arguments after the @file override it.
"""


class ArgFileParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line: str):
        import shlex
        return shlex.split(line, comments=True)


def cmd_calibrate(a) -> int:
    """`decnef calibrate`: the fieldmap-only part of the offline preparation, in the real tree."""
    from .calibration import build_native_calibration, native_paths
    from .layout import RunLayout

    _logging(a.verbose)
    settings = load_settings(a.config, a.overrides)
    layout = RunLayout(settings.paths.base_data, a.sub, a.day, "0", None, a.ap_block, a.pa_block)
    spec = build_native_calibration(layout.fmap_dir, max_iter=a.max_iter, log_=log)
    print(json.dumps({"pair_dir": str(layout.fmap_dir), "calibration": str(native_paths(layout.fmap_dir)[2]), "spec": spec}, indent=2))
    return 0


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
    p = ArgFileParser(prog="decnef", description=__doc__ + ARGFILE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter,
                      fromfile_prefix_chars="@")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the real-time pipeline (no display) for one run")
    _common(r)
    r.add_argument("--run", required=True, help="run id (the run field of the DICOM names)")
    r.add_argument("--incoming", type=Path, default=None, help="incoming DICOM folder (overrides paths.incoming_root)")
    r.set_defaults(func=cmd_run)

    sm = sub.add_parser("simulate", help="run a launcher-less simulation: fake scores (no tools) or synthetic scanner output through the real pipeline")
    _common(sm)
    sm.add_argument("--run", required=True, help="run id for the simulated run (any; the run folder is regenerated)")
    sm.add_argument("--level", choices=["scores", "scanner"], default="scores")
    sm.add_argument("--out", type=Path, default=None, help="simulation directory (default: a fresh temp dir; an existing one is reused)")
    sm.add_argument("--n-volumes", type=int, default=60)
    sm.add_argument("--tr", type=float, default=None,
                    help="seconds between delivered volumes (default: pipeline.TR). Pacing only: the pipeline keeps pipeline.TR")
    sm.add_argument("--motion-mm", type=float, default=0.5, help="rigid jitter amplitude (scanner level)")
    sm.add_argument("--signal", type=float, default=0.02, help="fractional signal injected in the decoder's positive voxels")
    sm.add_argument("--noise", type=float, default=5.0, help="additive noise sd (scanner level)")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--from-dicoms", type=Path, default=None, help="scanner level: replay a recorded run's DICOMs instead of the phantom")
    sm.add_argument("--source-run", type=int, default=None, help="run field of the recorded DICOM names (with --from-dicoms)")
    sm.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for the last volume after feeding ended")
    sm.add_argument("--quiet", action="store_true", help="do not print every event")
    sm.set_defaults(func=cmd_simulate)

    rp = sub.add_parser("replay", help="replay a recorded run's DICOMs at the TR into a fresh run (= simulate --level scanner --from-dicoms)")
    _common(rp)
    rp.add_argument("--dicom-dir", type=Path, required=True)
    rp.add_argument("--run", type=int, required=True, help="run field of the recorded DICOM names")
    rp.add_argument("--replay-run", type=int, required=True, help="new run id to write into")
    rp.add_argument("--out", type=Path, required=True, help="simulation directory (data/, incoming/ and the run folder go here)")
    rp.add_argument("--tr", type=float, default=None,
                    help="seconds between arrivals (default: pipeline.TR). Pacing only: the pipeline keeps pipeline.TR")
    rp.add_argument("--max-volumes", type=int, default=None)
    rp.add_argument("--timeout", type=float, default=120.0)
    rp.set_defaults(func=cmd_replay)

    c = sub.add_parser("compare", help="compare two replay output folders (timestamps ignored)")
    c.add_argument("ref", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--rtol", type=float, default=0.0)
    c.add_argument("--atol", type=float, default=0.0)
    c.set_defaults(func=cmd_compare)

    cb = sub.add_parser("calibrate", help="build the native fieldmap calibration (native_unwarp_v1/) for a subject/day's AP/PA pair")
    _common(cb)
    cb.add_argument("--max-iter", type=int, default=50, help="PyHySCO iterations (the preparation uses 50)")
    cb.set_defaults(func=cmd_calibrate)

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
