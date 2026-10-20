#!/usr/bin/env python
import argparse
import csv
import json
import logging
from multiprocessing import Queue
from queue import Empty
from pathlib import Path
import multiprocessing as mp
import time
from typing import Optional

import numpy as np
from scipy.stats import norm

from rt_global_settings import load_regressor_settings
from rt_session import (
    PhysioConfig,
    ScoreEvent,
    SessionConfig,
    StageSchedule,
    merge_session_metadata as _merge_session_metadata,  # kept under the old name for callers
    prepare_session,
    start_session,
)
from rt_session import _physio_main as _run_biopac_listener  # noqa: F401  (old name, same function)


log = logging.getLogger(__name__)

# This launcher is a reference example of the session API (docs/session_api.md):
# it owns the NF-events display and trial structure; rt_session owns the
# pipeline / physio processes, metadata and shutdown. Press 's' to start on the
# scanner trigger, 'q' or Escape to end the run cleanly.

# python rt_nf_events_parallel.py \
#   --sub 00086 \
#   --day 3 \
#   --run 4 \
#   --incoming-root /path/to/incoming \
#   --base-data /SSD2/DecNef \
#   --n-trials 20 \
#   --baseline-trs 20 \
#   --iti-trs 3 \
#   --cue-trs 4 \
#   --scans-trs 3 \
#   --delay-trs 3 \
#   --feedback-trs 3 \
#   --score-delay 3 \
#   --settings-file jane_settings.json
#   --rs XXX\


#

def _load_reg_ready_map(run_dir: Path) -> Optional[dict[int, bool]]:
    reg_path = run_dir / "regression_status_rt.csv"
    if not reg_path.exists():
        return None
    with open(reg_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "volume_idx" not in reader.fieldnames or "reg_ready" not in reader.fieldnames:
            return None
        reg_ready_map: dict[int, bool] = {}
        for row in reader:
            try:
                vol = int(row["volume_idx"])
                reg_ready_map[vol] = bool(int(row["reg_ready"]))
            except (TypeError, ValueError):
                continue
    return reg_ready_map


def _plot_qc(run_dir: Path, prefer_reg_ready: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores_path = run_dir / "scores.csv"
    motion_path = run_dir / "motion_rt.1D"
    if not scores_path.exists() or not motion_path.exists():
        return

    reg_ready_map = _load_reg_ready_map(run_dir) if prefer_reg_ready else None
    if prefer_reg_ready and reg_ready_map is None:
        return

    first_reg_ready_vol = None
    if reg_ready_map:
        ready_vols = [vol for vol, ready in reg_ready_map.items() if ready]
        if ready_vols:
            first_reg_ready_vol = min(ready_vols)

    qc_exclude_until_vol = 0
    metadata_path = run_dir / "session_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            qc_exclude_until_vol = max(
                int(metadata.get("voxel_norm_ref_volumes", 0) or 0),
                int(metadata.get("pre_trial_scans", 0) or 0),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            qc_exclude_until_vol = 0

    vols = []
    scores = []
    with open(scores_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vol = int(row["volume_idx"])
                score = float(row["score_raw"])
            except (TypeError, ValueError):
                continue
            if reg_ready_map is not None and not reg_ready_map.get(vol, False):
                continue
            if first_reg_ready_vol is not None and vol <= first_reg_ready_vol:
                continue
            if vol <= qc_exclude_until_vol:
                continue
            vols.append(vol)
            scores.append(score)

    if not scores:
        return

    motion = np.loadtxt(motion_path)
    if motion.ndim == 1:
        motion = motion[None, :]

    motion_vols = np.arange(1, motion.shape[0] + 1)
    include_motion = np.isin(motion_vols, np.asarray(vols, dtype=int))
    motion = motion[include_motion]
    motion_vols = motion_vols[include_motion]

    if motion.shape[0] == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(vols, scores, label="Decoder score (regressed)")
    axes[0].set_xlabel("Volume")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")

    for idx in range(min(motion.shape[1], 6)):
        axes[1].plot(motion_vols, motion[:, idx], label=f"Motion {idx + 1}")
    axes[1].set_xlabel("Volume")
    axes[1].set_ylabel("Motion")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    out_png = run_dir / "qc_scores_motion.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _append_acquisition_speed(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "volume_idx",
                "estimated_trigger_timestamp",
                "watchdog_timestamp",
                "analysis_timestamp",
                "trigger_to_watchdog_s",
                "watchdog_to_analysis_s",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _run_pipeline_with_settings(cfg: "RTSessionConfig", score_queue: Queue, settings: dict) -> None:
    """Old process target, kept for callers; new code goes through rt_session."""
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    for key, value in settings.items():
        if hasattr(REGRESSOR_SETTINGS, key):
            setattr(REGRESSOR_SETTINGS, key, value)
    run_rt_pipeline(cfg, score_queue)


def _release_mouse_cursor(win) -> None:
    try:
        win.mouseVisible = True
    except Exception as exc:
        log.warning("Could not show PsychoPy mouse cursor: %s", exc)

    handles = [getattr(win, "winHandle", None)]
    backend = getattr(win, "backend", None)
    if backend is not None:
        handles.append(getattr(backend, "winHandle", None))

    for handle in handles:
        if handle is None or not hasattr(handle, "set_exclusive_mouse"):
            continue
        try:
            handle.set_exclusive_mouse(False)
        except Exception as exc:
            log.warning("Could not release exclusive mouse control: %s", exc)


def _build_presentation_window(visual, color):
    default_size = (1000, 700)
    window_kwargs = {
        "size": default_size,
        "color": color,
        "units": "pix",
        "screen": 0,
        "fullscr": False,
    }
    try:
        import pyglet

        screens = pyglet.canvas.get_display().get_screens()

        if len(screens) > 1:
            second_screen = screens[1]
            window_kwargs.update(
                {
                    "size": (second_screen.width, second_screen.height),
                    "screen": 1,
                    "fullscr": True,
                }
            )

    except Exception as exc:
        log.warning("Could not detect external monitor; using default window size: %s", exc)

    try:
        win = visual.Window(**window_kwargs)
    except Exception as exc:
        # PsychoPy can fail on fullscreen/monitor-size combinations depending on
        # backend and local monitor settings. Retry with a conservative setup so
        # the presentation can still run.
        log.warning("Primary PsychoPy window config failed; retrying with safe defaults: %s", exc)
        win = visual.Window(
            size=default_size,
            color=color,
            units="pix",
            screen=0,
            fullscr=False,
        )
    _release_mouse_cursor(win)
    return win


def _append_trial_score(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "score_window_start_tr",
                "score_window_end_tr",
                "n_scores_used",
                "trial_score_raw",
                "trial_score",
                "score_missing",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _score_to_percentile(score_raw: float, reference_stats: Optional[dict]) -> float:
    if not np.isfinite(score_raw):
        return float("nan")
    if reference_stats is None:
        return float(np.clip(score_raw, 0.0, 100.0))
    z = (score_raw - reference_stats["mean"]) / reference_stats["std"]
    return float(np.clip(norm.cdf(z) * 100.0, 0.0, 100.0))


def run_nf_events_presentation(
    session,
    skip_first_trs: int,
    baseline_trs: int,
    n_trials: int,
    iti_trs: int,
    cue_trs: int,
    scans_trs: int,
    delay_trs: int,
    feedback_trs: int,
    score_delay: int,
    tr_seconds: float,
    trial_scores_path: Path,
    reference_stats: Optional[dict] = None,
) -> None:
    """The NF-events display. `session` is an rt_session.Session (anything with a
    `poll() -> list[ScoreEvent]`); the display runs in the calling (main) process."""
    from psychopy import core, event, visual

    stage_sum = iti_trs + cue_trs + scans_trs + delay_trs + feedback_trs
    if stage_sum <= 0:
        raise ValueError("At least one per-trial stage duration must be > 0")

    win = _build_presentation_window(visual, color=[-0.004, -0.004, -0.004])
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner trigger ('s')...",
        color="black",
        height=36,
    )
    fixation = visual.ShapeStim(
        win,
        vertices="cross",
        size=[0.1 * win.size[1], 0.1 * win.size[1]],
        lineWidth=1.0,
        fillColor="white",
        lineColor="white",
    )
    feedback_circle = visual.Circle(
        win,
        radius=1,
        lineWidth=10,
        fillColor=[-0.5, 0.7, -0.5],
        lineColor=[-0.5, 0.7, -0.5],
    )
    # Scale circles relative to the presentation window; reference and
    # feedback share the same maximum radius at a perfect score.
    max_reference_radius = 0.60 * min(win.size)
    max_feedback_radius = max_reference_radius
    max_reference_circle = visual.Circle(
        win,
        radius=max_reference_radius,
        fillColor=[-0.004,-0.004,-0.004], #same as background
        lineColor=[0.85, 0.85, 0.85],
        lineWidth=10,
    )
    missing_score_text = visual.TextStim(
        win,
        text="",
        color=[0.9, 0.2, 0.2],
        pos=(0, -int(0.35 * min(win.size))),
        height=30,
    )

    max_seen_vol = 0
    score_by_exp_tr: dict[int, float] = {}
    first_nonbaseline_tr = baseline_trs + 1
    first_trigger_timestamp: Optional[float] = None
    acquisition_speed_path = trial_scores_path.parent / "acquisition_speed_rt.csv"

    def _exp_tr_for_volume(vol_idx: int) -> Optional[int]:
        if vol_idx <= 0:
            return None
        return vol_idx

    def _analysis_tr_for_volume(vol_idx: int) -> Optional[int]:
        exp_tr = _exp_tr_for_volume(vol_idx)
        if exp_tr is None:
            return None
        analysis_tr = exp_tr - skip_first_trs
        if analysis_tr <= 0:
            return None
        return analysis_tr

    def drain_queue() -> None:
        nonlocal max_seen_vol
        for ev in session.poll():
            if ev.kind == "error":
                log.warning("Session error event: %s", ev.meta.get("error"))
            vol_idx = int(ev.volume_idx or 0)
            if vol_idx > 0:
                max_seen_vol = max(max_seen_vol, vol_idx)
            if first_trigger_timestamp is not None and vol_idx > 0:
                estimated_trigger_timestamp = first_trigger_timestamp + ((vol_idx - 1) * tr_seconds)
                watchdog_timestamp = ev.watchdog_timestamp
                analysis_timestamp = ev.analysis_timestamp if ev.analysis_timestamp is not None else ev.timestamp
                if watchdog_timestamp is not None and analysis_timestamp is not None:
                    _append_acquisition_speed(
                        acquisition_speed_path,
                        {
                            "volume_idx": vol_idx,
                            "estimated_trigger_timestamp": f"{estimated_trigger_timestamp:.6f}",
                            "watchdog_timestamp": f"{float(watchdog_timestamp):.6f}",
                            "analysis_timestamp": f"{float(analysis_timestamp):.6f}",
                            "trigger_to_watchdog_s": f"{(float(watchdog_timestamp) - estimated_trigger_timestamp):.6f}",
                            "watchdog_to_analysis_s": f"{(float(analysis_timestamp) - float(watchdog_timestamp)):.6f}",
                        },
                    )
            exp_tr = _analysis_tr_for_volume(vol_idx)
            if exp_tr is None:
                continue

            # Baseline TRs are for normalization / denoising warmup and
            # must never leak into trial-scoring windows.
            if exp_tr < first_nonbaseline_tr:
                continue
            # Accept only explicitly regressed samples (a missing reg_ready counts as not ready).
            if bool(ev.reg_ready) and ev.score_raw is not None:
                score_by_exp_tr[exp_tr] = float(ev.score_raw)

    # Wait for manual trigger.
    while True:
        win.fullscr = True
        _release_mouse_cursor(win)
        drain_queue()
        waiting_text.draw()
        win.flip()
        keys = event.getKeys()
        if "s" in keys:
            first_trigger_timestamp = time.time()
            break
        if "escape" in keys or "q" in keys:
            win.close()
            return
        core.wait(0.02)

    current_end = 0
    schedule: list[dict] = []
    if baseline_trs > 0:
        current_end += baseline_trs
        schedule.append({"trial": 0, "stage": "baseline", "start_tr": 1, "end_tr": current_end})

    trial_windows: dict[int, tuple[int, int]] = {}
    for trial in range(1, n_trials + 1):
        iti_start = current_end + 1
        current_end += iti_trs
        iti_end = current_end

        cue_start = current_end + 1
        current_end += cue_trs
        cue_end = current_end

        scans_start = current_end + 1
        current_end += scans_trs
        scans_end = current_end

        delay_start = current_end + 1
        current_end += delay_trs
        delay_end = current_end

        fb_start = current_end + 1
        current_end += feedback_trs
        fb_end = current_end

        schedule.extend(
            [
                {"trial": trial, "stage": "iti", "start_tr": iti_start, "end_tr": iti_end},
                {"trial": trial, "stage": "cue", "start_tr": cue_start, "end_tr": cue_end},
                {"trial": trial, "stage": "scans", "start_tr": scans_start, "end_tr": scans_end},
                {"trial": trial, "stage": "delay", "start_tr": delay_start, "end_tr": delay_end},
                {"trial": trial, "stage": "feedback", "start_tr": fb_start, "end_tr": fb_end},
            ]
        )

        score_start = cue_start + max(0, score_delay)
        score_start = min(score_start, scans_end)
        trial_windows[trial] = (score_start, scans_end)

    trial_scores: dict[int, float] = {}

    for stage in schedule:
        trial = stage["trial"]
        stage_name = stage["stage"]
        stage_end = stage["end_tr"]

        while True:
            drain_queue()
            current_exp_tr = _analysis_tr_for_volume(max_seen_vol) or 0

            win.color = [-0.004,-0.004,-0.004]  # gray background
            if stage_name == "cue":
                fixation.draw()
            elif stage_name == "feedback" and trial > 0:
                if trial not in trial_scores:
                    s_start, s_end = trial_windows[trial]
                    used = [score_by_exp_tr[tr] for tr in range(s_start, s_end + 1) if tr in score_by_exp_tr]
                    trial_score_raw = float(np.mean(used)) if used else float("nan")
                    trial_score = _score_to_percentile(trial_score_raw, reference_stats)
                    trial_scores[trial] = trial_score
                    score_missing = int(len(used) == 0)
                    _append_trial_score(
                        trial_scores_path,
                        {
                            "trial": trial,
                            "score_window_start_tr": s_start,
                            "score_window_end_tr": s_end,
                            "n_scores_used": len(used),
                            "trial_score_raw": trial_score_raw,
                            "trial_score": trial_score,
                            "score_missing": score_missing,
                        },
                    )
                    if score_missing:
                        log.warning(
                            "Trial %02d score not received in time for TR window [%d, %d]",
                            trial,
                            s_start,
                            s_end,
                        )
                    print(
                        f"Trial {trial:02d} score over TRs [{s_start}, {s_end}] "
                        f"from {len(used)} samples: raw={trial_score_raw:.4f} percentile={trial_score:.4f}"
                    )

                t_score = trial_scores[trial]
                if np.isnan(t_score):
                    radius = 0.0
                else:
                    rating = float(np.clip(t_score, 0.0, 100.0))
                    radius = max_feedback_radius * (rating / 100.0)
                feedback_circle.setRadius(radius, log=False)
                # print(feedback_circle.fillColor)
                feedback_circle.fillColor = [-0.5, 0.7, -0.5]
                feedback_circle.lineColor = [-0.5, 0.7, -0.5]
                # Draw both circles on every feedback frame. Draw the filled
                # feedback first, then the outline reference on top so both are
                # visible even when the score reaches the maximum radius.
                max_reference_circle.draw()
                feedback_circle.draw()
                missing_score_text.text = "Score not received in time" if np.isnan(t_score) else ""
                if missing_score_text.text:
                    missing_score_text.draw()

            win.flip()

            keys = event.getKeys()
            if "escape" in keys or "q" in keys:
                win.close()
                return

            if current_exp_tr >= stage_end:
                break
            core.wait(0.02)

    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run rt_pipeline in parallel with an event-based NF PsychoPy presentation."
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--incoming-root",
        required=False,
        default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya",
        help="Folder where scanner writes DICOMs in real-time.",
    )
    parser.add_argument(
        "--base-data",
        required=False,
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder (same as offline pipeline).",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of NF trials to run.",
    )
    parser.add_argument(
        "--skip-first-trs",
        type=int,
        default=0,
        help="TRs to skip entirely after trigger (excluded from baseline/trials/scoring).",
    )
    parser.add_argument("--baseline-trs", type=int, default=None, help="Baseline duration before trial 1 (TRs).")
    parser.add_argument("--iti-trs", type=int, default=3, help="ITI duration (TRs).")
    parser.add_argument("--cue-trs", type=int, default=4, help="Cue duration (TRs).")
    parser.add_argument("--scans-trs", type=int, default=3, help="Scans-collection duration (TRs).")
    parser.add_argument("--delay-trs", type=int, default=3, help="Delay duration (TRs).")
    parser.add_argument("--feedback-trs", type=int, default=3, help="Feedback duration (TRs).")
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--score-delay",
        type=int,
        default=3,
        help="TRs to wait after cue onset before score-window accumulation starts.",
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP/file input.",
    )
    parser.add_argument("--biopac-host", default="0.0.0.0", help="Host to bind BIOPAC receiver.")
    parser.add_argument("--biopac-port", type=int, default=15000, help="Port to bind BIOPAC receiver.")
    parser.add_argument(
        "--biopac-timeout",
        type=float,
        default=0.3,
        help="Seconds to wait for physio regressors before zero-fill.",
    )
    parser.add_argument(
        "--biopac-phys-reg",
        default="RICOR8",
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
        help="Physio regressor family to expect from BIOPAC stream.",
    )
    parser.add_argument(
        "--biopac-handshake",
        action="store_true",
        default=True,
        help="Send a handshake with TR to the BIOPAC streamer.",
    )
    parser.add_argument(
        "--biopac-start-online",
        action="store_true",
        default=False,
        help="Defer BIOPAC receiver start until after offline DICOM processing.",
    )
    parser.add_argument(
        "--biopac-mode",
        default="tcp",
        choices=["tcp", "file"],
        help="BIOPAC input mode: tcp (listen on socket) or file (tail CSV).",
    )
    parser.add_argument(
        "--biopac-file",
        default=None,
        help="Path to BIOPAC regressors CSV when using --biopac-mode=file.",
    )
    parser.add_argument(
        "--biopac-poll",
        type=float,
        default=0.05,
        help="Polling interval (seconds) for file-backed BIOPAC buffer.",
    )
    parser.add_argument(
        "--biopac-listener",
        action="store_true",
        help="Spawn a dedicated BIOPAC listener process that writes regressors to CSV.",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    parser.add_argument(
        "--ap-block",
        type=int,
        default=None,
        help="AP b0 block to use",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        default=None,
        help="PA b0 block to use",
    )

    args = parser.parse_args()
    from rt_pipeline import REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))

    baseline_trs = (
        int(args.baseline_trs)
        if args.baseline_trs is not None
        else int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)
    )
    if args.skip_first_trs < 0:
        raise ValueError("--skip-first-trs must be >= 0")
    if baseline_trs < 0:
        raise ValueError("--baseline-trs must be >= 0")

    for name in ["n_trials", "iti_trs", "cue_trs", "scans_trs", "delay_trs", "feedback_trs", "score_delay"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    run_dir = Path(args.base_data) / f"sub-{args.sub}" / args.day / "func" / args.run
    trial_scores_path = run_dir / "trial_scores.csv"
    stage_cycle = [
        ("iti", args.iti_trs),
        ("cue", args.cue_trs),
        ("scans", args.scans_trs),
        ("delay", args.delay_trs),
        ("feedback", args.feedback_trs),
    ]

    cfg = SessionConfig(
        sub=args.sub,
        day=args.day,
        run=args.run,
        base_data=Path(args.base_data),
        incoming_root=Path(args.incoming_root),
        settings_file=Path(args.settings_file) if args.settings_file else None,
        # Baseline TRs define the warmup window for voxel-wise normalization and
        # denoising startup; keep pipeline normalization in sync with presentation.
        # (--skip-first-trs is display-side only, as before.)
        settings_overrides={"voxel_norm_ref_volumes": max(1, baseline_trs)},
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
        reference_score_run=args.reference_score_run,
        ap_block=args.ap_block,
        pa_block=args.pa_block,
        physio=PhysioConfig(
            enable=args.biopac_enable,
            host=args.biopac_host,
            port=args.biopac_port,
            timeout=args.biopac_timeout,
            phys_reg=args.biopac_phys_reg,
            handshake=args.biopac_handshake,
            start_online_only=args.biopac_start_online,
            mode=args.biopac_mode,
            file=Path(args.biopac_file) if args.biopac_file else None,
            poll_interval=args.biopac_poll,
            listener=args.biopac_listener,
        ),
        # generic stage labels for scores.csv / ScoreEvent.stage ...
        stages=StageSchedule(
            skip_first_trs=args.skip_first_trs,
            baseline_trs=baseline_trs,
            cycle=stage_cycle,
            n_cycles=args.n_trials,
        ),
        # ... and the launcher's own record (the "script" key keeps older readers working)
        metadata={
            "psychopy": {
                "script": "rt_nf_events_parallel.py",
                "n_trials": args.n_trials,
                "skip_first_trs": args.skip_first_trs,
                "baseline_trs": baseline_trs,
                "iti_trs": args.iti_trs,
                "cue_trs": args.cue_trs,
                "scans_trs": args.scans_trs,
                "delay_trs": args.delay_trs,
                "feedback_trs": args.feedback_trs,
                "score_delay": args.score_delay,
                "trial_scores_csv": str(trial_scores_path),
                "decoder_template": str(args.decoder_template) if args.decoder_template else None,
            }
        },
        queue_size=200,
    )

    prepared = prepare_session(cfg)
    try:
        with start_session(cfg, prepared) as session:
            run_nf_events_presentation(
                session=session,
                skip_first_trs=args.skip_first_trs,
                baseline_trs=baseline_trs,
                n_trials=args.n_trials,
                iti_trs=args.iti_trs,
                cue_trs=args.cue_trs,
                scans_trs=args.scans_trs,
                delay_trs=args.delay_trs,
                feedback_trs=args.feedback_trs,
                score_delay=args.score_delay,
                tr_seconds=prepared["tr"],
                trial_scores_path=trial_scores_path,
                reference_stats=prepared["reference_score_stats"],
            )
    finally:
        _plot_qc(prepared["run_dir"])


if __name__ == "__main__":
    main()