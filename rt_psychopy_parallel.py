#!/usr/bin/env python
import argparse
import csv
import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Queue
from queue import Empty
from pathlib import Path
import multiprocessing as mp
import time
from typing import Optional

import numpy as np

from rt_global_settings import load_regressor_settings
import rs_pca_runtime as pca_rt
from rt_session import (
    PhysioConfig,
    ProcessSpec,
    SessionConfig,
    merge_session_metadata as _merge_session_metadata,  # kept under the old name for callers
    prepare_session,
    start_session,
)
from rt_session import _physio_main as _run_biopac_listener  # noqa: F401  (old name, same function)


log = logging.getLogger(__name__)

# Reference launcher on the session API (docs/session_api.md): this file owns the
# score-plot display and the condition bookkeeping; rt_session owns the pipeline,
# physio and PCA-scorer processes, metadata and shutdown. 's' starts on the
# scanner trigger, 'q' or Escape ends the run cleanly.


@dataclass(frozen=True)
class Condition:
    condition_id: str
    roi: str
    direction: str
    symbol: str


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_conditions(roi_labels: list[str], directions: list[str], symbols: list[str]) -> list[Condition]:
    conditions: list[Condition] = []
    for roi in roi_labels:
        for direction in directions:
            condition_id = f"{roi}_{direction}".lower().replace(" ", "_")
            conditions.append(
                Condition(
                    condition_id=condition_id,
                    roi=roi,
                    direction=direction,
                    symbol="",
                )
            )

    if len(symbols) < len(conditions):
        raise ValueError(
            f"Need at least {len(conditions)} symbols, got {len(symbols)}"
        )

    enriched: list[Condition] = []
    for cond, symbol in zip(conditions, symbols):
        enriched.append(
            Condition(
                condition_id=cond.condition_id,
                roi=cond.roi,
                direction=cond.direction,
                symbol=symbol,
            )
        )
    return enriched


def _shuffle_symbols(symbols: list[str], seed: int | None) -> list[str]:
    rng = random.Random(seed)
    shuffled = symbols[:]
    rng.shuffle(shuffled)
    return shuffled


def _load_or_create_schedule(
    schedule_path: Path,
    conditions: list[Condition],
    seed: int | None = None,
    symbol_seed: int | None = None,
) -> dict:
    if schedule_path.exists():
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)
        existing = {c["condition_id"] for c in schedule.get("conditions", [])}
        expected = {cond.condition_id for cond in conditions}
        if existing == expected and schedule.get("order"):
            return schedule

    rng = random.Random(seed)
    condition_ids = [cond.condition_id for cond in conditions]
    rng.shuffle(condition_ids)
    schedule = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol_seed": symbol_seed,
        "order": condition_ids,
        "conditions": [
            {
                "condition_id": cond.condition_id,
                "roi": cond.roi,
                "direction": cond.direction,
                "symbol": cond.symbol,
            }
            for cond in conditions
        ],
    }
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
    return schedule


def _condition_for_run(schedule: dict, run: str) -> Condition:
    order = schedule["order"]
    condition_lookup = {c["condition_id"]: c for c in schedule["conditions"]}
    try:
        run_idx = int(run) - 1
    except ValueError:
        run_idx = 0
    if run_idx < 0:
        run_idx = 0
    if run_idx >= len(order):
        run_idx = run_idx % len(order)
    condition_id = order[run_idx]
    cond = condition_lookup[condition_id]
    return Condition(
        condition_id=cond["condition_id"],
        roi=cond["roi"],
        direction=cond["direction"],
        symbol=cond["symbol"],
    )


def _write_run_assignment(run_path: Path, condition: Condition, schedule_path: Path) -> None:
    payload = {
        "condition_id": condition.condition_id,
        "roi": condition.roi,
        "direction": condition.direction,
        "symbol": condition.symbol,
        "schedule_path": str(schedule_path),
    }
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "condition_assignment.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_pca_run_assignment(
    run_path: Path,
    condition: pca_rt.PCACondition,
    public_schedule_path: Path,
) -> None:
    public_payload = {
        "condition_id": condition.condition_id,
        "symbol": condition.symbol,
        "schedule_path": str(public_schedule_path),
    }
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "condition_assignment.json", "w", encoding="utf-8") as f:
        json.dump(public_payload, f, indent=2)


def _append_condition_score(csv_path: Path, message: dict, condition: Condition) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "volume_idx",
        "timestamp",
        "score_raw",
        "feedback_score",
        "score_z",
        "raw_component_score",
        "directed_score",
        "condition_id",
        "roi",
        "score_label",
        "pc",
        "direction",
        "symbol",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        blind_condition = bool(message.get("blind_condition", False))
        writer.writerow(
            {
                "volume_idx": message.get("volume_idx"),
                "timestamp": message.get("timestamp"),
                "score_raw": message.get("score_raw"),
                "feedback_score": message.get("feedback_score"),
                "score_z": message.get("score_z"),
                "raw_component_score": message.get("raw_component_score"),
                "directed_score": message.get("directed_score"),
                "condition_id": condition.condition_id,
                "roi": "" if blind_condition else getattr(condition, "roi", ""),
                "score_label": message.get("score_label"),
                "pc": "" if blind_condition else getattr(condition, "pc", ""),
                "direction": "" if blind_condition else getattr(condition, "direction", ""),
                "symbol": condition.symbol,
            }
        )




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


def _run_pipeline_with_settings(cfg: "RTSessionConfig", score_queue: Queue, settings: dict) -> None:
    """Old process target, kept for callers; new code goes through rt_session."""
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings)
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
        "fullscr": True,
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
    win = visual.Window(**window_kwargs)
    _release_mouse_cursor(win)
    return win


def run_psychopy_presentation(
    session,
    max_points: int,
    condition: Condition,
    condition_scores_path: Path,
    max_trs: Optional[int],
) -> None:
    """The score-plot display. `session` is an rt_session.Session (anything with a
    `poll() -> list[ScoreEvent]`); runs in the calling (main) process."""
    from psychopy import core, event, visual

    background_color = [-0.004, -0.004, -0.004]
    foreground_color = "black"
    score_min = 0.0
    score_max = 100.0
    score_span = score_max - score_min

    win = _build_presentation_window(visual, color=background_color)
    title = visual.TextStim(win, text="Real-time Scores", pos=(0, 300), color=foreground_color)
    condition_text = visual.TextStim(
        win,
        text=condition.symbol,
        pos=(0, 200),
        color=foreground_color,
        height=80,
    )
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner trigger ('s')...",
        pos=(0, 0),
        color=foreground_color,
    )

    margins = {"left": 80, "right": 40, "bottom": 80, "top": 80}
    plot_width = win.size[0] - margins["left"] - margins["right"]
    plot_height = win.size[1] - margins["bottom"] - margins["top"]
    origin_x = -win.size[0] / 2 + margins["left"]
    origin_y = -win.size[1] / 2 + margins["bottom"]

    x_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x + plot_width, origin_y),
        lineColor=foreground_color,
    )
    y_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x, origin_y + plot_height),
        lineColor=foreground_color,
    )
    # start as a degenerate 2-point line (valid Nx2)
    score_line = visual.ShapeStim(
        win,
        vertices=[(origin_x, origin_y), (origin_x, origin_y)],
        closeShape=False,
        lineColor=foreground_color,
    )
    y_min_text = visual.TextStim(
        win,
        text="0",
        pos=(origin_x - 30, origin_y),
        color=foreground_color,
        height=24,
    )
    y_max_text = visual.TextStim(
        win,
        text="100",
        pos=(origin_x - 40, origin_y + plot_height),
        color=foreground_color,
        height=24,
    )
    last_score_text = visual.TextStim(win, text="", pos=(0, -300), color=foreground_color)

    scores = deque(maxlen=max_points)
    needs_redraw = True
    reg_ready_seen = False
    seen_vols: set[int] = set()
    acquisition_speed_path = condition_scores_path.parent / "acquisition_speed_rt.csv"

    first_trigger_timestamp: Optional[float] = None
    waiting = True
    while waiting:
        win.fullscr = True
        _release_mouse_cursor(win)
        waiting_text.draw()
        win.flip()
        keys = event.getKeys()
        if "s" in keys:
            first_trigger_timestamp = time.time()
            waiting = False
        if "escape" in keys or "q" in keys:
            win.close()
            return
        core.wait(0.02)

    def update_plot() -> None:
        nonlocal needs_redraw
        if not scores:
            score_line.vertices = [(origin_x, origin_y), (origin_x, origin_y)]
            if reg_ready_seen:
                last_score_text.text = "Waiting for scores..."
            else:
                last_score_text.text = "Waiting for regression..."
            needs_redraw = True
            return

        # If only 1 point, duplicate it so vertices is still Nx2 with N>=2
        data = list(scores)
        if len(data) == 1:
            data = [data[0], data[0]]

        x_step = plot_width / max(1, max_points - 1)

        vertices = []
        for idx, score in enumerate(data):
            x = origin_x + idx * x_step
            score_clipped = min(max(score, score_min), score_max)
            y_norm = (score_clipped - score_min) / score_span
            y = origin_y + y_norm * plot_height
            vertices.append((x, y))

        score_line.vertices = vertices
        last_score_text.text = f"Last score: {scores[-1]:.4f}"
        needs_redraw = True

    while True:
        updated = False
        for ev in session.poll():
            if ev.kind == "error":
                log.warning("Session error event: %s", ev.meta.get("error"))
            vol_idx = int(ev.volume_idx or 0)
            if vol_idx:
                seen_vols.add(vol_idx)
            if first_trigger_timestamp is not None and vol_idx > 0:
                # NOTE: 1.4 s is a hardcoded TR here (pre-existing; the NF-events launcher uses the settings TR)
                estimated_trigger_timestamp = first_trigger_timestamp + ((vol_idx - 1) * 1.4)
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
                            "trigger_to_watchdog_s": (
                                f"{(float(watchdog_timestamp) - estimated_trigger_timestamp):.6f}"
                            ),
                            "watchdog_to_analysis_s": (
                                f"{(float(analysis_timestamp) - float(watchdog_timestamp)):.6f}"
                            ),
                        },
                    )

            # a decoder score (pipeline) or a PCA feedback score (scorer process); a
            # missing reg_ready counts as ready, as before
            has_score = ev.score_raw is not None or "score_raw" in ev.meta
            if has_score and (ev.reg_ready is None or ev.reg_ready):
                reg_ready_seen = True
                scores.append(float(ev.score_raw))
                _append_condition_score(condition_scores_path, ev.to_message(), condition)
                updated = True
            elif has_score:
                updated = True

        if max_trs is not None and len(seen_vols) >= max_trs:
            break

        if updated or needs_redraw:
            update_plot()
            win.clearBuffer()
            title.draw()
            condition_text.draw()
            x_axis.draw()
            y_axis.draw()
            y_min_text.draw()
            y_max_text.draw()
            score_line.draw()
            last_score_text.draw()
            win.flip()
            needs_redraw = False

        keys = event.getKeys()
        if "escape" in keys or "q" in keys:
            break

        core.wait(0.02)

    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)  # <-- IMPORTANT: before CUDA touches anything
    parser = argparse.ArgumentParser(
        description=(
            "Run rt_pipeline in parallel with a PsychoPy visualization of the last 20 scores."
        )
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--ap-block",
        default=None,
        type=int,
        help="AP block to use for b0",
    )
    parser.add_argument(
        "--pa-block",
        default=None,
        type=int,
        help="PA block to use for b0",
    )
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
        "--max-points",
        type=int,
        default=20,
        help="Number of most recent scores to plot.",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default for decoder scoring.",
    )
    parser.add_argument(
        "--decoder-roi-txt",
        required=False,
        default=None,
        help=(
            "Optional ROI_DECODER-style text file for decoder scoring. "
            "Ignored when using PCA scoring."
        ),
    )
    parser.add_argument(
        "--score-source",
        choices=["decoder", "pca"],
        default=None,
        help=(
            "Score backend for feedback. Defaults to decoder unless --pca-mode "
            "is supplied; --pca-mode is kept as an alias for --score-source=pca."
        ),
    )
    parser.add_argument(
        "--roi-labels",
        default="LPFC,Sensorimotor,EVC",
        help="Comma-separated ROI labels for condition mapping.",
    )
    parser.add_argument(
        "--direction-labels",
        default="up,down",
        help="Comma-separated regulation directions.",
    )
    parser.add_argument(
        "--condition-symbols",
        default="A,B,C,D,E,F",
        help="Comma-separated symbols to display for each condition.",
    )
    parser.add_argument(
        "--condition-seed",
        type=int,
        default=None,
        help="Optional random seed for condition order (per day)."
    )
    parser.add_argument(
        "--symbol-seed",
        type=int,
        default=None,
        help="Optional random seed for symbol assignment (per subject).",
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP/file input.",
    )
    parser.add_argument(
        "--biopac-host",
        default="0.0.0.0",
        help="Host to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-port",
        type=int,
        default=15000,
        help="Port to bind BIOPAC receiver.",
    )
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
        "--max-trs",
        type=int,
        default=None,
        help="Stop the run after this many TRs (or press ESC).",

    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=None,
        help="Stop after this many minutes, using TR from settings. Ignored if --max-trs is set.",
    )
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--pca-mode",
        action="store_true",
        help=(
            "Deprecated alias for --score-source=pca: disable decoder scoring "
            "and drive feedback from PCA outputs."
        ),
    )
    parser.add_argument(
        "--pca-day",
        default=None,
        help="Day/session whose PCA decoder bundles should be used. Defaults to --day.",
    )
    parser.add_argument(
        "--pca-run",
        default=None,
        help="Run whose PCA decoder bundles should be used. Defaults to --run.",
    )
    parser.add_argument(
        "--pca-root",
        type=Path,
        default=None,
        help="Explicit PCA decoder root containing ROI folders.",
    )
    parser.add_argument(
        "--pca-input",
        choices=["auto", "mc", "reg", "t1", "smooth", "masked"],
        default="t1",
        help="PCA decoder/output mode folder name.",
    )
    parser.add_argument(
        "--pca-space",
        choices=["epi", "t1", "mni"],
        default="t1",
        help="rt_pipeline output space to create for PCA feedback.",
    )
    parser.add_argument(
        "--pca-reference-image",
        type=Path,
        default=None,
        help="Explicit reference image/grid for PCA-space transforms.",
    )
    parser.add_argument(
        "--pca-reference-resolution",
        choices=["epi", "t1"],
        default="epi",
        help="Default PCA T1 reference resolution when --pca-reference-image is not provided.",
    )
    parser.add_argument(
        "--pca-reference-scores",
        type=Path,
        default=None,
        help="Daily RS raw PCA scores CSV from rs_pca_score_all_rois.py.",
    )
    parser.add_argument(
        "--pca-reference-stats",
        type=Path,
        default=None,
        help="Deprecated explicit daily RS reference stats JSON.",
    )
    parser.add_argument(
        "--pca-reference-day",
        default=None,
        help="Day/session of the daily RS stats run. Defaults to --day when --pca-reference-run is set.",
    )
    parser.add_argument(
        "--pca-reference-run",
        default=None,
        help="Run whose daily RS PCA scores should be used for feedback.",
    )
    parser.add_argument(
        "--pca-volume-kind",
        choices=["reg", "mc", "unwarped", "t1", "mni", "smooth", "masked"],
        default=None,
        help="Realtime volume folder to score with PCA.",
    )
    parser.add_argument(
        "--pca-target-pc",
        default=None,
        help="Legacy single PC to modulate when --pca-score-metric is projection/cosine.",
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["zscore", "demean", "none"],
        default="zscore",
        help="Voxel normalization before PCA projection.",
    )
    parser.add_argument(
        "--pca-score-metric",
        choices=sorted(pca_rt.ALL_SCORE_METRICS),
        default="subspace_cosine",
        help="PCA scalar score. Default is cosine closeness to the top-PC subspace.",
    )
    parser.add_argument(
        "--pca-top-pcs",
        default="auto",
        help=(
            "Top PCs for multi-PC metrics: auto uses a cumulative explained-variance "
            "target; also accepts integer count, var:0.01, or all."
        ),
    )
    parser.add_argument(
        "--pca-top-pc-variance",
        type=float,
        default=0.10,
        help=(
            "Cumulative explained-variance threshold used only when "
            "--pca-top-pcs=auto. If the saved PCs do not reach it, auto uses all saved PCs."
        ),
    )
    parser.add_argument(
        "--pca-poll-interval",
        type=float,
        default=0.05,
        help="Seconds between checks for newly processed realtime volumes.",
    )
    parser.add_argument(
        "--condition-index",
        type=int,
        default=None,
        help="1-based condition index for this NF run. Defaults to numeric --run.",
    )
    parser.add_argument(
        "--condition-private-key",
        type=Path,
        default=None,
        help="Private A-F to ROI/direction mapping. Keep hidden from subject/experimenter.",
    )
    parser.add_argument(
        "--condition-public-schedule",
        type=Path,
        default=None,
        help="Blinded A-F schedule shown/recorded for the run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--rt-max-scan-length",
        type=int,
        default=None,
        help="Maximum TR count preallocated by RTPSpy regression.",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    args = parser.parse_args()
    from rt_pipeline import REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))
    max_trs = args.max_trs
    if max_trs is None and args.duration_min is not None:
        max_trs = int(round((float(args.duration_min) * 60.0) / float(REGRESSOR_SETTINGS.TR)))
    base_data = Path(args.base_data)
    subject_root = base_data / f"sub-{args.sub}"
    pca_reference_image = None
    score_source = args.score_source or ("pca" if args.pca_mode else "decoder")
    use_pca = score_source == "pca"
    if args.pca_mode and args.score_source == "decoder":
        raise ValueError("--pca-mode conflicts with --score-source=decoder")
    if use_pca and args.decoder_roi_txt:
        log.info("PCA score source enabled: ignoring --decoder-roi-txt=%s", args.decoder_roi_txt)

    cfg_decoder_template = (
        None
        if use_pca
        else Path(args.decoder_template) if args.decoder_template else None
    )
    cfg = SessionConfig(
        sub=args.sub,
        day=args.day,
        run=args.run,
        base_data=base_data,
        incoming_root=Path(args.incoming_root),
        settings_file=Path(args.settings_file) if args.settings_file else None,
        decoder_template=cfg_decoder_template,
        decoder_roi_txt=None if use_pca else Path(args.decoder_roi_txt) if args.decoder_roi_txt else None,
        reference_score_run=None if use_pca else args.reference_score_run,
        enable_scoring=not use_pca,
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
        queue_size=100,
    )
    # the run directory (created by the pipeline config, as before)
    from rt_session import build_rt_config

    run_dir = build_rt_config(cfg).rt_work_dir

    if use_pca and args.reference_score_run:
        log.info("PCA score source enabled: ignoring --rs=%s", args.reference_score_run)

    if args.rt_max_scan_length is not None:
        cfg.settings_overrides["rt_max_scan_length"] = max(1, int(args.rt_max_scan_length))
    if use_pca:
        cfg.settings_overrides["analysis_space"] = args.pca_space
    pca_volume_kind = args.pca_volume_kind
    if use_pca and pca_volume_kind is None:
        pca_volume_kind = "reg" if args.pca_space == "epi" else args.pca_space
    pca_score_label = None
    if use_pca:
        pca_score_label = pca_rt.resolve_score_label(
            score_metric=args.pca_score_metric,
            target_pc=args.pca_target_pc,
            top_pcs=args.pca_top_pcs,
            top_pc_variance=args.pca_top_pc_variance,
        )

    roi_labels = _parse_csv_list(args.roi_labels)
    direction_labels = _parse_csv_list(args.direction_labels)
    symbols = _parse_csv_list(args.condition_symbols)
    symbol_seed = args.symbol_seed
    if symbol_seed is None and not use_pca:
        try:
            symbol_seed = int(args.sub)
        except ValueError:
            symbol_seed = None
    pca_day = None
    pca_run = None
    pca_root = None
    if use_pca:
        default_private_key, default_public_schedule = pca_rt.default_condition_paths(
            subject_root
        )
        private_key_path = args.condition_private_key or default_private_key
        public_schedule_path = args.condition_public_schedule or default_public_schedule
        schedule = pca_rt.load_or_create_condition_schedule(
            private_path=private_key_path,
            public_path=public_schedule_path,
            roi_labels=roi_labels,
            direction_labels=direction_labels,
            symbols=symbols,
            target_pc=pca_score_label,
            condition_seed=args.condition_seed,
            symbol_seed=symbol_seed,
        )
        if args.condition_index is not None:
            condition_index = args.condition_index
        else:
            try:
                condition_index = int(args.run)
            except ValueError:
                condition_index = 1
        condition = pca_rt.condition_for_index(
            schedule,
            condition_index,
            score_label=pca_score_label,
        )
        _write_pca_run_assignment(
            run_dir,
            condition,
            public_schedule_path,
        )
        pca_run = args.pca_run or args.run
        pca_day = args.pca_day or args.day
        pca_run_dir = pca_rt.build_run_dir(base_data, args.sub, pca_day, pca_run)
        pca_day_dir = pca_run_dir.parent.parent
        pca_root = args.pca_root or pca_rt.build_pca_root(
            pca_day_dir,
            pca_run_dir.name,
            args.pca_input,
        )
        if not pca_root.exists():
            raise FileNotFoundError(f"PCA decoder root not found: {pca_root}")
        if args.pca_space in {"t1", "mni"}:
            decoder_trans_dir = subject_root / pca_day / "func" / "trans"
            pca_reference_image = pca_rt.resolve_pca_transform_reference(
                subject_root=subject_root,
                decoder_trans_dir=decoder_trans_dir,
                pca_root=pca_root,
                explicit_path=args.pca_reference_image,
                resolution=args.pca_reference_resolution,
                truncate_to_epi_fov=bool(REGRESSOR_SETTINGS.truncate_t1_to_epi_fov),
                padding_vox=int(REGRESSOR_SETTINGS.truncate_t1_padding_vox),
            )
            cfg.decoder_template = pca_reference_image
            log.info("Using PCA transform reference: %s", pca_reference_image)
    else:
        symbols = _shuffle_symbols(symbols, symbol_seed)
        conditions = _build_conditions(roi_labels, direction_labels, symbols)
        schedule_path = subject_root / "condition_schedule.json"
        schedule = _load_or_create_schedule(
            schedule_path,
            conditions,
            seed=args.condition_seed,
            symbol_seed=symbol_seed,
        )
        condition = _condition_for_run(schedule, args.run)
        _write_run_assignment(run_dir, condition, schedule_path)
    condition_scores_path = run_dir / "scores_with_conditions.csv"
    reference_scores_path = args.pca_reference_scores
    reference_stats_path = args.pca_reference_stats
    reference_stats = {}
    reference_stats_day = args.pca_reference_day
    reference_stats_run = args.pca_reference_run
    if use_pca:
        if reference_scores_path is not None and reference_stats_path is not None:
            raise ValueError("Use either --pca-reference-scores or --pca-reference-stats, not both.")
        if reference_stats_path is not None:
            if not reference_stats_path.exists():
                raise FileNotFoundError(f"PCA daily RS reference stats not found: {reference_stats_path}")
            reference_stats = pca_rt.load_reference_stats(reference_stats_path)
        elif reference_stats_run is not None:
            reference_stats_day = reference_stats_day or args.day
            reference_scores_path = pca_rt.build_reference_scores_path(
                base_data,
                args.sub,
                reference_stats_day,
                reference_stats_run,
                args.pca_input,
            )
        elif reference_stats_day is not None and reference_scores_path is None:
            raise ValueError(
                "--pca-reference-day requires --pca-reference-run unless "
                "--pca-reference-scores is provided explicitly."
            )
        if reference_scores_path is not None:
            if not reference_scores_path.exists():
                raise FileNotFoundError(
                    f"PCA daily RS reference scores not found: {reference_scores_path}. "
                    "Run rs_pca_score_all_rois.py for that daily RS run first."
                )
            reference_stats = pca_rt.load_reference_stats_from_score_csv(
                reference_scores_path,
                [condition.score_column],
            )

    condition_payload = {
        "condition_id": condition.condition_id,
        "symbol": condition.symbol,
    }
    if not use_pca:
        condition_payload.update(
            {
                "roi": condition.roi,
                "direction": condition.direction,
            }
        )

    cfg.metadata = (
        {
            "psychopy": {
                "max_points": args.max_points,
                "max_trs": max_trs,
                "ap_block": args.ap_block,
                "pa_block": args.pa_block,
                "fmap_dir": str(build_rt_config(cfg).fmap_dir),
                "duration_min": args.duration_min,
                "roi_labels": roi_labels,
                "direction_labels": direction_labels,
                "condition_symbols": symbols,
                "condition_seed": args.condition_seed,
                "symbol_seed": symbol_seed,
                "condition_schedule": str(
                    public_schedule_path if use_pca else schedule_path
                ),
                "condition_assignment": condition_payload,
                "score_source": score_source,
                "pca_mode": use_pca,
                "decoder_template": str(cfg.decoder_template) if cfg.decoder_template else None,
                "decoder_roi_txt": str(cfg.decoder_roi_txt) if cfg.decoder_roi_txt else None,
                "pca_decoder": {
                    "day": pca_day,
                    "run": pca_run,
                    "root": str(pca_root) if pca_root else None,
                    "input": args.pca_input,
                    "space": args.pca_space,
                    "reference_image": (
                        str(pca_reference_image)
                        if pca_reference_image
                        else None
                    ),
                    "reference_resolution": args.pca_reference_resolution,
                    "volume_kind": pca_volume_kind,
                    "target_pc": args.pca_target_pc,
                    "score_label": pca_score_label,
                    "top_pcs": args.pca_top_pcs,
                    "top_pc_variance": args.pca_top_pc_variance,
                    "normalization": args.pca_normalization,
                    "score_metric": args.pca_score_metric,
                    "reference_scores": str(reference_scores_path) if reference_scores_path else None,
                    "reference_stats": str(reference_stats_path) if reference_stats_path else None,
                    "reference_stats_day": reference_stats_day,
                    "reference_stats_run": reference_stats_run,
                }
                if use_pca
                else None,
            }
        }
    )

    if use_pca:
        # the PCA scorer attaches through the generic extra-process hook: the session
        # gives it the score queue and a stop event, starts it after the pipeline and
        # stops it (5 s grace, then terminate) before the pipeline
        cfg.extra_processes.append(
            ProcessSpec(
                "pca_scorer",
                pca_rt.run_realtime_pca_scorer,
                kwargs={
                    "run_dir": run_dir,
                    "pca_root": pca_root,
                    "condition": condition,
                    "reference_stats_path": reference_stats_path,
                    "reference_stats": reference_stats,
                    "volume_kind": pca_volume_kind,
                    "normalization": args.pca_normalization,
                    "score_metric": args.pca_score_metric,
                    "target_pc": args.pca_target_pc,
                    "top_pcs": args.pca_top_pcs,
                    "top_pc_variance": args.pca_top_pc_variance,
                    "max_trs": max_trs,
                    "poll_interval": args.pca_poll_interval,
                    "qc_plot_path": run_dir / "qc_pca_scores_motion.png",
                    "qc_update_every": 5,
                },
                pass_queue="score_queue",
                pass_stop_event="stop_event",
            )
        )

    prepared = prepare_session(cfg)
    try:
        with start_session(cfg, prepared) as session:
            run_psychopy_presentation(
                session,
                args.max_points,
                condition,
                condition_scores_path,
                max_trs,
            )
    finally:
        if use_pca:
            pca_rt.plot_pca_scores_motion(
                run_dir=run_dir,
                scores_csv=run_dir / "pca_realtime_scores.csv",
                out_png=run_dir / "qc_pca_scores_motion.png",
                score_columns=[
                    "raw_component_score",
                    "directed_score",
                    "score_z",
                    "feedback_score",
                ],
                title="PCA neurofeedback scores and motion",
            )
        else:
            _plot_qc(run_dir)


if __name__ == "__main__":
    main()
