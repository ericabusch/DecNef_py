#!/usr/bin/env python
"""Example launcher: the event-based neurofeedback paradigm on the decnef session API.

    python examples/nf_events_launcher.py --config decnef.yaml --sub 00086 --day 3 --run 4 \
        --n-trials 20 --baseline-trs 20 --iti-trs 3 --cue-trs 4 --scans-trs 3 --delay-trs 3 --feedback-trs 3
    python examples/nf_events_launcher.py ... --simulate scores            # dry run: fake scores at the TR, no tools
    python examples/nf_events_launcher.py ... --simulate scanner --sim-out /tmp/sim   # synthetic DICOMs through the real pipeline

What this file owns: the trial structure (baseline, then n_trials x
iti/cue/scans/delay/feedback), the PsychoPy display (fixation during cue, a
feedback circle sized by the trial's percentile score), and trial_scores.csv.
What decnef owns: everything else (pipeline, physio, metadata, shutdown).
Keys: 's' starts on the scanner trigger, 'q' or Escape ends the run cleanly.

The trial-stage schedule is passed to decnef as `settings.stages`, so
scores.csv and every ScoreEvent carry the stage label, and the pipeline's
`skip_first_trs` is the one setting that defines which TRs are skipped.
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

from decnef.config import StagesConfig, load_settings
from decnef.display import STOP_KEYS, AcquisitionSpeedLog, build_window, pressed, wait_for_start
from decnef.events import StageSchedule
from decnef.session import SessionConfig, start_session
from decnef.simulate import SimulationConfig

log = logging.getLogger("nf_events")
STAGES = ("iti", "cue", "scans", "delay", "feedback")
BACKGROUND = [-0.004, -0.004, -0.004]


def append_trial_score(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "score_window_start_tr", "score_window_end_tr", "n_scores_used",
                                          "trial_score_raw", "trial_score", "score_missing"])
        if not exists:
            w.writeheader()
        w.writerow(row)


def score_to_percentile(score_raw: float, reference_stats: Optional[dict]) -> float:
    if not np.isfinite(score_raw):
        return float("nan")
    if reference_stats is None:
        return float(np.clip(score_raw, 0.0, 100.0))
    z = (score_raw - reference_stats["mean"]) / reference_stats["std"]
    return float(np.clip(norm.cdf(z) * 100.0, 0.0, 100.0))


def run_display(session, schedule: StageSchedule, score_delay: int, tr_seconds: float, trial_scores_path: Path,
                reference_stats: Optional[dict]) -> None:
    """The paradigm. `session` provides `poll() -> list[ScoreEvent]`; runs in the main process."""
    from psychopy import core, event, visual

    win = build_window(visual, color=BACKGROUND)
    waiting_text = visual.TextStim(win, text="Waiting for scanner trigger ('s')...", color="black", height=36)
    fixation = visual.ShapeStim(win, vertices="cross", size=[0.1 * win.size[1]] * 2, lineWidth=1.0, fillColor="white", lineColor="white")
    feedback_circle = visual.Circle(win, radius=1, lineWidth=10, fillColor=[-0.5, 0.7, -0.5], lineColor=[-0.5, 0.7, -0.5])
    max_radius = 0.60 * min(win.size)
    reference_circle = visual.Circle(win, radius=max_radius, fillColor=BACKGROUND, lineColor=[0.85, 0.85, 0.85], lineWidth=10)
    missing_text = visual.TextStim(win, text="", color=[0.9, 0.2, 0.2], pos=(0, -int(0.35 * min(win.size))), height=30)

    skip = schedule.skip_first_trs
    max_seen_vol = 0
    score_by_exp_tr: dict[int, float] = {}
    speed_log: Optional[AcquisitionSpeedLog] = None

    def drain() -> None:
        nonlocal max_seen_vol
        for ev in session.poll():
            if ev.kind == "error":
                log.warning("session error: %s", ev.meta.get("error"))
                continue
            if ev.volume_idx is None:
                continue
            max_seen_vol = max(max_seen_vol, ev.volume_idx)
            if speed_log is not None:
                speed_log.record(ev)
            exp_tr = ev.volume_idx - skip
            if ev.kind == "score" and ev.reg_ready and ev.score_raw is not None and exp_tr > schedule.baseline_trs:
                score_by_exp_tr[exp_tr] = float(ev.score_raw)

    trigger = wait_for_start(win, event, core, draw=waiting_text.draw, poll=drain)
    if trigger is None:
        win.close()
        return
    speed_log = AcquisitionSpeedLog(trial_scores_path.parent / "acquisition_speed_rt.csv", tr_seconds, trigger)

    windows = schedule.windows()
    cue_start = {w["cycle"]: w["start_tr"] for w in windows if w["stage"] == "cue"}
    scans_end = {w["cycle"]: w["end_tr"] for w in windows if w["stage"] == "scans"}
    trial_scores: dict[int, float] = {}
    try:
        for w in windows:
            trial, stage, stage_end = w["cycle"], w["stage"], w["end_tr"]
            while True:
                drain()
                current_exp_tr = max(0, max_seen_vol - skip)
                if stage == "cue":
                    fixation.draw()
                elif stage == "feedback" and trial > 0:
                    if trial not in trial_scores:
                        s_start = min(cue_start[trial] + max(0, score_delay), scans_end[trial])
                        s_end = scans_end[trial]
                        used = [score_by_exp_tr[tr] for tr in range(s_start, s_end + 1) if tr in score_by_exp_tr]
                        raw = float(np.mean(used)) if used else float("nan")
                        trial_scores[trial] = score_to_percentile(raw, reference_stats)
                        append_trial_score(trial_scores_path, {"trial": trial, "score_window_start_tr": s_start,
                                                               "score_window_end_tr": s_end, "n_scores_used": len(used),
                                                               "trial_score_raw": raw, "trial_score": trial_scores[trial],
                                                               "score_missing": int(not used)})
                        log.info("trial %02d: TRs [%d, %d], %d samples, raw=%.4f pct=%.1f", trial, s_start, s_end, len(used), raw, trial_scores[trial])
                    t = trial_scores[trial]
                    feedback_circle.setRadius(0.0 if np.isnan(t) else max_radius * float(np.clip(t, 0, 100)) / 100.0, log=False)
                    reference_circle.draw()
                    feedback_circle.draw()
                    missing_text.text = "Score not received in time" if np.isnan(t) else ""
                    if missing_text.text:
                        missing_text.draw()
                win.flip()
                if pressed(event.getKeys(), STOP_KEYS):
                    return
                if current_exp_tr >= stage_end:
                    break
                core.wait(0.02)
    finally:
        win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--sub", required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--incoming", type=Path, default=None)
    p.add_argument("--ap-block", type=int, default=None)
    p.add_argument("--pa-block", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--baseline-trs", type=int, default=None, help="default: pipeline.voxel_norm_ref_volumes")
    for stage, default in zip(STAGES, (3, 4, 3, 3, 3)):
        p.add_argument(f"--{stage}-trs", type=int, default=default)
    p.add_argument("--score-delay", type=int, default=3, help="TRs after cue onset before the score window starts")
    p.add_argument("--simulate", choices=["scores", "scanner"], default=None,
                   help="dry run without a scanner: 'scores' = fake pipeline events, 'scanner' = synthetic DICOMs through the real pipeline")
    p.add_argument("--sim-out", type=Path, default=None, help="simulation directory (default: fresh temp dir; reused if it exists)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    settings = load_settings(a.config, a.overrides, **({"paths.incoming_root": str(a.incoming)} if a.incoming else {}))
    baseline = a.baseline_trs if a.baseline_trs is not None else settings.pipeline.voxel_norm_ref_volumes
    if baseline < 0 or any(getattr(a, f"{s}_trs") < 0 for s in STAGES) or a.n_trials < 0:
        raise SystemExit("stage lengths, baseline and n_trials must be >= 0")
    # keep the pipeline's normalisation warm-up in sync with the baseline
    settings.pipeline.voxel_norm_ref_volumes = max(1, baseline)
    settings.stages = StagesConfig(baseline_trs=baseline, cycle=[[s, getattr(a, f"{s}_trs")] for s in STAGES], n_cycles=a.n_trials)
    schedule = StageSchedule.from_config(settings.stages, settings.pipeline.skip_first_trs)

    cfg = SessionConfig(settings, a.sub, a.day, a.run, ap_block=a.ap_block, pa_block=a.pa_block,
                        metadata={"launcher": {"script": Path(__file__).name, "n_trials": a.n_trials, "baseline_trs": baseline,
                                               **{f"{s}_trs": getattr(a, f"{s}_trs") for s in STAGES}, "score_delay": a.score_delay}})
    reference_stats = None
    if settings.decoder.reference_score_run:
        from decnef.outputs import load_reference_score_stats

        reference_stats = load_reference_score_stats(cfg.layout.other_run(settings.decoder.reference_score_run).run_dir,
                                                     settings.decoder.reference_score_run)
    # the only difference between a real run and a dry run: the `simulate` argument
    simulate = None
    if a.simulate:
        n_volumes = settings.pipeline.skip_first_trs + baseline + a.n_trials * sum(getattr(a, f"{s}_trs") for s in STAGES) + 2
        simulate = SimulationConfig(level=a.simulate, n_volumes=n_volumes, out_dir=a.sim_out, signal_stages=("scans",))
    with start_session(cfg, simulate=simulate) as session:
        if simulate is not None:
            log.info("simulation outputs: %s", session.simulation.out_dir)
        # session.run_dir is the real run folder, or the simulation's run folder when simulating
        run_display(session, schedule, a.score_delay, settings.pipeline.TR, session.run_dir / "trial_scores.csv", reference_stats)


if __name__ == "__main__":
    main()
