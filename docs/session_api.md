# Session API

`rt_session.py` separates two jobs that the `rt_*_parallel.py` launchers used to do together: starting the real-time machinery (pipeline process, BIOPAC listener, extra scorers, metadata, shutdown) and running one study's display. A launcher now owns only the display and the trial structure; it talks to the machinery through `SessionConfig`, `ScoreEvent` and `session.stop()`. `rt_nf_events_parallel.py` and `rt_psychopy_parallel.py` are the reference launchers.

```python
from rt_session import PhysioConfig, ProcessSpec, SessionConfig, StageSchedule, start_session

cfg = SessionConfig(
    sub="00086", day="3", run="4",
    base_data=Path("/SSD2/DecNef_py/data"), incoming_root=Path("/path/to/incoming"),
    settings_file=Path("my_settings.json"),
    settings_overrides={"voxel_norm_ref_volumes": 20},
    decoder_template=Path("decoders/my_decoder.nii"), reference_score_run="3",
    physio=PhysioConfig(enable=True, mode="file", listener=True),
    stages=StageSchedule(skip_first_trs=0, baseline_trs=20, n_cycles=20,
                         cycle=[("iti", 3), ("cue", 4), ("scans", 3), ("delay", 3), ("feedback", 3)]),
    metadata={"psychopy": {...}},            # anything the launcher wants recorded
    extra_processes=[ProcessSpec("pca_scorer", rs_pca_runtime.run_realtime_pca_scorer, kwargs={...})],
)
with start_session(cfg) as session:          # pipeline (+ physio, extras) start here, as spawned children
    for event in session.scores(timeout=0.02):   # the display runs here, in the main process
        if event.kind == "score" and event.reg_ready:
            ...
        if user_pressed_q:
            break
# leaving the block stops every child in order, also on Ctrl-C, a display exception or a pipeline error
```

`prepare_session(cfg)` does everything up to (not including) starting processes and returns `run_dir`, `settings`, `rt_cfg`, `tr`; pass its result to `start_session(cfg, prepared)` when the launcher needs to write files into the run directory first.

## Process model

```text
launcher (main process: display, session.scores())
├── pipeline   rt_session._pipeline_main -> rt_pipeline.run_rt_pipeline(cfg, score_queue, stop_event)
│   └── (optional) QC volume viewer, daemon grandchild
├── physio     rt_session._physio_main -> biopac_rt.BiopacRetroTSReceiver   (PhysioConfig.listener)
└── extras     one spawned process per ProcessSpec, started after the pipeline
```

All children use the `spawn` start method. `stop()` runs in this order and is idempotent: extras (stop event -> join `stop_timeout_s` -> terminate -> kill), then the pipeline (stop event -> join `SessionConfig.stop_timeout_s` -> terminate -> kill), then physio. While waiting it drains the score queue so a child blocked on a full queue can exit. The pipeline exits its poll loop when the stop event is set, so its own cleanup (`DICOMHandler.stop`: BIOPAC run_end, executor shutdown, QC viewer stop) runs; `terminate()` is only the fallback. `stop()` is also registered with `atexit`.

## ScoreEvent

Processes still exchange plain dicts on the score queue; `ScoreEvent.from_message` builds the event on the consumer side, so unported launchers and `rs_pca_runtime` are unaffected.

| field | type | meaning |
|---|---|---|
| `kind` | str | `tick` (volume processed, no score), `score` (decoder score), `scorer` (message from an extra process), `error`, `end` (pipeline exited; the last event) |
| `volume_idx` | int | DICOM scan number, 1-based |
| `timestamp` | float | pipeline: time the score was written; scorers: their own |
| `watchdog_timestamp` | float | pipeline only: start of DICOM conversion |
| `analysis_timestamp` | float | pipeline only: right after scoring |
| `score_raw` | float | pipeline: `dot(voxels, weights) + bias`; PCA scorer: its 0–100 feedback value |
| `score_z` | float | z against the reference run's stats (`--rs`); absent -> None |
| `score_original` | float | score on the un-regressed volume (`enable_original_score`) |
| `reg_ready` | bool | RTPSpy regression warm-up passed; None when the message had no key (the old displays defaulted it differently: NF-events False, others True) |
| `stage` | str | trial-stage label (`event_type`), see below |
| `source` | str | `pipeline`, the `ProcessSpec.name`, or `session` for `end`/`error` |
| `meta` | dict | every other key verbatim (PCA fields `condition_id`, `symbol`, `score_label`, `raw_component_score`, `directed_score`, `feedback_score`, `blind_condition`; `error`; `exit_code`) |

Wire dicts and how they map:

| sender | dict | kind |
|---|---|---|
| `rt_pipeline.process_volume`, scoring on | `kind="score", volume_idx, timestamp, analysis_timestamp, watchdog_timestamp, score_raw, score_original, reg_ready, event_type[, score_z]` | `score` |
| `rt_pipeline.process_volume`, scoring off | `kind="tick", volume_idx, watchdog_timestamp` | `tick` |
| `rs_pca_runtime.run_realtime_pca_scorer` | `volume_idx, timestamp, condition_id, symbol, score_label, raw_component_score, directed_score, score_z, feedback_score, score_raw, reg_ready=True, blind_condition=True` | `scorer` (no `kind` key -> attributed to the first extra process) |
| same, on failure | `volume_idx, timestamp, reg_ready=False, error` | `error` |
| session | — | `end` (exit code 0) or `error` (`meta.exit_code`, `meta.error`); `scores()` raises `SessionError` after an `error` from the session |

Messages are dropped by the sender when the queue is full (`queue_size`, default 200); the CSVs on disk are complete regardless. In PCA mode the same `volume_idx` arrives twice (a `tick` from the pipeline and a `scorer` event), in either order.

## Stages schema

`StageSchedule` is written to `session_metadata.json` under `"stages"` before the pipeline starts; `rt_pipeline.ScoreEventTracker` reads it and labels every volume in `scores.csv` (`event_type`) and in `ScoreEvent.stage`.

```json
"stages": {
  "version": 1,
  "skip_first_trs": 0,            "pre_label": "background",
  "baseline_trs": 20,             "baseline_label": "background",
  "cycle": [{"name": "iti", "trs": 3}, {"name": "cue", "trs": 4}, {"name": "scans", "trs": 3},
            {"name": "delay", "trs": 3}, {"name": "feedback", "trs": 3}],
  "n_cycles": 20,                 "post_label": "post_task"
}
```

Volume `v` (1-based): `pre_label` while `v <= skip_first_trs`; `baseline_label` for the next `baseline_trs` volumes; then the cycle repeats `n_cycles` times (`null` = forever); `post_label` afterwards. An empty cycle labels only the pre/baseline volumes. `rt_session.label_for_volume` is the reference implementation.

Backward compatibility: without a `stages` key the tracker behaves exactly as before — `fixation_display` (rest runs: `rest` after the baseline) and `psychopy.script == "rt_nf_events_parallel.py"` (the NF-events cycle). The ported launchers write both the old key and `stages`; `tests/test_score_event_tracker.py` checks the labels are identical.

## Extra scorers

Anything that needs the score queue and a stop signal attaches with a `ProcessSpec(name, target, kwargs, pass_queue="score_queue", pass_stop_event="stop_event", stop_timeout_s=5.0)`. The session adds the queue and an `Event` to `kwargs` under those names, starts the process after the pipeline, and stops it before the pipeline. The resting-state PCA scorer is `ProcessSpec("pca_scorer", rs_pca_runtime.run_realtime_pca_scorer, kwargs={run_dir, pca_root, condition, ...})` in `rt_psychopy_parallel.py`.

## Decoder file formats (`decoder_score.DecoderScorer`)

`DecoderScorer(decoder_path, roi_txt=None)`:

- `roi_txt` (if it exists): text file; line 1 `nx ny nz`, line 2 `N`, then `N` lines `x y z weight template_value` (1-based voxel indices), then a final line `0 0 0 bias 0` whose 4th column is the bias.
- otherwise `decoder_path`: any nibabel-readable image (`.nii`, `.nii.gz`); weights are the non-zero, non-NaN voxels; bias 0. The decoder must be on the grid of the scored volume: for `analysis_space` `t1`/`mni` it is also used as the resampling reference, for `epi` it must already be on the EPI grid.

`score_from_array(volume)` returns one float, `dot(volume[mask], weights) + bias`. The PCA path uses its own `.npz`/`.npy` bundles (`rs_pca_runtime.load_decoder_artifacts`).

## Run output files

Run directory: `base_data/sub-<sub>/<day>/func/<run>`.

| file | columns / content |
|---|---|
| `scores.csv` | `volume_idx, timestamp, score_raw, score_original, reg_ready, event_type` |
| `scores_z.csv` | `volume_idx, timestamp, score_raw, score_z, ref_run, ref_mean, ref_std, ref_n, ref_used_reg_ready, reg_ready` (only with `--rs`) |
| `motion_rt.1D` | 6 columns (roll pitch yaw dS dL dP), no header, commit order |
| `fd_rt.csv` | `volume_idx, fd` (NaN while `volume_idx <= skip_first_trs`) |
| `dvars_rt.csv` | `volume_idx, fd, dvars, robust_z` (DVARS censoring on) |
| `regressors_rt.csv` | `volume_idx, <regressor names>` |
| `regression_status_rt.csv` | `volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, reg_ready, biopac_missing` |
| `biopac_timelag.csv`, `biopac_regressors_rx.csv` | physio timing / received regressors |
| `session_metadata.json` | launcher keys (`psychopy`, `stages`, ...) merged with the pipeline's (`subject`, `day`, `run`, `decoder_template`, `regression{...}`, `biopac{...}`, `tr`, ...) |
| `preprocessing_order.json` | provenance; a rerun into the same folder with different settings is rejected |
| `raw/ unwarped/ mc/ reg/ t1/ mni/ smooth/ masked/` | per-volume NIfTIs `vol_<idx:05d>_<stage>.nii` |
| launcher-owned | `trial_scores.csv`, `acquisition_speed_rt.csv`, `scores_with_conditions.csv`, `pca_realtime_scores.csv`, `qc_*.png` |

Timestamp columns are wall-clock; every other column is deterministic for a given input, device and torch build.

## Replay test (exact-match check)

`replay_run.py` copies a recorded run's DICOMs into a temporary incoming folder in scan order at the TR (staged on the same filesystem and moved in with `os.replace`, so the watcher sees complete files), runs the pipeline headless, waits for the last volume, and copies the score CSVs to an output folder. `compare_scores.py` compares two such folders column by column, ignoring the timestamp columns. Run on the lab machine (needs `dcm2niix`, the offline preparation outputs for that subject/day, and the same GPU for both runs); each replay needs a fresh run id.

```bash
# reference, on the pre-refactor code (branch cleanup/01-packaging)
python replay_run.py --dicom-dir /path/to/recorded/run --run <RUN> --tr 1.4 --driver legacy \
    --sub <SUB> --day <DAY> --base-data /SSD2/DecNef_py/data --replay-run 901 --out /tmp/replay_ref \
    --decoder-template <decoder> --settings-file <settings.json>
# after the refactor (branch cleanup/01b-session-api), same arguments, new run id
python replay_run.py ... --driver session --replay-run 902 --out /tmp/replay_new
python compare_scores.py /tmp/replay_ref /tmp/replay_new
```

`--driver legacy` drives `rt_pipeline.run_rt_pipeline` the way the old launchers did (child process, `terminate()` at the end); `--driver session` uses `start_session`. The DICOMs are copied with `--run` rewritten to `--replay-run` so the run folder is new. Physio is off in both.
