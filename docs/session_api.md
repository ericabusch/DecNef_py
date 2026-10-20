# Session API (`decnef.session`)

A launcher owns the display and the trial structure; the session owns the
machinery. The display runs in the launcher's main process and talks to the
session only through `ScoreEvent`s and `session.stop()`.

```python
from decnef.config import StagesConfig, load_settings
from decnef.session import ProcessSpec, SessionConfig, start_session

settings = load_settings("decnef.yaml")
settings.stages = StagesConfig(baseline_trs=20, cycle=[["iti", 3], ["cue", 4], ["scans", 3], ["delay", 3], ["feedback", 3]], n_cycles=20)

cfg = SessionConfig(
    settings, sub="00086", day="3", run="4",
    incoming_root=None,                         # overrides settings.paths.incoming_root
    ap_block=None, pa_block=None,               # fieldmap pair (both or neither)
    metadata={"launcher": {...}},               # merged into session_metadata.json before the pipeline starts
    extra_processes=[ProcessSpec("pca_scorer", target, kwargs={...})],
    queue_size=200, stop_timeout_s=10.0,
)
with start_session(cfg) as session:             # physio listener -> pipeline -> extras, all spawned
    for event in session.scores(timeout=0.02):  # or session.poll() inside a frame loop
        ...
        if user_pressed_q:
            break
# leaving the block runs session.stop(): extras -> pipeline -> physio, also on Ctrl-C,
# a display exception, or a pipeline error (scores() raises SessionError after the error event)
```

`cfg.layout` gives the `RunLayout` (run directory, file paths) before starting.

## ScoreEvent (`decnef.events`)

Every message on the queue is a dict with `kind` and `source`; `ScoreEvent.from_message` builds the event, no guessing.

| field | meaning |
|---|---|
| `kind` | `tick` (volume processed, scoring off), `score`, `scorer` (an extra process), `error`, `end` (pipeline exited; last event) |
| `source` | `pipeline`, the `ProcessSpec.name`, or `session` for `end`/`error` |
| `volume_idx` | DICOM scan number, 1-based |
| `timestamp`, `watchdog_timestamp`, `analysis_timestamp` | score written / DICOM picked up / after scoring |
| `score_raw`, `score_z`, `score_original` | decoder score; z vs the reference run; score of the un-regressed volume |
| `reg_ready` | always present on `score`: regression warm-up passed |
| `stage` | trial-stage label (`scores.csv` `event_type`) |
| `meta` | everything else: an extra process's values (`feedback_score`, ...), `error`, `exit_code` |

Messages are dropped by the sender when the queue is full; the CSVs on disk
are complete regardless.

## Stage schedule

`settings.stages` (`docs/config.md`) labels volume `v` (1-based): `pre_label`
while `v <= pipeline.skip_first_trs`; `baseline_label` for the next
`baseline_trs` volumes; then the cycle repeats `n_cycles` times (`null` =
forever); `post_label` afterwards. `StageSchedule.from_config(settings.stages,
settings.pipeline.skip_first_trs)` gives the schedule object (`label_for_volume`,
`windows()`); the pipeline writes the same labels to `scores.csv` and the
schedule to `session_metadata["stages"]`.

## Extra processes

`ProcessSpec(name, target, kwargs, pass_queue="score_queue", pass_stop_event="stop_event", stop_timeout_s=5.0)`:
the session adds the queue and a stop `Event` to `kwargs` under those names,
starts the process after the pipeline and stops it before. Messages it puts
on the queue must carry `kind` and `source`.

## Testing a launcher without a scanner

`Session(run_dir, pipeline_target, pipeline_args)` accepts any module-level
function as the pipeline (`tests/decnef/test_session.py`); a display can be
driven by any object with `poll() -> list[ScoreEvent]` and a stub `psychopy`
(`tests/decnef/test_nf_events_example.py`).
