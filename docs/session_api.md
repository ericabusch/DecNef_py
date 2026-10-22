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

## Simulation

A launcher is tested end to end without a scanner by passing one extra
argument; nothing else in the launcher changes:

```python
from decnef.simulate import SimulationConfig

with start_session(cfg, simulate=SimulationConfig(level="scores")) as session:
    for ev in session.scores():
        ...
```

Two levels:

| level | what runs | what it guarantees | tools |
|---|---|---|---|
| `scores` | a fake pipeline process | one `score` `ScoreEvent` per volume at the TR, correct stage labels (`settings.stages`), `reg_ready` flipping at the volume the real RtpRegress would (`max(voxel_norm_ref_volumes, n_regressors + 2)`), an `end` event, `session_metadata.json` + `scores.csv` in the run folder; events carry `meta["simulated"] = True` | none |
| `scanner` | the **real** pipeline on synthetic Siemens-mosaic DICOMs written into an incoming folder at the TR | every real output file (`raw/ unwarped/ mc/ reg/`, `motion_rt.1D`, `fd_rt.csv`, `regressors_rt.csv`, `regression_status_rt.csv`, `scores.csv`, ...), real motion correction / censoring on injected jitter and a motion spike at volume 15, a signal injected in the decoder's positive voxels during `signal_stages` that shows up in the scores | `dcm2niix`; `antsApplyTransforms` only for `analysis_space: t1/mni` without `resampler.use_gpu` |

What is synthetic at the scanner level: the phantom (a blurred ellipsoid
"brain" on a `grid` of `voxel_mm` voxels), its reference EPI and masks, the
PyHySCO calibration (a smooth field on the staggered grid, i.e. unwarping
runs but does little), the decoder (half the brain positive, half negative)
and, for t1/mni space, an identity GPU resampling grid. The DICOMs are mosaics
dcm2niix converts exactly like scanner output; `ingest.parse_dicom_name`
names (`SSS_RRRRRR_VVVVVV.dcm`) are used.

Nothing is written under the real `paths.base_data`: the session gets a copy
of `settings` whose `paths.base_data`, `paths.incoming_root` and
`decoder.template` point into the simulation directory, and the run lives at
`<out_dir>/data/sub-<sub>/<day>/func/<run>`. `session.run_dir` is that folder;
`session.simulation` (a `SimPaths`) exposes `out_dir`, `base_data`,
`incoming`, `run_dir`, `decoder`, `phantom_manifest`; `session.feeder` is the
DICOM feeder thread at the scanner level (`None` otherwise).

`SimulationConfig` fields:

| field | default | meaning |
|---|---|---|
| `level` | `"scores"` | `"scores"` or `"scanner"` |
| `n_volumes` | 60 | volumes to produce; the fake pipeline / feeder stops after them |
| `tr` | `settings.pipeline.TR` | seconds between *delivered* volumes (pacing only). The pipeline always runs with `pipeline.TR`, which sets RtpRegress's polynomial order and the physio alignment, so a slow replay gives the same numbers as a real-time one |
| `out_dir` | fresh `tempfile.mkdtemp("decnef_sim_")` | an existing directory is **reused**: the phantom and calibration are regenerated only when `phantom.json` is missing or its parameters differ; the run folder is always recreated |
| `sub`, `day` | `"sim"`, `"1"` | phantom subject/day under `out_dir/data` |
| `grid`, `voxel_mm` | `(32, 32, 20)`, 3.0 | phantom geometry |
| `motion_amp_mm`, `drift`, `noise_sd` | 0.5, 0.001, 5.0 | rigid jitter amplitude, linear drift per volume (fraction), Gaussian noise |
| `signal_amp` | 0.02 | fractional signal added to the decoder's positive voxels while "on" |
| `signal_stages` | `("scans",)` | stage labels during which the signal is on; without `settings.stages`, 10-on/10-off blocks |
| `seed` | 0 | RNG seed |
| `from_dicoms`, `source_run` | `None` | scanner level from a recorded run instead of a phantom (= `decnef replay`): the real `cfg.sub/cfg.day` is kept, the subject's prepared data (`anat/`, `fmap/`, `func/trans/`, the reference run) is copied into `out_dir/data` once and the real tree is only read; the decoder is read in place; a missing native calibration is built in the copy from the raw AP/PA |
| `reg_ready_from` | RtpRegress rule | scores level: first `reg_ready` volume |
| `baseline_score`, `score_noise_sd` | 0.0, 1.0 | scores level: score distribution |

CLI, sharing the implementation with `decnef replay`:

```bash
decnef simulate --config decnef.yaml --sub 00 --day 1 --run 4 --level scores --n-volumes 40
decnef simulate --config decnef.yaml --sub 00 --day 1 --run 4 --level scanner --out /tmp/sim --tr 1.0
decnef replay   --config decnef.yaml --sub 00 --day 1 --run 4 --from-dicoms /path/to/incoming/run4   # scanner level, recorded data
python examples/nf_events_launcher.py --config decnef.yaml --sub 00 --day 1 --run 4 --simulate scanner --sim-out /tmp/sim
```

`decnef simulate` prints one line per volume and ends with a JSON summary
(`decnef.simulate.summarize_run`: `mean_on` / `mean_off` / `corr_with_signal`
over the `reg_ready` volumes), which is the quick check that the injected
signal came through the pipeline.

For unit tests, `Session(run_dir, pipeline_target, pipeline_args)` also
accepts any module-level function as the pipeline
(`tests/decnef/test_session.py`); a display can be driven by any object with
`poll() -> list[ScoreEvent]` and a stub `psychopy`
(`tests/decnef/test_nf_events_example.py`).
