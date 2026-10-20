# README_ELB — fork cleanup log

This file tracks what I (ELB) am changing in this fork of
[K-Demin/DecNef_py](https://github.com/K-Demin/DecNef_py) and why. The
upstream `readme.txt` documents the pipeline itself; this file documents the
packaging/cleanup work, the decisions behind it, and what is still open.
The phase plan is in `CLAUDE.md`.

## Phase status

| phase | branch | status |
|---|---|---|
| 1. Packaging (`pyproject.toml`, `environment.yml`, editable install, tests) | `cleanup/01-packaging` | done, PR pending |
| 1b. Session API (`rt_session.py`; launchers own only the display) | `cleanup/01b-session-api` (stacked on 01) | done, PR pending |
| 2. Clean package, real-time core (`src/decnef/`) | `cleanup/02-decnef-core` (stacked on 01b) | done locally; replay equivalence pending on the lab machine |
| 3. Simulator (`decnef.simulate`: scores / scanner levels, `decnef simulate`) | `cleanup/03-simulator` (stacked on 02) | done locally, both levels run |
| 2. Data hygiene (participant data, non-fixture decoders) | – | not started |
| 3. Configuration (default config, no `/home/sin` paths, personal settings out) | – | not started |
| 4. Vendored code (PyHySCO as a dependency, `librtp` build step, licenses) | – | not started |
| 5. Layout (`src/decnef/`, entry points, tasks to `examples/` or out) | – | not started |
| 6. Docs + CI | – | not started |

## Phase 1 decisions

- Distribution name `decnef`; import names unchanged (flat layout, no moves).
- `requires-python >= 3.10`: runtime `Path | None` annotations
  (`fmri_rt_preproc/config.py:22`, `decoder_score.py:9`, `run_preproc.py:266`, …)
  fail on 3.9. The old readme note that ANTs needs 3.9 referred to an external
  CLI and was corrected. `environment.yml` pins 3.10 because psychopy's
  supported range tops out there.
- torch, PyQt5, pyserial, pandas, matplotlib are **core**, not extras:
  `fmri_rt_preproc/RTPSpy_tools/rtp_common.py` (+ `rtp_volreg.py`, `rtp_regress.py`)
  import them at module level and `rt_pipeline.py` / `run_preproc.py` import
  those. Making them optional needs a lazy-import refactor (phase 4/5).
  torch has a CPU fallback everywhere, so a CPU build is enough to develop and
  test; the real-time machine's CUDA version is still to be confirmed.
- Extras: `[task]` = psychopy, pyglet, pyyaml; `[analysis]` = scikit-learn, h5py;
  `[dev]` = pytest.
- Only editable installs are supported: `run_preproc.py`, `init_pca_conditions.py`,
  `roi_rs_pca_decoder_prep.py` use `<repo>/data`; `fmri_rt_preproc/config.py` uses
  `<repo>/decoders`; `pyhysco_apply.py` and `pipeline.py` use
  `PyHySCO-main/src`; `rtp_volreg.py` loads `librtp.*` next to itself;
  `rt_brainpong_parallel.py` reads `brainpong_settings.json` next to itself.
- The resting-state PCA decoder path is kept as an **optional plugin**, not
  study-specific code: it stays importable, becomes opt-in once the core
  modules stop importing it unconditionally (phase 5).
- Not dependencies: `rtpspy` (only in `__main__` self-test blocks; vendored copies
  are used), `EPI_MRI` (vendored PyHySCO via `sys.path`), the BIOPAC `mpdev.dll`
  (ctypes, Windows). ANTs, FSL, FreeSurfer (SynthStrip, SynthMorph, FastSurfer),
  AFNI, dcm2niix and SPM/MATLAB are subprocess CLIs, documented in `readme.txt`.

## Files not needed to run the package

These stay where they are in phase 1 (no file moves) but are **not** part of
the installed distribution (`py-modules` / `packages` in `pyproject.toml`) and
are not import-tested. In the layout phase they move to a backup folder
(`bkup/`, gitignored) or to `examples/`, except where noted.

### Study-specific

| files | what | blocker before moving |
|---|---|---|
| `rt_robot_parallel.py`, `robot_practice.py` | robot-fight NF display and its standalone practice task | none (`robot_practice.py` runs a PsychoPy window on import) |
| `rt_brainpong_parallel.py`, `brainpong_adapter.py`, `task/Minjoo/`, `brainpong_settings.json` | BrainPong paradigm; hardcoded `/home/sin/...` default | none |
| `roi_rs_pca_decoder_prep.py`, `init_pca_conditions.py`, `rs_pca_runtime.py`, `rs_pca_score_all_rois.py` | resting-state PCA decoder with a blinded condition key | **wired into core**: `rs_pca_runtime` is imported by `rs_realtime_parallel`, `rt_psychopy_parallel`, `rt_robot_parallel` and three test files; `rs_realtime_parallel` launches `roi_rs_pca_decoder_prep.py` and `rs_pca_score_all_rois.py` by path, and `rs_pca_score_all_rois` imports helpers back from it; `tests/test_volume_smooth.py` imports `roi_rs_pca_decoder_prep`. **Decision: becomes an optional plugin** (e.g. a `[pca]` extra / `decnef.plugins.rs_pca` in phase 5), not moved to `bkup/`. **Still installed in phase 1** until the imports above are made lazy and `test_volume_smooth.py` is split. |
| `fmri_rt_preproc/prep_surface_rois.py` | surface ROI prep, only called on the PCA path | inside the `fmri_rt_preproc` package; moves with the PCA set |
| `fmri_rt_preproc/t2_spm_afni_closed_loop.py` | T2 hippocampal (DG) pipeline; nothing imports or launches it | inside the package; none otherwise |

### Analysis and debugging one-offs

| files | what |
|---|---|
| `plot_decoder_dvars_filtered.py`, `plot_decoder_fd_filtered.py`, `plot_fd_decoder.py`, `plot_pca_day_symbol_scores.py`, `plot_scores.py`, `scores_filtered.py` | post-hoc plots of scores against FD and DVARS (study analysis) |
| `scorer_plots.py` | unused plotting class; nothing imports it |
| `compare_decoders.py`, `compare_roi.py` | one-time validation of the Python port against the MATLAB pipeline; `compare_roi.py` has hardcoded paths and runs on import |
| `compare_tsnr_full.py`, `analyze_motion_threshold.py` | ad-hoc QC analyses |
| `streamer_debugger.py` | hardcoded lab IP and Windows paths around `biopac_streamer`; runs on import |
| `biopac_rt/fft_physio_check.py` | physio spectrum check with pasted-in helpers (inside the `biopac_rt` package, so still installed); could become a tool if people use it |
| `*_settings.json` (jane, kostya, minjoo, sara, rt), `minjoo sequence*`, `testOlga.txt`, `data/`, `decoders/` | personal configs, run notes, a participant T1 and study decoders (phase 2/3) |

### Other things noticed for later phases

- `rt_pipeline.py` recognises `rt_nf_events_parallel.py` by name to label trial
  stages; a generic package should read stage definitions from metadata.
- `fmri_rt_preproc/` and `task/` have no `__init__.py` (namespace packages);
  `task` is a generic top-level name. `RTPSpy_tools/RetroTS.py` imports
  `lib_RetroTS` as a top-level name (only works from its own directory).
- `lib_RetroTS/PhaseEstimator.py:22` uses `numpy.Inf` (removed in NumPy 2); not
  on the real-time path.
- Hardcoded `/home/sin/...`, `/SSD2/...`, `/home/sin/anaconda3/.../conda.sh`
  (`fmri_rt_preproc/utils.py:34`) and the FreeSurfer license path
  (`prep_surface_rois.py:159`); `run_in_conda` runs `bash -lc python`, not
  `sys.executable`.
- No `LICENSE` file in the repo.
- `tests/`: no `conftest.py`; bare `pytest tests/` only works after the editable
  install (or `python -m pytest`).

## Phase 1b: session API

> **Direction change (2026-10-20):** phases 1 and 1b were done "upstream-mergeable"
> (every quirk preserved, old names re-exported, legacy metadata forms kept).
> From here on the goal is a **new, lighter, documented package that does the
> same thing** (same numerics, same inputs/outputs) **without the untidy bits**:
> no compatibility shims, no hardcoded lab paths or default decoders, one
> convention for scores/metadata/settings, study code out. Branch `cleanup/01b`
> stays as the last compatible step and as the reference for the replay check;
> the clean package is built next to it (`src/decnef/`), module by module, with
> the replay comparison as the proof of equivalence.

Goal: DecNef_py provides the real-time machinery through a small public API;
the display and trial structure belong to whoever writes the launcher
(eventually my study repo). Pipeline numerics unchanged; no files moved.
Full description: `docs/session_api.md`.

### What was added

- `rt_session.py`: `SessionConfig` (sub/day/run, roots, settings file +
  overrides, decoder, physio, QC viewer, stages, launcher metadata, extra
  processes), `ScoreEvent` (one fixed shape for every queue message; wire
  format stays a dict), `StageSchedule` (generic trial-stage labels written to
  `session_metadata["stages"]`), `PhysioConfig`, `ProcessSpec` (generic hook
  for extra processes such as the PCA scorer), `start_session()` /
  `prepare_session()` and the `Session` object (`scores()`, `poll()`, `stop()`,
  context manager; ordered cooperative shutdown with terminate/kill fallback;
  `atexit`).
- `rt_pipeline.py` (three small edits): `run_rt_pipeline(..., stop_event=None)`
  exits its poll loop when asked so its own cleanup runs; queue payloads carry
  `kind`; `ScoreEventTracker` reads the generic `stages` schema and keeps the
  legacy `fixation_display` / `psychopy.script` forms (tests prove identical
  labels).
- `rt_nf_events_parallel.py`, `rt_psychopy_parallel.py`: ported; reference
  launchers. The display runs in the main process and talks to the session
  only through `ScoreEvent`s; `q` (or Escape) ends the run cleanly. The PCA
  scorer attaches via `ProcessSpec`.
- `replay_run.py` + `compare_scores.py`: exact-match replay harness (lab
  machine; needs `dcm2niix`, a recorded run, the same GPU for both runs).
- Tests: `tests/test_rt_session.py` (fake spawned processes: order, end/error,
  stop, no orphans incl. a daemon grandchild, extra-process hook, physio,
  cleanup on display exception), `tests/test_score_event_tracker.py`,
  `tests/test_nf_events_launcher.py`, `tests/test_psychopy_launcher.py`
  (fake psychopy). 73 -> 97 tests.

### Behavior preserved on purpose (quirks, documented rather than fixed)

- NF-events: `--skip-first-trs` is display-side only (not sent to the
  pipeline); `reg_ready` missing counts as *not* ready (psychopy/robot/
  brainpong: counts as ready).
- psychopy launcher: TR hardcoded to 1.4 s in `acquisition_speed_rt.csv`.
- `_plot_qc` reads top-level `voxel_norm_ref_volumes` / `pre_trial_scans`
  that nothing writes (exclusion always 0).
- Intended change: the PCA scorer now gets a 5 s grace period after its stop
  event before `terminate()` (it was terminated immediately), so its own QC
  plot may complete; the parent re-plot is unchanged.

### Still coupling core to study code (for later phases)

- Default NSF decoder paths (`rt_pipeline.resolve_decoder_template`,
  `run_preproc.py`, `fmri_rt_preproc/config.py`); `/home/sin`, `/SSD2`
  argparse defaults (phase 3); lab IP in `biopac_receiver.py`.
- PCA path couples indirectly: `enable_scoring=False`, `analysis_space`
  override, the PCA reference image passed as `decoder_template` (used as the
  resampling reference); `rs_pca_runtime` layout conventions.
- `psychopy.script == "rt_nf_events_parallel.py"` kept for backward
  compatibility; new runs also write `stages`.
- `rt_robot_parallel.py`, `rt_brainpong_parallel.py`, `rs_realtime_parallel.py`,
  `rs_pca_score_all_rois.py` still own their process code (functionally
  unchanged in this phase).

### Porting the remaining launchers

| launcher | what it would take |
|---|---|
| `rt_robot_parallel.py` | same shape as psychopy (identical helpers and teardown): `SessionConfig` + `ProcessSpec` for PCA; display takes `session`; `q` key. Low risk; left for the study-specific move. |
| `rt_brainpong_parallel.py` | settings come from the BrainPong JSON through `setattr` (no `update()` validation): switching to the session applies `RegressorSettings.update()` to `TR` / `analysis_space`, which could reject or coerce values -> verify with the lab's config first. `--no-pipeline` / simulation paths bypass the session. |
| `rs_realtime_parallel.py` | the display is a *child* and the main process runs preprocessing subprocesses, waits on the pipeline with an unbounded `join()`, then runs `--pca-prep`. Needs: fixation display moved to the main process (or kept as a `ProcessSpec`), a stop condition (max_trs / `q`) feeding `session.stop()`, `fixation_display` metadata -> `StageSchedule(cycle=[("rest", 1)])`. Also fixes today's skipped `--pca-prep` on Ctrl-C. |
| `rs_pca_score_all_rois.py` | imports eight helpers from `rs_realtime_parallel` (spawn children re-import it); its all-ROI scorer is a natural `ProcessSpec`; its `finally` never stops the pipeline today. Port together with RS. |

## Phase 2: the clean package (`src/decnef/`)

Built next to the old tree (kept as the reference until the replay check
passes). Algorithms were moved, not rewritten; signatures, names, error
handling and configuration changed. `README.md`, `docs/config.md`,
`docs/architecture.md`, `docs/session_api.md` describe the result.

### What moved where

| old | new |
|---|---|
| `rt_global_settings.RegressorSettings` + per-launcher argparse + hardcoded defaults | `decnef.config.Settings` from one YAML/JSON (`decnef config migrate` translates old files) |
| `rt_pipeline.RTSessionConfig` path properties (mkdir on access) | `decnef.layout.RunLayout` (no side effects; `create_run_dirs()`) |
| `rt_pipeline` CSV writers, `write_session_metadata`, `load_reference_score_stats` | `decnef.outputs` |
| `decoder_score.DecoderScorer` | `decnef.decoder` (no dead baseline code; grid check) |
| `rt_session.ScoreEvent/StageSchedule` | `decnef.events` (kind + source mandatory) |
| `rt_pipeline.parse_dicom_name`, `prepare_volume_input`, `DICOMHandler` (watch half) | `decnef.ingest` |
| `DICOMHandler` (engine half), `compute_stage`/`commit_stage` | `decnef.engine.OrderedEngine` (legacy engine removed) |
| `process_volume`, `MotionRegressor`, `RTPStyleVoxelNormalizer`, `compute_dvars`, `robust_z`, `motion_fd` | `decnef.stages` (`RunState` + one function per stage) |
| `DICOMHandler.__init__`, `run_rt_pipeline`, `validate_run_provenance`, `maybe_init_*` | `decnef.pipeline` (`check_prerequisites`, `build_state`, `run_pipeline`) |
| `fmri_rt_preproc/{native_fieldmap, analysis_mask, volume_smooth, pyhysco_apply}` | `decnef.{calibration, masks, smoothing, unwarp}` |
| `gpu_ants_like_resampler`, `volume_streamer` | `decnef.resampler`, `decnef.viewer` |
| `biopac_rt/biopac_receiver` | `decnef.physio` (no lab IP) |
| `fmri_rt_preproc/RTPSpy_tools` | `decnef.rtpspy` (PyQt5/pyserial optional, six removed) |
| `rt_session.Session/start_session` | `decnef.session` |
| `rt_pipeline.main`, `replay_run.py`, `compare_scores.py` | `decnef.cli` (`decnef run/replay/compare/config`) |
| `rt_nf_events_parallel.py` display + window helpers | `examples/nf_events_launcher.py` + `decnef.display` |

### Untidy bits removed

No module globals (`REGRESSOR_SETTINGS` gone); no hardcoded paths, lab IP,
default decoder; no `logging.basicConfig` at import; no `print`; no directory
creation in path properties; no dead code; no silent fallbacks (a failed
stage raises -> `kind="error"` on the queue); one `skip_first_trs`; one
meaning for `score_raw`; `reg_ready` always present on scores; TR only from
config; stage labels only from `settings.stages`; PyQt5/pyserial optional.

### Still outside the package (next phases)

Offline preparation (`run_preproc.py`, `fmri_rt_preproc/pipeline.py`, surface
ROIs) -> `decnef.preproc`; PyHySCO as a dependency (currently imported from
the vendored tree by `decnef.unwarp`); `librtp` build step; the rs_pca
plugin; removing the old tree, `data/`, `decoders/`; CI/release.

### Verification

`tests/decnef`: 42 tests (config/layout/outputs/decoder/events, physio socket
round-trip, smoothing numerics, engine ordering/timeout, stages + EPI-space
`process_volume` end to end on synthetic data, prerequisites/provenance,
session lifecycle, example launcher headless). Every `decnef.*` module
imports from outside the repo. `tests/decnef/test_equivalence_synthetic.py`
runs the old `rt_pipeline` and the new `decnef.pipeline` on a synthetic
subject (real PyHySCO / RtpVolreg / RtpRegress; baseline, smoothing,
regression off, original score, z-reference, physio file mode, MNI via the
GPU resampler) and asserts identical CSVs — **all pass**;
`test_pipeline_behaviour.py` covers online ingest, error isolation, the
session path, settings migration and the viewer. The exact-match replay on real data on the lab
machine (DICOM conversion, T1/MNI warp, CUDA) is still to do
(`docs/replay_check.md`); it gates deleting the old tree.

## Phase 3: simulator (`src/decnef/simulate.py`)

A launcher is tested without a scanner by one argument:
`start_session(cfg, simulate=SimulationConfig(level="scores"|"scanner", ...))`
(full description in `docs/session_api.md`, "Simulation").

- **scores**: a fake pipeline process emits one `ScoreEvent` per volume at
  the TR with the stage label from `settings.stages`, `reg_ready` from the
  RtpRegress rule (`max(voxel_norm_ref_volumes, n_regressors + 2)`), then
  `end`; writes `session_metadata.json` and `scores.csv`. No tools.
- **scanner**: a phantom (reference EPI, masks, PyHySCO calibration, decoder,
  identity GPU grid for t1/mni) is generated once per `out_dir`, synthetic
  Siemens-mosaic DICOMs (drift, rigid jitter, a motion spike at volume 15,
  noise, a signal in the decoder's positive voxels during `signal_stages`)
  are written into an incoming folder at the TR, and the **real** pipeline
  runs on them. Needs `dcm2niix` (now in `environment.yml`); ANTs only for
  t1/mni without the GPU resampler. `decnef replay` is the same feeder on
  recorded DICOMs (`SimulationConfig(from_dicoms=...)`).
- Nothing touches the real `paths.base_data`: settings are copied with paths
  into `out_dir` (fresh temp dir, or a given dir that is reused so the
  phantom is not regenerated each time).
- `decnef simulate --level scores|scanner` and
  `examples/nf_events_launcher.py --simulate scores|scanner [--sim-out DIR]`.
- Dependencies: `pydicom` (extra `decnef[sim]`, `environment.yml`), `dcm2niix` (conda-forge).

## Test record

| run | command | result |
|---|---|---|
| baseline (before packaging, env `decnef-baseline`) | `pytest tests/` | 6 collection errors (top-level modules not importable) |
| baseline | `python -m pytest tests/` | 73 passed |
| after packaging (env `decnef` from `environment.yml`, `pip install -e . --no-deps`) | `pytest tests/` and `python -m pytest tests/` | 73 passed, per-test identical to baseline |
| phase 1b (`cleanup/01b-session-api`) | `pytest tests/` | 97 passed (73 unchanged + 24 new); replay exact-match pending on the lab machine |
| phase 2 (`cleanup/02-decnef-core`) | `pytest tests/` | 155 passed (97 old-tree + 58 `tests/decnef`, incl. 7 synthetic old-vs-new equivalence runs: all IDENTICAL, and the online-ingest / error-isolation / session / migration / viewer behaviour tests); real-data replay pending on the lab machine |
| phase 3 (`cleanup/03-simulator`) | `pytest tests/` | 162 passed (155 unchanged + 7 `tests/decnef/test_simulate.py`: scores level, out_dir reuse, mosaic DICOM round trip through dcm2niix, phantom reuse, scanner level through the full pipeline with signal recovery and the FD spike, launcher `--simulate scores` dry run); `decnef simulate --level scores` and `--level scanner` run locally with `configs/jane.yaml` |

Verification details (phase 1): `pip check` clean; the pip step added only
`decnef` to the conda env; setuptools used its import-finder editable mode
(`__editable__.decnef-0.1.0.pth`), which resolves the namespace package
`fmri_rt_preproc` to the repo and keeps every `__file__` inside the repo
(`librtp.dylib` loads from `fmri_rt_preproc/RTPSpy_tools/`). From a directory
outside the repo, all 14 listed modules and all `fmri_rt_preproc.*`,
`biopac_rt.*` modules import; the excluded study/one-off scripts are not
importable from outside the repo by design (they still run from the repo
root with `python <script>.py`).



## Thinking about the independence of this package
my-decnef-study  ──imports──▶  decnef (DecNef_py)
  task/display code              real-time engine, preprocessing,
  launcher for my paradigm       scoring, physio, QC
  decoder training/analysis
  my settings

The key change: who owns the launcher

Right now the launcher scripts (rt_*_parallel.py) sit inside DecNef_py and combine two jobs:

starting the pipeline process, the physio listener and the QC viewer
running a specific task display

For a clean split, DecNef_py should provide the first job as a function, and your repo should own the second. Roughly: 