# decnef — real-time fMRI decoded neurofeedback

A pipeline that takes each incoming DICOM volume through unwarping, motion
correction, quality metrics, nuisance regression, resampling to a decoder's
space, optional masking/smoothing and decoder scoring within the TR, and
hands the score to a display. This fork (`ericabusch/DecNef_py`) turns the
original flat scripts of [K-Demin/DecNef_py](https://github.com/K-Demin/DecNef_py)
into an installable, documented package, `src/decnef/`, without changing the
numerics: on a recorded 7T run the package and the original scripts produce
byte-identical outputs (section 7).

The original README is kept as `readme_old.txt` (offline preparation,
settings keys, data layout as written by the lab). `README_ELB.md` is the
detailed cleanup log (decisions, every bug found, the test record).
`docs/decnef_overview.md` is the review document: what changed, how the
modules interact, what has been validated, open questions.

```
src/decnef/
  config.py       Settings: one YAML/JSON file, validated, no hidden defaults     docs/config.md
  layout.py       every path of a subject/day/run
  session.py      start_session(): pipeline + physio + extras as child processes;  docs/session_api.md
                  ScoreEvents to the display; simulate=SimulationConfig(...) for dry runs
  pipeline.py     run_pipeline(): prerequisites, provenance, the per-volume loop    docs/architecture.md
  stages.py       unwarp, motion+FD, censor+DVARS, regression, transform, smooth, score
  engine.py       parallel compute, ordered commit
  ingest.py       DICOM names, dcm2niix, incoming-folder watcher
  simulate.py     scores-level and scanner-level simulation, replay feeder, phantom
  calibration.py  native PyHySCO calibration: load/validate/build
  decoder.py      decoder file formats and scoring
  physio.py       BIOPAC RetroTS receiver / file buffer
  masks.py, smoothing.py, resampler.py, unwarp.py, viewer.py, outputs.py, events.py, tools.py
  rtpspy/         vendored RTPSpy volreg / regress (+ librtp)
  display.py      optional PsychoPy helpers (window, trigger, stop keys)
  cli.py          decnef run | simulate | replay | compare | calibrate | config
examples/nf_events_launcher.py     a complete paradigm on the session API (--simulate for dry runs)
examples/*.args                    argument files for the equivalence check
configs/*.yaml                     the lab's settings files, migrated (configs/README.md)
docs/                              config, session API, architecture, replay check, overview
tests/decnef/                      the package's tests; tests/test_*.py still cover the old scripts
```

## 1. Status

| | |
|---|---|
| Real-time core (`src/decnef/`) | done; **IDENTICAL** to the original scripts on a full real run (600 volumes, lab machine, 2026-10-23) and on 7 synthetic configurations |
| Simulator / replay / compare | done; both simulation levels run on macOS and on the lab's Linux machine |
| Offline preparation (`run_preproc.py`, `fmri_rt_preproc/`) | **still the old scripts** (planned `decnef.preproc`) |
| Resting-state PCA decoder, robot / BrainPong / RS launchers | still the old scripts (planned plugin / examples) |
| Old tree removal, participant data out of the repo, PyHySCO as a dependency, `librtp` build, CI | not started (section 8) |
| Remaining validation | second subject/config, MNI-space subject, display on the scanner machine, live phantom scan (`docs/decnef_overview.md` §6.4) |

## 2. Install

```bash
git clone https://github.com/ericabusch/DecNef_py.git && cd DecNef_py
conda env create -f environment.yml && conda activate decnef    # or: conda env update -f environment.yml --prune
pip install -e . --no-deps                                       # the package is `decnef`; only editable installs for now
pip install -e ".[task]"                                         # PsychoPy, for launchers / examples (not on conda-forge)
python -m pytest tests/decnef -q                                 # ~90 s; the two scanner-level tests skip without dcm2niix
```

- `environment.yml` brings `dcm2niix` and `pydicom`; `torch` falls back to CPU
  everywhere, the real-time machine should have the CUDA build.
- ANTs (`antsApplyTransforms`) is needed only for `analysis_space: t1|mni`
  **without** `resampler.use_gpu`. The offline preparation additionally needs
  FSL, FreeSurfer (SynthStrip/SynthMorph), FastSurfer (`readme_old.txt`).
- macOS: the committed `librtp.dylib` needs `brew install libxt`.
- The lab machine: `/SSD2/erica/DecNef_py`, env `decnef_elb`.

## 3. Configuration

One file per study, YAML or JSON (`docs/config.md`); every setting is in it,
no hidden defaults, a missing required value fails at load time. The lab's
five settings files are already migrated in `configs/` (`configs/README.md`
tabulates how they differ). Useful commands:

```bash
decnef config migrate old_settings.json --base-data /SSD2/DecNef_py/data --decoder-template /path/d.nii --out my.yaml
decnef config show --config configs/minjoo.yaml                     # the effective tree
decnef run ... --set pipeline.TR=1.4 --set decoder.template=/path/d.nii   # override any key on the command line
```

Simulation parameters are not config keys: they are passed to
`start_session(cfg, simulate=...)` or as `decnef simulate` flags.

## 4. Running a session

### 4.1 Prepare the subject/day (offline, once)

Still the original scripts: `python run_preproc.py --sub S --day D ...`
(`readme_old.txt`). It produces `sub-S/anat/`, `S/D/fmap/pair-apNNN_paMMM/`,
`S/D/func/trans/` under `paths.base_data`. The PyHySCO **native
calibration** (`native_unwarp_v1/` inside the pair folder) is built by the
preparation, or at run start from the raw `AP/PA.nii.gz` if missing (as the
old pipeline did), or explicitly:

```bash
decnef calibrate --config configs/minjoo.yaml --sub 99988 --day 8 --ap-block 2 --pa-block 3
```

The fieldmap pair is the AP/PA series acquired just before the run
(`fmap/pair-ap002_pa003` → `--ap-block 2 --pa-block 3`).

### 4.2 Run

```bash
# pipeline only, headless
decnef run --config configs/minjoo.yaml --sub 99988 --day 8 --run 4 --ap-block 2 --pa-block 3 \
    --incoming /home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/<session folder>

# with the example display (PsychoPy): 's' starts on the scanner trigger, 'q' / Esc ends the run
python examples/nf_events_launcher.py --config configs/minjoo.yaml --sub 99988 --day 8 --run 4 \
    --ap-block 2 --pa-block 3 --incoming /path/to/dicoms --n-trials 20 --baseline-trs 30
```

A launcher is a few lines around `start_session` (`docs/session_api.md`):

```python
from decnef.config import load_settings
from decnef.session import SessionConfig, start_session

settings = load_settings("configs/minjoo.yaml")
cfg = SessionConfig(settings, sub="99988", day="8", run="4", ap_block=2, pa_block=3, incoming_root="/path/to/dicoms")
with start_session(cfg) as session:
    for event in session.scores(timeout=0.02):          # or session.poll() inside a display loop
        if event.kind == "score" and event.reg_ready:    # use only reg_ready scores for feedback
            show(event.score_raw, event.stage)
        if event.kind == "end" or finished:
            break
```

Every event is a `ScoreEvent` (`kind` = `tick | score | scorer | error | end`,
`volume_idx`, `score_raw`, `reg_ready`, `stage`, timestamps). `reg_ready`
is false until RtpRegress has `max(voxel_norm_ref_volumes-1, n_regressors+1)`
volumes; scores before that are not comparable and must not drive feedback.
A volume that fails a stage arrives as `kind="error"`; nothing is substituted.

### 4.3 Outputs (`<base_data>/sub-S/D/func/R/`)

`raw/ unwarped/ mc/ reg/ t1|mni/ [smooth|masked/]` per-volume NIfTIs;
`scores.csv` (`volume_idx, timestamp, score_raw, score_original, reg_ready, event_type`),
`scores_z.csv`, `motion_rt.1D`, `fd_rt.csv`, `dvars_rt.csv`, `regressors_rt.csv`,
`regression_status_rt.csv`, `session_metadata.json` (pipeline keys, the full
settings tree, the launcher's keys, a `simulation` block for simulated runs),
`preprocessing_order.json` (provenance; a run folder with a different marker or
with processed volumes is refused). Column lists: `docs/architecture.md`.

### 4.4 Testing a launcher without a scanner

```bash
decnef simulate --config configs/rt.yaml --set physio.enable=false --sub 00 --day 1 --run 1 --level scores --n-volumes 40
decnef simulate --config configs/rt.yaml --set physio.enable=false --sub 00 --day 1 --run 1 --level scanner --out /SSD2/erica/sim
python examples/nf_events_launcher.py --config configs/rt.yaml --sub 00 --day 1 --run 1 --simulate scanner --sim-out /SSD2/erica/sim
```

- `scores`: a fake pipeline emits one event per TR with correct stage labels
  and `reg_ready` timing; no tools.
- `scanner`: a phantom and synthetic Siemens-mosaic DICOMs (drift, jitter, a
  motion spike, noise, a signal injected during `signal_stages`) go through
  the **real** pipeline; needs `dcm2niix`. `--out` is reused between runs
  (the phantom is generated once); nothing is written under the study's
  `paths.base_data`.
- `--tr` paces delivery only; the pipeline always runs with the config's TR.

In code: `start_session(cfg, simulate=SimulationConfig(level="scanner", out_dir=...))`;
`session.run_dir` is the simulated run folder, `session.simulation` its paths.

### 4.5 Replaying a recorded run and comparing with the original scripts

```bash
python replay_run.py @examples/legacy_99988.args          # old rt_pipeline.py on a copy of the subject's data -> run 903
decnef replay @examples/replay_99988.args --max-volumes 40 # the package on the same copy -> run 902
decnef compare /SSD2/erica/replay_99988/legacy_ref3 /SSD2/erica/replay_99988/data/sub-99988/8/func/902
```

`decnef replay` copies `anat/`, `fmap/` (calibration built in the copy if
missing), `func/trans/` and the z-reference run into `--out/data/`, feeds the
recorded DICOMs at `--tr` with the run number rewritten, and never writes the
real tree. `IDENTICAL (timestamp columns ignored)` is the pass condition. Use
`--tr 3.0` for the legacy driver (it loses a volume arriving during its
startup at the real TR). Recipe and pitfalls: `docs/replay_check.md`.
Argument files (`@file`, one or more flags per line, `#` comments, later
flags override) work with every `decnef` subcommand and `replay_run.py`.

## 5. Changes relative to the original scripts

Numerics unchanged (PyHySCO, RtpVolreg/librtp, RtpRegress, GPU resampler,
decoder, output files, data layout, `unwarp_then_mc_v1` order). Changed:

- one package, one config file, `start_session` instead of per-launcher
  process code; no module globals, no hard-coded lab paths, decoders or IPs;
- a failed volume is reported (`kind="error"`), never replaced by zeros or an
  un-regressed volume; `reg_ready` always present; one `skip_first_trs`;
  stage labels from `settings.stages`; TR only from config;
- ordered shutdown (`q`/Esc, Ctrl-C) with terminate/kill fallback;
- two watcher bugs fixed (see below); PyQt5/pyserial optional;
- new: simulator, replay/compare, `decnef calibrate`, argument files.

Full table and rationale: `docs/decnef_overview.md` §3.

### Bugs found in the original scripts (open upstream)

1. **Startup gap** — `rt_pipeline.py` lists the incoming folder, builds its
   state for 1–2 s, then starts the watcher; a DICOM landing in between is
   silently lost. Harmless when the pipeline is started before the scan.
2. **Renamed files invisible on Linux** — only `created`/`closed` events are
   handled; a file moved into the folder is ignored.

Both are fixed in the package and have regression tests.

## 6. Repository layout: what stays, moves or goes

| area | files | plan |
|---|---|---|
| **Package (stays)** | `src/decnef/`, `examples/`, `configs/`, `docs/`, `tests/decnef/`, `pyproject.toml`, `environment.yml` | — |
| **Replaced by the package → delete** once the remaining validation (section 7) is done | `rt_pipeline.py`, `rt_global_settings.py`, `rt_session.py`, `decoder_score.py`, `motion_fd.py`, `gpu_ants_like_resampler.py`, `volume_streamer.py`, `biopac_rt/biopac_receiver.py`, `fmri_rt_preproc/{native_fieldmap,analysis_mask,volume_smooth,pyhysco_apply}.py`, `fmri_rt_preproc/RTPSpy_tools/`, `replay_run.py` (after the last legacy comparison), `compare_scores.py`, `rt_nf_events_parallel.py`, `tests/test_*.py` | delete |
| **Still needed → into the package later** | `run_preproc.py`, `fmri_rt_preproc/{pipeline,config,utils}.py`, `prep_surface_rois.py` | `decnef.preproc` |
| | `fmri_rt_preproc/PyHySCO-main/` | a PyHySCO dependency (`decnef.unwarp` and `decnef.calibration` import/run it from the vendored tree today) |
| | `fmri_rt_preproc/RTPSpy_tools/afni_module/` (C sources) | `librtp` build step; committed `.so`/`.dylib` removed |
| | `rs_pca_runtime.py`, `roi_rs_pca_decoder_prep.py`, `rs_pca_score_all_rois.py`, `init_pca_conditions.py` | `decnef.plugins.rs_pca` (optional) |
| | `rt_psychopy_parallel.py` | `examples/` |
| | `biopac_rt/biopac_streamer.py`, `biopac_receiver_tester.py` | BIOPAC-PC side: `tools/` or its own package |
| **Study-specific / one-off → out of the repository** (study repo, or gitignored `bkup/`) | `rt_robot_parallel.py`, `robot_practice.py`, `rt_brainpong_parallel.py`, `brainpong_adapter.py`, `brainpong_settings.json`, `task/Minjoo/`, `rs_realtime_parallel.py`, `plot_*.py`, `scores_filtered.py`, `scorer_plots.py`, `compare_decoders.py`, `compare_roi.py`, `compare_tsnr_full.py`, `analyze_motion_threshold.py`, `streamer_debugger.py`, `biopac_rt/fft_physio_check.py`, `fmri_rt_preproc/t2_spm_afni_closed_loop.py`, `*_settings.json` (migrated to `configs/`), `minjoo sequence*`, `testOlga.txt` | move out |
| **Data → out of the repository** | `data/sub-*`, `decoders/` | data-hygiene phase; history scrubbing coordinated with the lab |
| **Docs → folded into `docs/`** | `readme_old.txt`, `PLAN.md`, `docs/parallel_compute_commit_design.md`, `docs/unwarp_before_mc.md` | merge |

The old scripts still run from the repository root exactly as before until
they are removed.

## 7. Validation

- `python -m pytest tests/` — 170 tests: 97 for the old scripts, 73 for the
  package (config, layout, outputs, engine, every stage, session lifecycle,
  online ingest with out-of-order arrivals, error isolation, the startup-gap
  regression, the example launcher with a stub PsychoPy, both simulation
  levels, replay staging, calibration build with real PyHySCO).
- **7 synthetic old-vs-new runs** (`tests/decnef/test_equivalence_synthetic.py`):
  identical CSVs for baseline, masking + smoothing, regression off, original
  score, z-reference, physio file mode, MNI via the GPU resampler.
- **Real data, lab machine, 2026-10-22/23**: sub-99988 day 8 run 4 (7T, T1
  space, GPU resampler, CUDA), package vs `rt_pipeline.py` on the same copied
  data — 40 volumes: `IDENTICAL` scores / regression status / FD / motion,
  md5-identical `reg/ mc/ t1/` volumes, identical regressors; then the
  **full 600-volume run** (package at the real TR 1.0 s, legacy at 3.0 s):
  `IDENTICAL`.
- Still to do: a second subject/config,
  an MNI-space subject, the display and QC viewer on the scanner machine,
  shutdown mid-run, BIOPAC in TCP mode, and a live phantom scan with
  `decnef run` (`docs/decnef_overview.md` §6.4).

Per-phase test record: `README_ELB.md`.

## 8. Next phases

1. Finish the validation list above; then delete the replaced scripts.
2. `decnef.preproc`: the offline preparation with tool paths, conda env and
   FreeSurfer licence in the config.
3. Dependencies: PyHySCO from a package index or a pinned git source,
   `librtp` built at install time, licences checked.
4. Data hygiene: participant data and non-fixture decoders out of the tree.
5. rs-PCA plugin; remaining launchers to `examples/` or the study repo.
6. CI (pytest on push), tagged releases, a `LICENSE` file.

## Documentation index

| file | content |
|---|---|
| `docs/decnef_overview.md` | review document: changes, diagrams, validation, open questions |
| `docs/config.md` | every setting, migration from the old JSON |
| `docs/session_api.md` | `start_session`, `ScoreEvent`, stages, extra processes, simulation |
| `docs/architecture.md` | processes, run start, per-volume stages, output columns, engine |
| `docs/replay_check.md` | the equivalence recipe, pitfalls, what is removed afterwards |
| `configs/README.md` | the lab's migrated settings files |
| `README_ELB.md` | cleanup log: decisions, bugs, test record |
| `readme_old.txt` | the original README (offline preparation, settings keys) |
