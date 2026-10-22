# Equivalence check on the lab machine, and what gets removed afterwards

The clean package (`src/decnef/`) must produce the same numbers as the
original scripts before the old tree is deleted. This page is the recipe for
that check, followed by the list of what moves or goes once it passes.

## Synthetic check (no scanner, runs anywhere)

`tests/decnef/test_equivalence_synthetic.py` (helpers in `tests/decnef/synth.py`)
builds a synthetic subject (reference EPI, masks, fieldmap calibration with a
small field, decoders, 20 drifting/noisy volumes), runs the **old**
`rt_pipeline` and the **new** `decnef.pipeline` on the same volumes and
asserts the output CSVs are identical, for each of:

- the baseline EPI-space run (regression active from volume 12 on)
- final masking + smoothing (`smoothing_fwhm_mm=4`)
- regression off (voxel normalisation)
- `enable_original_score`
- z-scoring against a reference run (`scores_z.csv`)
- physio regressors in file mode (RICOR8 from a CSV, one volume missing -> zero-filled)
- the MNI-space warp through the GPU resampler on CPU (synthetic sampling grid)

`tests/decnef/test_pipeline_behaviour.py` covers what has no old counterpart:
online ingest through the watcher with out-of-order arrivals, a corrupt
volume reported as `kind="error"` while the run continues (no substituted
zeros), the full `start_session` path with stage labels and metadata,
migration of the lab's real settings files, and the QC viewer process.

What is real in these: PyHySCO unwarp, RtpVolreg (librtp), RtpRegress,
FD/DVARS censoring, smoothing, the resampler, the decoder. What is not:
`dcm2niix` (raw volumes are pre-converted; a stub satisfies the prerequisite
check) and ANTs (`analysis_space=epi`, or the cached GPU grid). ~1 minute on
the development Mac. On macOS the committed `librtp.dylib` needs
`brew install libxt`; the tests skip when librtp cannot be loaded. The
lab-machine replay below additionally covers the real DICOM conversion, the
ANTs warp and the CUDA device.

## The replay check (real data, lab machine)

**Status: passed on 2026-10-22** - sub-99988 day 8 run 4 (7T, T1 space, GPU
resampler, pair ap002/pa003), 40 volumes, `IDENTICAL`. Details in
`README_ELB.md`, test record. Use `--tr 3.0` for the legacy driver: at the
real TR it loses the volume that arrives during its startup (fixed in the
package, not in the old tree).
`--tr` paces the arrivals only (both drivers); the pipeline runs with the
config's TR, so the two runs are comparable whatever the pacing.

Everything runs on the lab machine: it needs `dcm2niix` on `PATH`, the GPU
(both runs on the same device), and the subject/day's offline preparation
(`func/trans/`, the fieldmap calibration). Nothing here runs on a laptop.

### 0. Code

The branches live on the fork. On the lab machine:

```bash
git clone https://github.com/ericabusch/DecNef_py.git decnef_test && cd decnef_test
conda env create -f environment.yml && conda activate decnef   # or an existing env with torch (CUDA) and dcm2niix on PATH
```

### 1. Pick a recorded run

You need, for one run that was already acquired:

- the incoming folder with its DICOMs `SSS_RRRRRR_VVVVVV.dcm` (e.g. `.../incoming/pain7T/<date>.<sub>/`); note the run field `R`
- the `--base-data` root where that subject/day has `func/trans/` and the fieldmap pair (and `--ap-block/--pa-block` if the run used a specific pair)
- the decoder file the run used
- the settings JSON the run used (`rt_settings.json` / `<name>_settings.json`)
- optionally the z-score reference run id (`--rs`) if the run used one

### 2. Reference from the old code (branch `cleanup/01b-session-api`)

```bash
git checkout cleanup/01b-session-api && pip install -e . --no-deps
python replay_run.py --dicom-dir /path/incoming --run R --tr 1.4 --driver legacy \
    --sub SUB --day DAY --base-data /SSD2/DecNef_py/data --replay-run 901 --out /tmp/replay_ref \
    --settings-file /path/settings.json --decoder-template /path/decoder.nii \
    [--ap-block A --pa-block P] [--rs REFRUN]
```

- `--replay-run 901`: any run id with **no existing** `func/901/` folder (the provenance check refuses reuse).
- `--tr`: the arrival pacing in seconds; use the real TR.
- The DICOMs are copied into a temporary incoming folder with the run field rewritten to 901, in scan order, one per TR.

### 3. The package (branch `cleanup/02-decnef-core`)

```bash
git checkout cleanup/02-decnef-core && pip install -e . --no-deps
decnef config migrate /path/settings.json --base-data /SSD2/DecNef_py/data \
    --decoder-template /path/decoder.nii --out decnef.yaml
decnef config show --config decnef.yaml          # check the effective values against the old run's session_metadata.json
decnef replay --config decnef.yaml --sub SUB --day DAY --dicom-dir /path/incoming --run R \
    --replay-run 902 --out /tmp/replay_new [--ap-block A --pa-block P] \
    [--set decoder.reference_score_run=REFRUN]
```

`decnef replay` forces physio and the QC viewer off, like the legacy driver.
It never writes into the real `paths.base_data`: the subject's prepared data
(`anat/`, `<day>/fmap/` with its calibration, `<day>/func/trans/`, and the
reference run's folder if `decoder.reference_score_run` is set) is **copied**
once into `<out>/data/sub-SUB/DAY/` (reused on later replays into the same
`--out`), the recorded DICOMs are copied into `<out>/incoming/` one per TR
with the run field rewritten to `--replay-run`, and the new run's outputs
land in `<out>/data/sub-SUB/DAY/func/902/`. The decoder is read in place.
The copied `calibration.json` is relabelled to point at the copied AP/PA
sources (it records their path/size/mtime as provenance; the real one is
verified first). Subjects prepared before the native calibration
existed (pair folders holding only `pyhysco_epi-*` outputs) get it built in
the copy from the raw `AP.nii.gz`/`PA.nii.gz` before the pipeline starts -
the preparation's own step (means without motion correction, PyHySCO 50
iterations), which is what the old `rt_pipeline.py` did at startup
(`ensure_fieldmap_only`). Minutes on CPU, seconds on the GPU. To build it in
the real tree once instead: `decnef calibrate --config decnef.yaml --sub SUB
--day DAY --ap-block A --pa-block P`.

**Which reference to compare against.** The run's original `func/R/` outputs
were produced by the code of that day; upstream changed the pipeline since
(unwarp before motion correction and the native calibration on 2026-09-25,
final smoothing/mask selection on 2026-10-01). `decnef compare` against the
original `func/R/` is therefore only meaningful for runs acquired after those
changes; otherwise the reference is the legacy driver of step 2 run on the
same copied data. Compare against the legacy output with
`decnef compare /tmp/replay_ref /tmp/replay_new/data/sub-SUB/DAY/func/902`.

### Argument files (no copy-pasting)

Every `decnef` subcommand and `replay_run.py` accept `@file`: one or more
arguments per line, `#` comments, later flags override. The two runs above
are in `examples/replay_99988.args` and `examples/legacy_99988.args`:

```bash
python replay_run.py @examples/legacy_99988.args
decnef replay @examples/replay_99988.args --max-volumes 40
```

### 4. Compare

```bash
decnef compare /tmp/replay_ref /tmp/replay_new
```

`IDENTICAL (timestamp columns ignored)` is the pass condition. Otherwise the
first differing rows/columns of `scores.csv`, `scores_z.csv`,
`regression_status_rt.csv`, `fd_rt.csv` and `motion_rt.1D` are printed.

### Spurious differences and how to avoid them

| symptom | cause | fix |
|---|---|---|
| small numeric differences everywhere | different device between the two runs (RtpRegress uses CUDA when available) | same machine, same `pipeline.fieldmap.device` / `resampler` for both |
| `reg_ready`, FD NaNs or score rows differ from the start | a setting the old launcher passed on its command line (`--skip-first-trs`, `--baseline-trs` → `voxel_norm_ref_volumes`, `--rt-max-scan-length`) is not in `decnef.yaml` | add it with `--set pipeline.<key>=...`; compare `decnef config show` with the old run's `session_metadata.json` |
| row-count difference | DICOMs arrived faster than one driver processed them and a scan was skipped by the commit-wait timeout | rerun both with a slower `--tr` (e.g. 3.0); equivalence is about numerics, not timing |
| `already has scores.csv` | reused `--replay-run` | pick a new id |
| long runtime | a full run takes about N × TR seconds | `--max-volumes 40` on both for a first pass |

Send the `compare` output (and, if different, both `session_metadata.json`
files) to continue from there.

## What is removed or moved once the check passes

Destination per file; the top level of the repository ends up as
`src/decnef/`, `examples/`, `tests/`, `docs/`, `pyproject.toml`,
`environment.yml`, `README.md`, `LICENSE`.

### Already replaced by `decnef` → delete

`rt_pipeline.py`, `rt_global_settings.py`, `rt_session.py`, `decoder_score.py`,
`motion_fd.py`, `gpu_ants_like_resampler.py`, `volume_streamer.py`,
`biopac_rt/biopac_receiver.py`,
`fmri_rt_preproc/{native_fieldmap,analysis_mask,volume_smooth,pyhysco_apply}.py`,
`fmri_rt_preproc/RTPSpy_tools/` (now `decnef/rtpspy/`), `replay_run.py`,
`compare_scores.py`, `rt_nf_events_parallel.py` (now
`examples/nf_events_launcher.py`), the old `tests/test_*.py` (coverage moved to
`tests/decnef/`).

### Still needed → moved into the package in later phases

| files | destination |
|---|---|
| `run_preproc.py`, `fmri_rt_preproc/{pipeline,config,utils}.py`, `fmri_rt_preproc/prep_surface_rois.py` | `decnef.preproc` (offline preparation; tool paths, conda env and FreeSurfer license become config) |
| `fmri_rt_preproc/PyHySCO-main/` | a PyHySCO dependency (`decnef.unwarp` still imports from this tree) |
| `fmri_rt_preproc/RTPSpy_tools/afni_module/` (C sources) | `librtp` build step; committed `.so`/`.dylib` removed |
| `rs_pca_runtime.py`, `roi_rs_pca_decoder_prep.py`, `rs_pca_score_all_rois.py`, `init_pca_conditions.py` | `decnef.plugins.rs_pca` (optional plugin) |
| `rt_psychopy_parallel.py` | `examples/` (score plot + condition paradigm; generic parts already in `decnef.display`) |
| `biopac_rt/biopac_streamer.py`, `biopac_rt/biopac_receiver_tester.py` | BIOPAC-PC side: a `tools/` folder or its own small package |

### Study-specific or one-off → out of the repository (study repo or gitignored `bkup/`)

`rt_robot_parallel.py`, `robot_practice.py`, `rt_brainpong_parallel.py`,
`brainpong_adapter.py`, `brainpong_settings.json`, `task/Minjoo/`,
`rs_realtime_parallel.py`, `plot_*.py`, `scores_filtered.py`, `scorer_plots.py`,
`compare_decoders.py`, `compare_roi.py`, `compare_tsnr_full.py`,
`analyze_motion_threshold.py`, `streamer_debugger.py`,
`biopac_rt/fft_physio_check.py`, `fmri_rt_preproc/t2_spm_afni_closed_loop.py`,
the personal settings JSONs (`decnef config migrate` converts the ones to keep),
`minjoo sequence*`, `testOlga.txt`, and the data `data/sub-*`, `decoders/`
(data-hygiene phase; history scrubbing is coordinated with the lab).

### Folded into `docs/`

`readme_old.txt` (preparation, data layout, config.json sections), `PLAN.md`,
`docs/parallel_compute_commit_design.md`, `docs/unwarp_before_mc.md`.
