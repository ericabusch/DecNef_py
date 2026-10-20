# README_ELB — fork cleanup log

This file tracks what I (ELB) am changing in this fork of
[K-Demin/DecNef_py](https://github.com/K-Demin/DecNef_py) and why. The
upstream `readme.txt` documents the pipeline itself; this file documents the
packaging/cleanup work, the decisions behind it, and what is still open.
The phase plan is in `CLAUDE.md`.

## Phase status

| phase | branch | status |
|---|---|---|
| 1. Packaging (`pyproject.toml`, `environment.yml`, editable install, tests) | `cleanup/01-packaging` | in progress |
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

## Test record

| run | command | result |
|---|---|---|
| baseline (before packaging, env `decnef-baseline`) | `pytest tests/` | 6 collection errors (top-level modules not importable) |
| baseline | `python -m pytest tests/` | 73 passed |
| after packaging (env `decnef` from `environment.yml`, `pip install -e . --no-deps`) | `pytest tests/` and `python -m pytest tests/` | 73 passed, per-test identical to baseline |

Verification details (phase 1): `pip check` clean; the pip step added only
`decnef` to the conda env; setuptools used its import-finder editable mode
(`__editable__.decnef-0.1.0.pth`), which resolves the namespace package
`fmri_rt_preproc` to the repo and keeps every `__file__` inside the repo
(`librtp.dylib` loads from `fmri_rt_preproc/RTPSpy_tools/`). From a directory
outside the repo, all 14 listed modules and all `fmri_rt_preproc.*`,
`biopac_rt.*` modules import; the excluded study/one-off scripts are not
importable from outside the repo by design (they still run from the repo
root with `python <script>.py`).
