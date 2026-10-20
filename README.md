# decnef

Real-time fMRI decoded neurofeedback: a pipeline that takes each incoming
DICOM volume through unwarping, motion correction, quality metrics, nuisance
regression, resampling to a decoder's space, optional masking/smoothing and
decoder scoring, within the TR, and hands the score to a display. This is
the clean rewrite of [K-Demin/DecNef_py](https://github.com/K-Demin/DecNef_py)
(the original flat scripts are still in the repository as the reference
implementation while the equivalence check is completed; see `README_ELB.md`).

```
src/decnef/
  config.py       Settings: one YAML/JSON file, validated, no hidden defaults   docs/config.md
  layout.py       every path of a subject/day/run
  session.py      start_session(): pipeline + physio + extras as child processes, ScoreEvents to the display   docs/session_api.md
  pipeline.py     run_pipeline(): prerequisites, provenance, the per-volume stages   docs/architecture.md
  stages.py       unwarp, motion+FD, censor+DVARS, regression, transform, smooth, score
  engine.py       ordered compute/commit engine
  ingest.py       DICOM names, dcm2niix, incoming-folder watcher
  decoder.py      decoder file formats and scoring
  physio.py       BIOPAC RetroTS receiver / file buffer
  calibration.py, masks.py, smoothing.py, resampler.py, unwarp.py, viewer.py
  rtpspy/         vendored RTPSpy volreg / regress (+ librtp)
  display.py      optional PsychoPy helpers
  cli.py          decnef run | replay | compare | config
examples/nf_events_launcher.py   a complete paradigm on the session API
```

## Install

```bash
conda env create -f environment.yml && conda activate decnef
pip install -e . --no-deps             # editable; the package is `decnef`
pip install -e ".[task]"               # PsychoPy, for launchers / examples
pytest tests/decnef                    # the package's tests (tests/ also still covers the old scripts)
```

External tools on `PATH`: `dcm2niix`, ANTs (`antsApplyTransforms`), and for
the offline preparation FSL, FreeSurfer (SynthStrip/SynthMorph), FastSurfer.
`torch` falls back to CPU everywhere; the real-time machine should have the
CUDA build.

## Use

1. Write a config file (`docs/config.md`); `decnef config migrate old_settings.json --base-data ... --decoder-template ... --out decnef.yaml` translates an old `rt_settings.json`.
2. Prepare the subject/day offline (reference EPI, transforms, masks, fieldmap calibration — still the old `run_preproc.py` until the preparation phase).
3. Run:

```bash
decnef run --config decnef.yaml --sub 00086 --day 3 --run 4 --incoming /path/to/dicoms      # pipeline only
python examples/nf_events_launcher.py --config decnef.yaml --sub 00086 --day 3 --run 4       # with a display
```

A launcher is a few lines around `start_session` (`docs/session_api.md`):

```python
from decnef.config import load_settings
from decnef.session import SessionConfig, start_session

settings = load_settings("decnef.yaml")
with start_session(SessionConfig(settings, sub="00086", day="3", run="4")) as session:
    for event in session.scores(timeout=0.02):
        if event.kind == "score" and event.reg_ready:
            show(event.score_raw, event.stage)
        if finished:
            break
```

## Equivalence with the original

`decnef replay` copies a recorded run's DICOMs into a temporary incoming
folder at the TR and runs the pipeline headless; `decnef compare` diffs the
resulting CSVs against a reference produced by the original scripts
(`replay_run.py --driver legacy` on branch `cleanup/01b-session-api`). Numeric
outputs must be identical; the step-by-step recipe is `docs/replay_check.md`.

## What is not in the package (yet)

Offline preparation (`run_preproc.py`, `fmri_rt_preproc/`), the resting-state
PCA decoder (`rs_pca_runtime.py` and friends; planned as `decnef.plugins.rs_pca`),
PyHySCO as a proper dependency (still imported from the vendored tree), and
the study-specific tasks. They run from the repository root as before.
