# Architecture

## Processes

```text
launcher (main process: the display; consumes ScoreEvents from session.scores()/poll())
├── pipeline        decnef.session._pipeline_main -> decnef.pipeline.run_pipeline(settings, layout, queue, stop_event)
│   ├── watchdog observer thread      incoming-folder events
│   ├── thread pool (engine)          dcm2niix per volume, any order
│   ├── ordered commit                unwarp -> MC -> FD/DVARS -> regression -> transform -> smooth -> score, scan order
│   ├── BIOPAC receiver thread        physio.mode=tcp, in-process
│   └── (daemon) QC viewer            qc.volume_viewer
├── physio listener                   physio.listener (receiver writing the CSV the pipeline tails)
└── extras                            ProcessSpec per extra process (e.g. a scorer plugin)
```

All children use the `spawn` start method. `Session.stop()` stops extras,
then the pipeline, then physio: stop event -> join -> `terminate()` -> `kill()`;
the pipeline leaves its poll loop when the event is set so its own cleanup
runs (`run_end` to BIOPAC, executor shutdown, viewer stop).

## Run start (`decnef.pipeline.run_pipeline`)

1. `check_prerequisites`: tools on PATH (`dcm2niix`, `antsApplyTransforms` unless the GPU resampler is on), incoming folder, reference EPI `func/trans/epi_unwarped_mean.nii`, fieldmap calibration `fmap[/pair-*]/native_unwarp_v1/`, decoder. Fails here, not at volume 1.
2. `validate_run_provenance`: `preprocessing_order.json` records the processing order, calibration, reference, phase encoding and final-mask provenance; a run folder whose marker differs, or that already holds processed volumes, is refused.
3. `build_state`: loads the reference, the PyHySCO field (cached applier), RtpVolreg with the reference, the DVARS mask, physio (receiver or file buffer), RtpRegress with the nuisance masks, the GPU sampling grid (cached under `func/trans/gpu_grids/`), the final-mask smoother, the decoder, the stage schedule, the z-score reference stats.
4. `session_metadata.json` is written (pipeline keys + the full settings tree under `settings`; a launcher's own keys are merged before).
5. Volumes of this run already in the incoming folder are processed first, then the watcher goes online (the BIOPAC `run_start` is sent on the first online volume).

## Per-volume stages (`decnef.stages`)

| stage | reads | writes |
|---|---|---|
| compute: `ingest.convert_dicom` | DICOM `SSS_RRRRRR_VVVVVV.dcm` | `raw/vol_<v>.nii` (+ dcm2niix JSON) |
| `unwarp_stage` | raw, calibration (grid/polarity/readout checks) | `unwarped/vol_<v>_uw_native.nii` |
| `motion_stage` | unwarped, RtpVolreg (reference = unwarped mean) | `mc/vol_<v>_mc.nii`, `motion_rt.1D`, `fd_rt.csv` (NaN while `v <= skip_first_trs`) |
| `censor_stage` | mc, DVARS mask, history | `dvars_rt.csv`; FD/DVARS flags with +1 TR |
| `regression_stage` | mc, RtpRegress (motion, WM, Vent, GS, physio) or voxel normalisation | `reg/vol_<v>_reg.nii`, `regressors_rt.csv` |
| `status_stage` | | `regression_status_rt.csv`, `biopac_timelag.csv` |
| `transform_stage` | reg (+ mc for the original score), `epi2t1_Composite.h5`, `anat/warp_T1_to_MNI_synth.nii`, decoder grid | `t1/` or `mni/vol_<v>_<space>[_orig].nii` (GPU grid or antsApplyTransforms, linear) |
| `smooth_stage` | final mask | `smooth/` or `masked/vol_<v>_<kind>[_orig].nii`, `<kind>/mask.nii` |
| `score_stage` | decoder, stage schedule, reference stats | `scores.csv`, `scores_z.csv`; the queue message |

A stage that cannot do its job raises; the engine records the volume as
failed and the queue receives `kind="error"` for it. Nothing substitutes
zeros or an un-regressed volume.

### Output files

| file | columns |
|---|---|
| `scores.csv` | `volume_idx, timestamp, score_raw, score_original, reg_ready, event_type` |
| `scores_z.csv` | `volume_idx, timestamp, score_raw, score_z, ref_run, ref_mean, ref_std, ref_n, ref_used_reg_ready, reg_ready` |
| `motion_rt.1D` | roll pitch yaw dS dL dP (no header) |
| `fd_rt.csv` | `volume_idx, fd` |
| `dvars_rt.csv` | `volume_idx, fd, dvars, robust_z` |
| `regressors_rt.csv` | `volume_idx, <regressor names>` |
| `regression_status_rt.csv` | `volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, reg_ready, biopac_missing` |
| `biopac_timelag.csv`, `biopac_regressors_rx.csv` | physio timing / received regressors |
| `session_metadata.json`, `preprocessing_order.json` | metadata, provenance |

Timestamp columns are wall-clock; everything else is deterministic for a
given input, device and torch build.

### Decoder files (`decnef.decoder`)

ROI text (line 1 `nx ny nz`, line 2 `N`, N lines `x y z weight template`
with 1-based indices, then `0 0 0 bias 0`), or a NIfTI whose non-zero,
non-NaN voxels are the weights (bias 0). The decoder must be on the scored
volume's grid; a mismatch is an error.

## Engine (`decnef.engine.OrderedEngine`)

Compute runs in a pool of `pipeline.engine.max_workers` threads; commits run
strictly in scan order, serialised, in whichever worker finished last. A scan
the watcher has seen but that never arrives is skipped after
`commit_wait_timeout_s` once a newer scan is ready; late results for an
already-committed scan are dropped.

## Replay (equivalence check)

```bash
# reference from the original scripts (branch cleanup/01b-session-api, lab machine)
python replay_run.py --dicom-dir REC --run 4 --tr 1.4 --driver legacy --sub S --day D --base-data B --replay-run 901 \
    --out /tmp/ref --decoder-template DEC --settings-file old_settings.json
# the package
decnef replay --config decnef.yaml --sub S --day D --dicom-dir REC --run 4 --replay-run 902 --out /tmp/new
decnef compare /tmp/ref /tmp/new          # IDENTICAL (timestamp columns ignored) is the pass condition
```

Both need `dcm2niix`, the offline preparation for that subject/day, a fresh
run id each, and the same GPU for both runs. The replay copies DICOMs in scan
order at the TR (staged on the same filesystem, moved in with `os.replace`),
with physio and the viewer off.
