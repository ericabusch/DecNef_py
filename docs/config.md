# Configuration

One file per study (YAML or JSON), loaded with `decnef.config.load_settings`
or `--config` on the command line. Every setting lives in it; there is no
module global and no hidden default for a path or a decoder — a missing
required value is an error at load time. `--set key.path=value` overrides any
field; `decnef config show --config f.yaml` prints the effective tree;
`decnef config migrate old.json` translates an old `rt_settings.json` (its
`_comment_*` keys are skipped, `pipeline_engine` is dropped: only the ordered
engine exists).

Simulation parameters (`decnef.simulate.SimulationConfig`: level, number of
volumes, phantom geometry, noise, injected signal, output directory) are not
config keys: they are passed to `start_session(cfg, simulate=...)` or as
`decnef simulate` flags, so a study config never carries test settings.

```yaml
paths:
  base_data: /data/decnef          # required. sub-<sub>/<day>/... lives here       (old --base-data)
  incoming_root: /incoming         # optional here, else --incoming on the CLI       (old --incoming-root)

decoder:
  template: decoders/d.nii         # required when scoring (and for t1/mni spaces: it is the resampling reference)
  roi_txt: null                    # ATR-style ROI_DECODER.txt; preferred over the NIfTI when given
  reference_score_run: null        # run whose scores.csv gives the z-score mean/std   (old --rs)
  enable_scoring: true
  enable_original_score: false     # also score the un-regressed volume -> score_original

pipeline:
  TR: 1.4                          # required
  analysis_space: mni              # epi | t1 | mni
  voxel_norm_ref_volumes: 20       # warm-up window: regression readiness / voxel normalisation
  skip_first_trs: 0                # volumes excluded from FD/censoring and labelled pre_label (one setting for everything)
  smoothing_fwhm_mm: 0.0
  motion:
    enable_regression: true        # (old enable_motion_regression)
    mot_reg: mot6
    max_poly_order: inf
    brain_radius_mm: 50.0          # FD radius (was hardcoded)
    volreg_mode: heptic            # RtpVolreg regmode (was hardcoded)
  censor:
    fd:    {enable: true, thr: 0.3}                                   # (old enable_fd_censor_reg, fd_thr)
    dvars: {enable: true, thr_robust_z: 3.0, warmup: 20, mask_source: ref_mask}
    plus1: true                    # also censor the volume after a spike (old censor_plus1)
  nuisance: {use_gs: false, use_wm: true, use_vent: true}
  mask:
    analysis_mask: whole_brain     # whole_brain | cortical_gm | custom
    file: null                     # custom mask (old analysis_mask_file)
    space: final                   # final | epi | t1 | mni (old analysis_mask_space)
  fieldmap:
    method: pyhysco
    epi_phase_encoding: PA         # AP | PA
    preloaded: true                # cache the field once per run (old use_preloaded_pyhysco)
    device: cuda                   # CPU fallback when unavailable
    backend: grid_sample           # grid_sample | gather1d
    save_intermediate_unwarped: true
  resampler: {use_gpu: false, device: cuda}     # GPU warp to the decoder grid instead of antsApplyTransforms
  engine:
    max_workers: 6                 # dcm2niix threads (<= 10)
    commit_wait_timeout_s: 1.0     # skip a scan that was seen but never arrived, once a newer one is ready
    max_retries: 3                 # dcm2niix retries
    rt_max_scan_length: 2000
  truncate_t1_to_epi_fov: true     # used by the rs_pca plugin's T1 reference
  truncate_t1_padding_vox: 2

physio:                            # BIOPAC RetroTS regressors (old biopac_*; enable was true by default, now false)
  enable: false
  host: 0.0.0.0                    # no lab default any more
  port: 15000
  timeout: 0.3                     # seconds to wait for a volume's regressors before zero-filling
  phys_reg: RICOR8                 # RICOR8 | RVT5 | RVT+RICOR13
  handshake: true
  start_online_only: false
  mode: tcp                        # tcp | file
  file: null                       # CSV for mode=file (written by the listener process)
  poll_interval: 0.05
  listener: false                  # run the receiver in its own process (requires enable and mode=file)
  timelag: false                   # write biopac_timelag.csv

qc:
  volume_viewer: false             # live matplotlib viewer in a daemon process (old enable_volume_streamer)
  stream_kind: mc                  # raw | mc | reg | score_input (old 'unwarped' alias = mc)
  every_n: 1
  max_queue: 2
  window_title: RT volume QC

stages:                            # optional: trial-stage labels for scores.csv / ScoreEvent.stage
  baseline_trs: 20                 # after pipeline.skip_first_trs
  cycle: [[iti, 3], [cue, 4], [scans, 3], [delay, 3], [feedback, 3]]
  n_cycles: 20                     # null = repeat forever
  pre_label: background
  baseline_label: background
  post_label: post_task
```

Validation: choices are checked (and case-folded where the old code did),
`TR > 0`, `smoothing_fwhm_mm >= 0`, `voxel_norm_ref_volumes >= 1`,
`mask.file` required for `custom`, `physio.listener` requires `enable` and
`mode=file`, unknown keys are errors.

Values can also be built in Python: `decnef.config.from_dict({...})`, and a
launcher may set `settings.stages` / `settings.pipeline.voxel_norm_ref_volumes`
before `start_session` (see `examples/nf_events_launcher.py`).
