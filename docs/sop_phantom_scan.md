# SOP — a live phantom scan with `decnef run`

The first test of the package on scanner: real DICOM arrival,
the scanner's file-writing pattern, the lab machine's GPU, in real time. A
phantom has no anatomy, so the run is in **EPI space** (no T1, no FastSurfer,
no WM/ventricle regressors); everything else — DICOM watch, unwarping, motion
correction, censoring, regression, scoring, events — is the production path.
Subject id below: `phantom`, day `1`. Nothing here touches the lab's
`/SSD2/DecNef_py/data`.

Time at the scanner: ~15 min of sequences plus ~2 min of preparation between
the transfer run and the NF run.

---

## 0. Before scan day (lab machine, 10 min)

```bash
cd /SSD2/erica/DecNef_py && git pull origin master
conda activate decnef_elb
python -m pytest tests/decnef -q                     # all green, incl. test_prepare.py (needs dcm2niix)
nvidia-smi                                            # the GPU is free
mkdir -p /SSD2/erica/phantom_data
decnef config show --config configs/phantom.yaml      # the config loads; check base_data and decoder.template
```

`configs/phantom.yaml` is ready to use. What it sets, and why:

| key | value | reason |
|---|---|---|
| `paths.base_data` | `/SSD2/erica/phantom_data` | your folder; the preparation and the run write here |
| `pipeline.analysis_space` | `epi` | no T1/MNI transforms for a phantom |
| `pipeline.nuisance.use_wm/use_vent` | `false` | no segmentation masks (the pipeline would warn and skip them anyway) |
| `pipeline.voxel_norm_ref_volumes` / `skip_first_trs` | 30 / 10 | as the lab's `rt.yaml`; regression becomes ready at volume 30 |
| `pipeline.fieldmap.epi_phase_encoding` | `PA` | the BOLD's phase-encoding polarity (same as the lab protocol; change if the phantom protocol differs) |
| `decoder.template` | `…/sub-phantom/1/func/trans/phantom_decoder.nii` | written by `prepare-epi`: random weights inside the phantom mask, so the scoring stage runs end to end. Set `decoder.enable_scoring: false` to run without any decoder (`tick` events) |
| `physio.enable` | `false` | no BIOPAC |
| `qc.volume_viewer` | `true` | a live window of each corrected volume |

Optional dry run of the whole procedure with synthetic data, before going anywhere near the scanner:

```bash
decnef simulate --config configs/phantom.yaml --sub sim --day 1 --run 1 --level scanner --n-volumes 60 --out /SSD2/erica/sim
```

---

## 1. At the scanner: the protocol

Run, in this order (series numbers are what the scanner assigns; write them down):

| # | sequence | purpose | notes |
|---|---|---|---|
| 1 | localizer | — | |
| 2 | **AP fieldmap** (spin-echo EPI, AP) | PyHySCO calibration | same geometry as the BOLD |
| 3 | **PA fieldmap** (PA) | PyHySCO calibration | |
| 4 | **transfer EPI**, ≥ 30 volumes | reference image + mask | the protocol's BOLD sequence; `prepare-epi` keeps scans 11–30 |
| 5 | **NF run**, 300–600 volumes | the live test | the protocol's BOLD sequence, same TR as `pipeline.TR` |
| 6 | (optional) a second NF run | repeat with a different run id | |

The AP/PA pair acquired just before the NF run is the pair to use
(`--ap-block 2 --pa-block 3` for the numbers above). The DICOMs arrive in
the session's incoming folder, e.g.
`/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/<date>.<date>_phantom.<name>/`,
as `SSS_RRRRRR_VVVVVV.dcm` (series, run, volume). Find it with
`ls -t /home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/ | head -1`.

```bash
INC=/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/<today's folder>
ls $INC | cut -d_ -f2 | sort | uniq -c            # one line per series with its file count; confirm 2, 3, 4
```

---

## 2. After the transfer run: prepare (lab machine, ~1–2 min)

```bash
decnef prepare-epi --config configs/phantom.yaml --sub phantom --day 1 \
    --incoming $INC --ap-block 2 --pa-block 3 --epi-block 4
```

What it does (all under `/SSD2/erica/phantom_data/sub-phantom/1/`):

1. converts series 2 and 3 → `fmap/pair-ap002_pa003/AP.nii.gz`, `PA.nii.gz` (+ dcm2niix sidecars);
2. builds the PyHySCO calibration → `fmap/pair-ap002_pa003/native_unwarp_v1/` (seconds on the GPU);
3. converts scans 11–30 of series 4, unwarps them, motion-corrects them to the
   first corrected volume, averages → `func/trans/epi_unwarped_mean.nii` (the
   motion/analysis reference), `epi_mc.nii`, `motion.1D`;
4. thresholds the mean → `func/trans/epi_mask_mean.nii` (phantom = everything bright);
5. writes `func/trans/phantom_decoder.nii` (random weights inside the mask).

It prints the paths and the `decnef run` command to use next. Check:

```bash
ls /SSD2/erica/phantom_data/sub-phantom/1/func/trans/        # epi_unwarped_mean.nii epi_mask_mean.nii phantom_decoder.nii ...
python -c "import nibabel as nib,numpy as np; m=nib.load('/SSD2/erica/phantom_data/sub-phantom/1/func/trans/epi_mask_mean.nii'); print(m.shape, int(np.asarray(m.dataobj).sum()),'mask voxels')"
```

A mask of a few thousand voxels for a 3 mm phantom is right; if it is nearly
empty or nearly the whole volume, rerun with `--mask-threshold 0.3` or `0.7`
after deleting `func/trans/`. If the fieldmap pair looks wrong (`AP.json` /
`PA.json` phase-encoding directions are not opposite), stop here.

---

## 3. Before the NF run starts: launch the pipeline

**Start the pipeline before the scanner starts the NF run** (it waits for the
first DICOM; a run started mid-acquisition is handled too, but start first).
The run id is the NF series number — the scanner assigns it when the series
starts, so take the next number after the last series you see (series 5 for
the table above; if a scout or anything else gets inserted, adjust).

```bash
decnef run --config configs/phantom.yaml --sub phantom --day 1 --ap-block 2 --pa-block 3 \
    --incoming $INC --run 5
```

Expected within a few seconds: `[FMAP] Preloaded…`, `RtpVolreg … Alignment reference = …epi_unwarped_mean.nii`,
`[GPU] resampler` is **not** printed (EPI space), `[SCORE] decoder: …phantom_decoder.nii`,
`[RT] online: watching <INC> for run 5`, and the QC window. Then tell the
operator to start series 5. From there, one block per volume:

```
[FMAP] vol 00001  (0.01s)
[MC] vol 00001 FD=NaN (skipped TR)          # volumes <= skip_first_trs have no FD
[REG] vol 00001 motion                       # "Wait until volume #30" until the regression is ready
[TRANS] vol 00001 skipped (EPI space)
[SCORE] vol 00001 raw=...
```

Watch for: FD values ≈ 0.0x–0.2 (a phantom barely moves; large FD means
geometry or reference problems), `reg_ready` turning on at volume 30 (`[REG] vol 00030 regression`),
the QC window updating every TR, and no `[COMMIT] volume … failed` lines.

Stop: **Ctrl-C** once the series is over (the pipeline exits its loop and
cleans up), or leave it until the next command.

### With the display instead (optional)

```bash
pip install -e ".[task]"                                      # once; PsychoPy
python examples/nf_events_launcher.py --config configs/phantom.yaml --sub phantom --day 1 --ap-block 2 --pa-block 3 \
    --incoming $INC --run 5 --n-trials 20 --baseline-trs 30
```

`s` on the scanner trigger (or by hand), `q`/Esc to end. Feedback circles
driven by a random decoder on a phantom are noise — the point is the timing
and the plumbing, not the values.

---

## 4. After the run: what to look at

```bash
RUN=/SSD2/erica/phantom_data/sub-phantom/1/func/5
wc -l $RUN/scores.csv $RUN/fd_rt.csv $RUN/motion_rt.1D          # = volumes acquired (+1 header for the CSVs)
ls $RUN/raw | wc -l; ls $RUN/mc | wc -l; ls $RUN/reg | wc -l   # same counts
python - <<'EOF'
import pandas as pd
r = "/SSD2/erica/phantom_data/sub-phantom/1/func/5"
s = pd.read_csv(f"{r}/scores.csv"); st = pd.read_csv(f"{r}/regression_status_rt.csv")
lat = s.timestamp - pd.read_csv(f"{r}/scores.csv").timestamp.min()
print("volumes:", len(s), " reg_ready from:", int(s.volume_idx[s.reg_ready == 1].min()))
print("censored (fd):", int(st.fd_censor.sum()), " (dvars):", int(st.dvars_censor.sum()))
EOF
grep -c "failed" ~/.decnef_run.log 2>/dev/null; true
```

Pass criteria for the live test:

- every acquired volume has a row in `scores.csv` (no gaps in `volume_idx`), no `error` events;
- per-volume processing time (the `(x.xxs)` in the log) stays well under the TR;
- the first volume was picked up as soon as it appeared (the `[RT] online` line precedes the first `[FMAP]`);
- `reg_ready` from volume 30; FD small and stable; `scores_raw` after volume 30 fluctuating around 0.

Then the equivalence check on the recording, exactly as for sub-99988 — copy
`examples/replay_99988.args` and `legacy_99988.args` to `phantom.args`, set
`--config configs/phantom.yaml --sub phantom --day 1 --ap-block 2 --pa-block 3 --dicom-dir $INC --run 5`
and `--base-data /SSD2/erica/phantom_replay/data` for the legacy one, then
`decnef compare`. (The legacy driver needs the old `*_settings.json` form:
`decnef config show --config configs/phantom.yaml` for the values, or run the
package replay only and compare it with the live run's `scores.csv`, which
must be identical — same DICOMs, same calibration, same reference.)

---

## 5. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `prepare-epi`: `no DICOMs for series N` | wrong series number or wrong folder | `ls $INC \| cut -d_ -f2 \| sort \| uniq -c` |
| `BOLD phase encoding does not match the selected calibration polarity` | `fieldmap.epi_phase_encoding` (PA) disagrees with the BOLD sidecar | set `--set pipeline.fieldmap.epi_phase_encoding=AP` (both for prepare and run) |
| `Raw BOLD/calibration/reference grids differ` | the NF sequence does not match the transfer/fieldmap geometry | same sequence for 2–5; re-run the transfer if the protocol changed |
| `fieldmap calibration missing … no raw AP/PA` | `--ap-block/--pa-block` differ from the prepared pair | use the same numbers as in `prepare-epi` |
| `decnef run` prints `online` but no volumes appear | wrong `--run` (series number) or wrong `--incoming` | watch `ls $INC \| tail` while the scan runs; the run field in the new file names is the one to use; restart with it — volumes already written are picked up |
| `decoder template not found` | `prepare-epi` run with `--decoder none`, or a different `base_data` | `--set decoder.enable_scoring=false`, or point `decoder.template` at the file |
| volumes arrive but processing lags behind the TR | `max_workers` / GPU contention | check `nvidia-smi`; `--set pipeline.engine.max_workers=2` |
| `dcm2niix produced no output` | a truncated DICOM (scanner still writing) | the pipeline retries; a single volume may be reported as `error` and the run continues |
| QC window does not open | no display / Qt backend | `--set qc.volume_viewer=false`; it never affects the run |

Logs: everything is on stderr; `decnef run ... 2>&1 | tee /SSD2/erica/phantom_data/run5.log` keeps a copy.


Step 1 — the series listing (ls $INC | cut -d_ -f2 | sort | uniq -c, run after the transfer EPI finishes):

     ```
     11 000001      localizer
      4 000002      AP fieldmap     ← --ap-block 2
      4 000003      PA fieldmap     ← --pa-block 3
     40 000004      transfer EPI    ← --epi-block 4   (≥ 30 files)
     ```

Sensible: the two fieldmap series have the same small count (the lab's have 4 files each), the EPI count equals the volumes you asked for, and the numbers are consecutive. A stray series (a scout repeated, an aborted run) just shifts the numbers — use what you see, not what the protocol sheet says.

Step 2 — decnef prepare-epi (~1–2 min):

```
[PREP] series 2 (4 DICOMs) -> .../sub-phantom/1/fmap/pair-ap002_pa003/AP.nii.gz
[PREP] series 3 (4 DICOMs) -> .../PA.nii.gz
[FMAP] AP/PA metadata missing: using legacy voxel phase-encoding axis 2      (expected: the lab's fieldmaps have no sidecar PE field)
[FMAP] PyHySCO estimation in .../native_unwarp_v1 (max_iter=50)
Iteration  Loss Value ...
0          4.0E+08
...
reached function value change tolerance        total runtime: 0 min 0.2 sec
[FMAP] native calibration built: .../native_unwarp_v1
[PREP] transfer series 4: 20 volumes -> .../func/trans/transfer_raw
RtpVolreg: Alignment reference = .../transfer_first_unwarped.nii
[PREP] reference .../epi_unwarped_mean.nii from 20 volumes; mask 6812 voxels
[PREP] stand-in decoder .../phantom_decoder.nii
{ "fmap_dir": ..., "reference": ..., "mask": ..., "decoder": ... }
next: decnef run --config configs/phantom.yaml --sub phantom --day 1 --ap-block 2 --pa-block 3 --incoming ... --run <NF series> --set decoder.template=...
```

Sensible: PyHySCO loss drops by an order of magnitude and stops after ~10–20 iterations (a phantom has little distortion; a loss that doesn't decrease, or 50 iterations without convergence, means the AP/PA pair is wrong or identical). Mask voxels: a 3 mm phantom of ~15 cm diameter is ~65 000 voxels; a few thousand to tens of thousands is fine, under ~500 or more than half the volume means the threshold is off. motion.1D in func/trans/ should be near zero (phantom).

Step 3 — decnef run, before the series starts:
```
[PyHySCO:init] prototype_image_shape=(140, 140, 68) fieldmap_image_shape=(140, 141, 68) ...
RtpVolreg: Alignment reference = .../func/trans/epi_unwarped_mean.nii[0]
RtpRegress: Mask = .../func/trans/epi_mask_mean.nii
[SCORE] decoder: .../phantom_decoder.nii
[RT] online: watching <INC> for run 
```

Then silence until the operator starts series 5 — that gap is the "pipeline started before the scanner" condition. From the first DICOM, once per TR:
```
[FMAP] vol 00001  (0.01s)
[MC] vol 00001 FD=NaN (skipped TR) (0.07s)
[REG] vol 00001 motion (0.36s)                 ← first volume only: RtpRegress initialisation
[TRANS] vol 00001 skipped (EPI space) (0.00s)
[SCORE] vol 00001 raw=-6.98e+07 (0.00s)       ← pre-reg_ready scores are meaningless, large, and that's expected
...
[MC] vol 00011 FD=0.02 (0.09s)                 ← FD appears after skip_first_trs; phantom: 0.00–0.1
RtpRegress: #11:Wait until volume #30
...
RtpRegress: #30;Regression;vol_00030_mc.nii
[SCORE] vol 00031 raw=-9.2 (0.00s)             ← from here scores are O(1–1000) and fluctuate around 0
```
Sensible: every stage time ≤ 0.1 s (sum ≈ 0.2 s against a 1.0 s TR); volume numbers increase by exactly one each TR with no [COMMIT] volume … failed or error; FD tiny; the QC window redraws each TR; no line of the form vol 000NN appearing more than ~0.3 s after the file landed (the watchdog_timestamp → timestamp gap in scores.csv, checked in step 4). If a volume number is skipped, that is the finding we are there to make — note the time and keep the run going.

Step 4 — the check script on func/5/:

```volumes: 600   reg_ready from: 30
censored (fd): 0   (dvars): 0–3
```
plus `wc -l`: 601 lines in scores.csv/fd_rt.csv, 600 in motion_rt.1D, 600 files in each of raw/ unwarped/ mc/ reg/. A phantom should produce 0 FD censors; a handful of DVARS censors in the first volumes after warm-up is normal (scanner settling). The latency column: timestamp − watchdog_timestamp per volume should sit around 0.15–0.25 s with no upward trend — a trend means the pipeline is falling behind.

And the final pass condition, after the scan: decnef replay of series 5 against the same prepared data produces a scores.csv identical to the live one. That closes the loop: live processing ≡ offline replay ≡ (via the earlier check) the original scripts.

One thing I noticed: docs/sop_phantom_scan.md has a one-line local edit on the scanner-side copy or here — if you changed something on purpose, commit it; if not, git diff docs/sop_phantom_scan.md will show what moved.