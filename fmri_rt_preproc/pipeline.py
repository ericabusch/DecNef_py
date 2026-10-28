import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fmri_rt_preproc_pipeline")

from fmri_rt_preproc.config import SubjectDayConfig
from fmri_rt_preproc.utils import run, ensure_dir, run_in_conda
from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from pathlib import Path
from nilearn import plotting, image
import nibabel as nib
import numpy as np
import gzip
import shutil
import os
import sys
import json
from fmri_rt_preproc.native_fieldmap import (
    PIPELINE_ORDER, native_paths, calibration_spec, load_calibration,
    save_native_mean, require_same_grid, validate_bold, source_identity,
)

from fmri_rt_preproc.pyhysco_apply import apply_pyhysco_fieldmap

def gunzip_python(gz_path):
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")  # removes .gz → gives .nii

    if out_path.exists() and out_path.stat().st_mtime >= gz_path.stat().st_mtime:
        return out_path

    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


def prefer_uncompressed_nifti(path_nii: Path) -> Path:
    """Prefer .nii, but fall back to .nii.gz for backward compatibility."""
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    if gz.exists():
        return gz
    return path_nii

class FMRIRealtimePreprocessor:
    def __init__(self, cfg: SubjectDayConfig,
                 fastsurfer_env: str = "fastsurfer",
                 ants_env: str = "ants_env"):
        self.cfg = cfg
        self.anat_dir = cfg.subject_root / "anat"
        self.fmap_dir = Path(getattr(cfg, "fieldmap_dir", None) or (cfg.root / "fmap"))
        self.func_dir = cfg.root / "func"
        self.decoder_template = (cfg.root / cfg.decoder_template).resolve()
        self.fastsurfer_env = fastsurfer_env
        self.ants_env = ants_env
        self.trans_dir = self.func_dir / "trans"
        self.fastsurfer_sid = self.cfg.subject_id
        self.fieldmap_method = str(getattr(cfg, "fieldmap_method", "pyhysco")).lower()
        self.epi_phase_encoding = str(getattr(cfg, "epi_phase_encoding", "PA")).upper()

        # Day-level references made from the first preprocessed run.
        self.rt_distorted_motion_ref_epi = self.trans_dir / "rt_ref_epi.nii"
        self.rt_distorted_motion_ref_mask = self.trans_dir / "rt_ref_epi_mask.nii"
        self.rt_unwarped_analysis_ref_epi = prefer_uncompressed_nifti(self.trans_dir / "epi_unwarped_mean.nii")
        self.rt_unwarped_analysis_ref_mask = prefer_uncompressed_nifti(
            self.trans_dir / "epi_mask_mean.nii"
        )

    # ---------- Top-level entry points ----------

    def run_all(self):
        ensure_dir(self.anat_dir)
        ensure_dir(self.func_dir)
        ensure_dir(self.trans_dir)
        self._prepare_anat()
        for run_cfg in self.cfg.runs:
            self._prepare_run(run_cfg)

    # ---------- T1 / anatomical ----------

    def _prepare_anat(self):
        ensure_dir(self.anat_dir)

        # ---------- 1. N4 BIAS CORRECTION ----------
        t1 = self.cfg.t1_file
        t1_n4 = self.anat_dir / "T1_N4.nii"

        if t1_n4.exists():
            print("✓ N4: already exists — skipping")
        else:
            print("→ Running N4")
            run(["N4BiasFieldCorrection", "-d", "3", "-i", str(t1), "-o", str(t1_n4)])

        # ---------- 2. FASTSURFER ----------
        fastsurfer_dir = self.anat_dir / "fastsurfer"
        ensure_dir(fastsurfer_dir)

        aparc_aseg = (
            fastsurfer_dir / self.fastsurfer_sid / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        )

        if aparc_aseg.exists():
            print("✓ FastSurfer: already exists — skipping")
        else:
            print("→ Running FastSurfer")
            self._run_fastsurfer(t1, fastsurfer_dir)

        # ---------- 3. BRAIN MASKS ----------
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        if t1_combined.exists():
            print("✓ Brain masks: already exist — skipping")
        else:
            print("→ Creating brain masks")
            self._make_brain_masks(t1_n4, fastsurfer_dir)

        # ---------- 4. SYNTHMORPH ----------
        warp = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        warped = self.anat_dir / "T1_warped_to_MNI_synth.nii"

        if warp.exists() and warped.exists():
            print("✓ SynthMorph: already exists — skipping")
        else:
            print("→ Running SynthMorph")
            self._run_synthmorph(t1_n4)

    def _run_fastsurfer(self, t1_n4: Path, fs_base_dir: Path):
        """
        Run FastSurfer in a separate conda env, once per subject (shared across days).
        """
        cmd_str = (
            f"run_fastsurfer.sh "
            f"--t1 {t1_n4} "
            f"--sid {self.fastsurfer_sid} "
            f"--sd {fs_base_dir} "
            "--threads max --seg_only --3T --device cuda"
        )
        run_in_conda("fastsurfer", cmd_str)

    def _make_brain_masks(self, t1_n4: Path, fastsurfer_dir: Path):
        """
        - Takes aparc+aseg.mgz from FastSurfer
        - Creates brainmask_noCSF_filled.nii
        - Skullstrips T1_N4 via SynthStrip
        - Resamples brainmask_noCSF_filled to T1 brain mask space
        - Multiplies to get T1_combined_mask.nii
        """
        fs_mri_dir = fastsurfer_dir / self.fastsurfer_sid / "mri"
        aparc_aseg = fs_mri_dir / "aparc.DKTatlas+aseg.deep.mgz"

        brainmask_noCSF = self.anat_dir / "brainmask_noCSF.nii"
        filled = self.anat_dir / "brainmask_noCSF_filled.nii"
        t1_brain = self.anat_dir / "T1_brain.nii"
        t1_mask_skull = self.anat_dir / "T1_mask_skull.nii"
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        # 1) binarize aparc+aseg into cortical/GM mask without CSF
        if not brainmask_noCSF.exists():
            run([
                "mri_binarize", "--i", str(aparc_aseg),
                "--match",
                # 2 3 7 8 10 11 12 13 16 17 18 26 28
                "2", "3", "7", "8", "10", "11", "12", "13",
                "16", "17", "18", "26", "28",
                # 41 42 46 47 49 50 51 52 53 54 58 60
                "41", "42", "46", "47", "49", "50", "51", "52",
                "53", "54", "58", "60",
                # 77
                "77",
                # 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015
                "1002", "1003", "1005", "1006", "1007", "1008", "1009",
                "1010", "1011", "1012", "1013", "1014", "1015",
                # 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028
                "1016", "1017", "1018", "1019", "1020", "1021", "1022",
                "1023", "1024", "1025", "1026", "1027", "1028",
                # 1029 1030 1031 1034 1035
                "1029", "1030", "1031", "1034", "1035",
                # 2002 2003 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015
                "2002", "2003", "2005", "2006", "2007", "2008", "2009",
                "2010", "2011", "2012", "2013", "2014", "2015",
                # 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028
                "2016", "2017", "2018", "2019", "2020", "2021", "2022",
                "2023", "2024", "2025", "2026", "2027", "2028",
                # 2029 2030 2031 2034 2035
                "2029", "2030", "2031", "2034", "2035",
                "--o", str(brainmask_noCSF)
            ])
            run(["fslmaths", str(brainmask_noCSF), "-fillh", str(filled)])
        # 2) SynthStrip on T1_N4
        if not t1_brain.exists() or not t1_mask_skull.exists():
            run([
                "mri_synthstrip",
                "-i", str(t1_n4),
                "-o", str(t1_brain),
                "-m", str(t1_mask_skull)
            ])

        # 3) Resample brainmask_noCSF_filled to T1 skullstrip mask space,
        #    then intersect to get combined mask.
        if not t1_combined.exists():
            brainmask_resampled = self.anat_dir / "brainmask_resampled_to_T1.nii"
            run([
                "flirt",
                "-in", str(filled),
                "-ref", str(t1_mask_skull),
                "-applyxfm", "-usesqform",
                "-out", str(brainmask_resampled)
            ])
            run([
                "fslmaths", str(brainmask_resampled),
                "-mul", str(t1_mask_skull),
                str(t1_combined)
            ])

    def _run_synthmorph(self, t1_n4: Path):
        warp = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        warped = self.anat_dir / "T1_warped_to_MNI_synth.nii"
        if warp.exists() and warped.exists():
            return
        run([
            "mri_synthmorph", "-g",
            "-t", str(warp),
            "-o", str(warped),
            str(t1_n4),
            str(self.cfg.mni_template)
        ])

    # ---------- Fieldmap / AP-PA ----------

    def _prepare_fieldmap(self, epi_ref=None):
        """Average raw AP/PA in place, without ANY motion correction."""
        if self.fieldmap_method != "pyhysco":
            raise ValueError("unwarp -> MC currently supports PyHySCO only; ANTs AP-to-PA warps are not interchangeable.")
        ap, pa = self.cfg.ap_file, self.cfg.pa_file
        spec = calibration_spec(ap, pa)
        if spec["ap_pe"] is None and spec["pa_pe"] is None:
            log.warning("AP/PA metadata missing: using legacy voxel phase-encoding axis 2.")
        folder, field, manifest = native_paths(self.fmap_dir)
        folder.mkdir(parents=True, exist_ok=True)
        if manifest.exists() and all(p.exists() for p in (field, folder / "AP_mean.nii", folder / "PA_mean.nii")):
            if json.loads(manifest.read_text()) == spec:
                return
        if manifest.exists():
            manifest.unlink()  # An interrupted build must never appear complete.
        require_same_grid(nib.load(str(ap)), nib.load(str(pa)))
        ap_mean, pa_mean = folder / "AP_mean.nii", folder / "PA_mean.nii"
        save_native_mean(ap, ap_mean)
        save_native_mean(pa, pa_mean)
        self._run_ap_pa_pyhysco(ap_mean, pa_mean, output_stem="pyhysco_native")
        if not field.exists():
            raise FileNotFoundError(f"PyHySCO did not produce {field}")
        manifest.write_text(json.dumps(spec, indent=2))

    def _is_current_output(self, out_path: Path, inputs: list[Path]) -> bool:
        if not out_path.exists():
            return False
        try:
            out_mtime = out_path.stat().st_mtime
            return all(out_mtime >= p.stat().st_mtime for p in inputs if p.exists())
        except OSError:
            return False

    def _is_current_grid_output(self, out_path: Path, ref_path: Path, inputs: list[Path]) -> bool:
        return (
            self._is_current_output(out_path, inputs)
            and self._same_grid(out_path, ref_path)
        )

    def _same_grid(self, a_path: Path, b_path: Path) -> bool:
        try:
            a_img = nib.load(str(a_path))
            b_img = nib.load(str(b_path))
        except Exception:
            return False
        return (
            a_img.shape[:3] == b_img.shape[:3]
            and np.allclose(a_img.affine, b_img.affine, atol=1e-3)
        )

    def _run_ap_pa_pyhysco(self, ap_mean: Path, pa_mean: Path, output_stem: str = "pyhysco"):
        pyhysco_field = ap_mean.parent / f"{output_stem}-EstFieldMap.nii"
        if self._is_current_output(pyhysco_field, [ap_mean, pa_mean]):
            print("✓ PyHySCO fieldmap already exists — skipping")
            return

        pyhysco_prefix = ap_mean.parent / output_stem
        generated_field = ap_mean.parent / f"{output_stem}-EstFieldMap.nii.gz"
        for stale in (pyhysco_field, generated_field):
            if stale.exists():
                stale.unlink()
        pyhysco_src = Path(__file__).resolve().parent / "PyHySCO-main" / "src"
        # The interpreter running this pipeline (it has torch); a login shell's `python` need not be it.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(pyhysco_src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [
            sys.executable, str(pyhysco_src / "scripts" / "pyhysco.py"),
            str(ap_mean), str(pa_mean), str(self.pyhysco_phase_encoding_direction),
            "--output_dir", str(pyhysco_prefix), "--max_iter", "50", "--correction", "jac",
        ]
        run(cmd, env=env)

        if not generated_field.exists() and not pyhysco_field.exists():
            raise FileNotFoundError(
                f"Expected PyHySCO fieldmap not found: {generated_field}"
            )
        if generated_field.exists():
            gunzip_python(generated_field)

    def _run_ap_pa_ants(self, ap_mean: Path, pa_mean: Path, output_prefix_name: str = "AP2PA_"):
        out_prefix = self.fmap_dir / output_prefix_name
        warped = self.fmap_dir / f"{output_prefix_name}Warped.nii"
        inverse_warped = self.fmap_dir / f"{output_prefix_name}InverseWarped.nii"
        cmd = [
            "bash", "-c",
            # wrap in bash so we can export vars easily
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsRegistration \
              --dimensionality 3 \
              --float 1 \
              --verbose 1 \
              --output [{out_prefix}, {warped}, {inverse_warped}] \
              --interpolation Linear \
              --winsorize-image-intensities [0.005,0.995] \
              --use-histogram-matching 1 \
              --initial-moving-transform [{pa_mean}, {ap_mean}, 1] \
              \
              --transform Affine[0.1] \
              --metric MI[{pa_mean}, {ap_mean}, 1, 64, Regular, 0.2] \
              --convergence [3000x2000x1000x500x250,1e-9,25] \
              --shrink-factors 16x12x8x4x2 \
              --smoothing-sigmas 5x4x3x2x1vox \
              \
              --transform SyN[0.1,3,0] \
              --metric MI[{pa_mean}, {ap_mean}, 1, 64, Regular, 0.2] \
              --convergence [200x150x100x50x20,1e-9,20] \
              --shrink-factors 6x5x4x2x1 \
              --smoothing-sigmas 3x2.5x2x1x0vox \
              \
              --transform SyN[0.05,1,0] \
              --metric CC[{pa_mean}, {ap_mean}, 1, 4] \
              --convergence [100x100x50x10,1e-9,10] \
              --shrink-factors 4x3x2x1 \
              --smoothing-sigmas 2x1.5x1x0vox
            """
        ]
        run(cmd)
        # unzip ANTs warps for faster repeated access during RT unwarping
        forward_warp_path = self.fmap_dir / f"{output_prefix_name}1Warp.nii.gz"
        inverse_warp_path = self.fmap_dir / f"{output_prefix_name}1InverseWarp.nii.gz"
        gunzip_python(forward_warp_path)
        gunzip_python(inverse_warp_path)


    # ---------- Per-run ----------

    def _prepare_run(self, run_cfg):
        run_dir = run_cfg.epi_file.parent
        ensure_dir(run_dir)

        # Use a fresh preparation directory when migrating old MC-first assets.
        epi_4d = run_cfg.epi_file
        marker = run_dir / "preparation_order.json"
        identity = {"order": PIPELINE_ORDER, "epi": source_identity(epi_4d),
                    "calibration": calibration_spec(self.cfg.ap_file, self.cfg.pa_file),
                    "epi_phase_encoding": self.epi_phase_encoding}
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise ValueError("Preparation inputs changed. Use a fresh preparation directory to avoid stale transforms.")
        elif any((run_dir / name).exists() for name in
                 ("epi_mc.nii", "epi_unwarped_mean.nii", "epi_unwarped_mean.nii.gz", "epi2t1_Composite.h5")):
            raise ValueError("Legacy preparation detected. Reuse its corrected reference from realtime, or prepare in a fresh day directory.")
        marker.write_text(json.dumps(identity, indent=2))
        self._prepare_fieldmap()

        # Unwarp raw BOLD first, then build an MC mean in corrected space.
        epi_unwarped = run_dir / "epi_unwarped.nii"
        self._unwarp_epi(epi_4d, epi_unwarped)
        epi_mean = run_dir / "epi_unwarped_mean.nii"
        epi_mask_mean = run_dir / "epi_mask_mean.nii"
        self._motion_correct_and_mean(
            epi_4d=epi_unwarped,
            epi_mean=epi_mean,
            n_vols_for_mean=20,
        )
        # Segment the aligned mean, never average unaligned per-volume masks.
        self._skullstrip_epi(epi_mean, run_dir / "epi_brain_mean.nii", epi_mask_mean)
        self._maybe_set_rt_reference(epi_mean, epi_mask_mean)

        # 6) EPI->T1 registration using EPI mean + masks
        self._register_epi_to_t1(run_dir, self.rt_unwarped_analysis_ref_epi,
                                 self.rt_unwarped_analysis_ref_mask)

        # 6.5 create masks for regression
        self._make_rtp_nuisance_masks()

        # 7) EPI mean -> MNI
        self._warp_epi_mean_to_mni(run_dir, epi_mean)

        # ----- Build ROI on *decoder/template* grid -----
        epi_mni_path = run_dir / "epi_in_MNI.nii"
        epi_mni = nib.load(str(epi_mni_path))

        # This is your template decoder (79x95x79 etc.)

        if self.decoder_template.exists():
            dec_img = nib.load(str(self.decoder_template))

            # resample epi to decoder/template grid
            epi_on_dec = image.resample_to_img(
                epi_mni,
                dec_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True,
            )

            # ROI will now be in decoder/template space
            roi_txt = run_dir / "ROI_DECODER.txt"
            self.export_decoder_roi_txt(
                stat_img=dec_img,      # decoder on its native/template grid
                reference_img=epi_on_dec,  # mean EPI resampled onto that grid
                out_txt=roi_txt,
                bias_value=0.0,
            )

    def _extract_first_epi_volume(self, epi_4d: Path, out: Path):
        if out.exists():
            return
        img_4d = nib.load(str(epi_4d))
        data_4d = np.asanyarray(img_4d.dataobj)
        first_vol = data_4d[..., 0] if data_4d.ndim == 4 else data_4d
        nib.save(nib.Nifti1Image(first_vol.astype(np.float32), img_4d.affine, img_4d.header), str(out))

    def _preferred_pyhysco_fieldmap(self) -> Path:
        load_calibration(self.fmap_dir)
        return native_paths(self.fmap_dir)[1]

    def _unwarp_epi(self, epi_4d: Path, out: Path):
        spec = validate_bold(nib.load(str(epi_4d)), self.fmap_dir, self.epi_phase_encoding, epi_4d)
        if out.exists():
            return
        apply_pyhysco_fieldmap(
            epi_path=epi_4d, fieldmap_path=self._preferred_pyhysco_fieldmap(),
            out_path=out, phase_encoding_direction=spec["phase_encoding_axis"],
            polarity=1 if self.epi_phase_encoding == "AP" else -1,
        )

    @property
    def pyhysco_phase_encoding_direction(self) -> int:
        return calibration_spec(self.cfg.ap_file, self.cfg.pa_file)["phase_encoding_axis"]

    def _skullstrip_epi(self, epi_unwarped: Path, brain: Path, mask: Path):
        if brain.exists() and mask.exists():
            return
        run([
            "mri_synthstrip",
            "-i", str(epi_unwarped),
            "-o", str(brain),
            "-m", str(mask)
        ])

    def _motion_correct_and_mean(
            self,
            epi_4d: Path,
            epi_mean: Path,
            n_vols_for_mean: int | None = 20,
    ):
        """
        Motion correction using RTPSpy's RtpVolreg + temporal means.

        Inputs:
          epi_4d       : unwarped 4D EPI before motion correction
        Outputs:
          epi_mc.nii      : motion-corrected EPI (subset of volumes)
          motion.1D           : AFNI-style motion params
          epi_mean            : mean over time of epi_mc
        """

        # Output paths (KEEP EXACT OLD NAMING)
        mc_epi = epi_4d.with_name("epi_mc.nii")
        motion_1d = epi_4d.with_name("motion.1D")

        # If already computed, skip
        if epi_mean.exists() and mc_epi.exists() and motion_1d.exists():
            return

        # ------------------------------------------------------------------
        #                LOAD INPUT EPI (4D)
        # ------------------------------------------------------------------
        img_4d = nib.load(str(epi_4d))
        data_4d = np.asanyarray(img_4d.dataobj)
        affine = img_4d.affine
        header = img_4d.header
        n_total_vols = data_4d.shape[-1]

        # Subset number of volumes (same behavior as AFNI [0..N])
        if n_vols_for_mean is not None:
            n_use = min(n_vols_for_mean, n_total_vols)
        else:
            n_use = n_total_vols

        # Prepare output arrays
        mc_data = np.zeros((*data_4d.shape[:3], n_use), dtype=np.float32)
        motion = np.zeros((n_use, 6), dtype=np.float32)

        # ------------------------------------------------------------------
        #                INITIALIZE RTP VOLREG
        # ------------------------------------------------------------------
        vr = RtpVolreg(regmode='heptic')  # AFNI default interpolation
        vr.ignore_init = 0
        vr.save_proc = False

        # Later reference runs use the same corrected session reference.
        ref = self.rt_unwarped_analysis_ref_epi
        if ref.exists() and ref.resolve() != epi_mean.resolve():
            require_same_grid(img_4d, nib.load(str(ref)))
            vr.set_ref_vol(str(ref))
        else:
            vr.set_ref_vol(f"{epi_4d}[0]")

        # ------------------------------------------------------------------
        #                RUN MOTION CORRECTION VOLUME-BY-VOLUME
        # ------------------------------------------------------------------
        for t in range(n_use):
            vol = data_4d[..., t]
            fmri_img = nib.Nifti1Image(vol, affine)
            fmri_img.set_filename(f"vol_{t:04d}.nii")  # required by RTPSpy internals

            vr.do_proc(fmri_img, vol_idx=t)

            # Extract corrected volume
            mc_vol = np.asanyarray(fmri_img.dataobj)
            mc_data[..., t] = mc_vol.astype(np.float32)

            # Extract motion params (already re-ordered inside RtpVolreg)
            motion[t] = vr._motion[t]

        # ------------------------------------------------------------------
        #                SAVE RESULTS (EXACT SAME NAMES AS OLD CODE)
        # ------------------------------------------------------------------
        nib.save(nib.Nifti1Image(mc_data, affine, header), str(mc_epi))
        np.savetxt(str(motion_1d), motion)

        # ------------------------------------------------------------------
        #                COMPUTE MEANS (EPI + MASK)
        # ------------------------------------------------------------------
        if not epi_mean.exists():
            mean_vol = mc_data.mean(axis=-1)
            nib.save(nib.Nifti1Image(mean_vol, affine, header), str(epi_mean))

    def _maybe_set_rt_reference(self, epi_mean: Path, epi_mask_mean: Path):
        """Publish corrected references; legacy filenames now contain corrected data."""
        for source, target in (
            (epi_mean, self.rt_unwarped_analysis_ref_epi),
            (epi_mask_mean, self.rt_unwarped_analysis_ref_mask),
            (epi_mean, self.trans_dir / "rt_ref_epi.nii"),
            (epi_mask_mean, self.trans_dir / "rt_ref_epi_mask.nii"),
        ):
            if source.resolve() != target.resolve() and not target.exists():
                nib.save(nib.load(str(source)), str(target))

    def _register_epi_to_t1(self, run_dir: Path, epi_mean: Path, epi_mask_mean: Path):
        t1_n4 = self.anat_dir / "T1_N4.nii"
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        out_prefix = self.trans_dir / "epi2t1_"
        warped = self.trans_dir / "epi2t1_Warped.nii"
        inv_warped = self.trans_dir / "epi2t1_InverseWarped.nii"
        composite = self.trans_dir / "epi2t1_Composite.h5"

        # If transforms already exist → reuse, do nothing
        if warped.exists() and inv_warped.exists() and composite.exists():
            print("✓ epi→T1 transform already exists — skipping antsRegistration")
            return

        # Otherwise, run registration ONCE using this run’s epi_mean/mask
        ensure_dir(self.trans_dir)
        print("→ Running epi→T1 antsRegistration (global)")

        cmd = [
            "bash", "-c",
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsRegistration \
              --dimensionality 3 \
              --float 1 \
              --verbose 1 \
              --output [{out_prefix}, {warped}, {inv_warped}] \
              --interpolation Linear \
              --winsorize-image-intensities [0.005,0.995] \
              --use-histogram-matching 0 \
              --write-composite-transform 1 \
              --initial-moving-transform [{epi_mean}, {t1_n4}, 1] \
              --masks [{epi_mask_mean}, {t1_combined}] \
              \
              --transform Affine[0.1] \
              --metric MI[{t1_n4}, {epi_mean}, 1, 32, Regular, 0.2] \
              --convergence [1000x500x250x100,1e-7,15] \
              --shrink-factors 8x4x2x1 \
              --smoothing-sigmas 3x2x1x0vox \
              \
              --transform SyN[0.1,1,0] \
              --metric MI[{t1_n4}, {epi_mean}, 1, 32, Regular, 0.2] \
              --convergence [100x70x50x20,1e-7,15] \
              --shrink-factors 6x4x2x1 \
              --smoothing-sigmas 3x2x1x0vox
            """
        ]
        run(cmd)
        # Copy geometry from T1 (helps keep everything aligned)
        run(["fslcpgeom", str(t1_n4), str(warped)])
        run(["fslcpgeom", str(t1_n4), str(inv_warped)])

    def _make_rtp_nuisance_masks(self):
        """
        Create WM / Vent masks for RTPSpy RtpRegress.

        Outputs (all in EPI/RT grid):
          trans/rt_WM_mask.nii
          trans/rt_Vent_mask.nii

        Notes:
          - GS uses epi_mask_mean directly (no rt_GS_mask file).

        Assumes:
          - FastSurfer outputs exist in anat/fastsurfer/<sid>/mri/
          - epi2t1 transforms exist in func/trans/ (from _register_epi_to_t1)
          - unwarped EPI reference/mask exist (from _unwarp_epi + mean)
        """
        ensure_dir(self.trans_dir)

        # -------------------------
        # 0) References (EPI grid)
        # -------------------------
        if not self.rt_unwarped_analysis_ref_epi.exists():
            raise FileNotFoundError(f"Unwarped RT reference missing: {self.rt_unwarped_analysis_ref_epi}")
        if not self.rt_unwarped_analysis_ref_mask.exists():
            raise FileNotFoundError(f"Unwarped RT mask missing: {self.rt_unwarped_analysis_ref_mask}")

        epi_ref = self.rt_unwarped_analysis_ref_epi
        epi_brainmask = self.rt_unwarped_analysis_ref_mask
        fsl_env = dict(os.environ, FSLOUTPUTTYPE="NIFTI")

        # GS regressor uses the EPI brain mask directly.
        gs_epi = epi_brainmask
        slice_slab = self.trans_dir / "rt_slice_slab.nii"
        if not slice_slab.exists():
            # collapse x/y -> leaves a z “which slices have brain” mask (broadcasts back fine)
            run(["fslmaths", str(gs_epi), "-Tmax", str(slice_slab)], env=fsl_env)


        # -----------------------------------------
        # 1) Build WM/Vent in T1 space from FastSurfer
        # -----------------------------------------
        fs_mri_dir = self.anat_dir / "fastsurfer" / self.fastsurfer_sid / "mri"
        #aseg_mgz = fs_mri_dir / "aseg.auto_noCCseg.mgz"
        # If you prefer your DKT aparc+aseg:
        aseg_mgz = fs_mri_dir / "aparc.DKTatlas+aseg.deep.mgz"

        if not aseg_mgz.exists():
            raise FileNotFoundError(f"FastSurfer aseg not found: {aseg_mgz}")

        wm_t1 = self.trans_dir / "WM_T1.nii"
        vent_t1 = self.trans_dir / "Vent_T1.nii"

        # Common FreeSurfer aseg labels:
        # WM: 2 (LH WM), 41 (RH WM)
        # Ventricles: 4,43 (lat), 5,44 (inf lat), 14 (3rd), 15 (4th)
        WM_LABELS = ["2", "41"]
        VENT_LABELS = ["4", "43", "5", "44", "14", "15"]

        if not wm_t1.exists():
            run([
                "mri_binarize",
                "--i", str(aseg_mgz),
                "--match", *WM_LABELS,
                "--o", str(wm_t1)
            ])

        if not vent_t1.exists():
            run([
                "mri_binarize",
                "--i", str(aseg_mgz),
                "--match", *VENT_LABELS,
                "--o", str(vent_t1)
            ])

        # -----------------------------------------
        # 2) Warp WM/Vent T1->EPI using *composite* (same style as other warps)
        # -----------------------------------------
        inv_comp = self.trans_dir / "epi2t1_InverseComposite.h5"
        if not inv_comp.exists():
            raise FileNotFoundError(f"Missing inverse composite: {inv_comp}")

        wm_epi = self.trans_dir / "rt_WM_mask.nii"
        vent_epi = self.trans_dir / "rt_Vent_mask.nii"

        def _apply_t1_to_epi_nn(in_mask_t1: Path, out_mask_epi: Path):
            if out_mask_epi.exists():
                return

            cmd = [
                "bash", "-c",
                f"""
                export ANTS_USE_GPU=1
                export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
                export OMP_NUM_THREADS=$(nproc)
                antsApplyTransforms -d 3 \
                  -i {in_mask_t1} \
                  -r {epi_ref} \
                  -o {out_mask_epi} \
                  -n NearestNeighbor \
                  -t {inv_comp} \
                  --float 1
                """
            ]
            run(cmd)

            # Clean binary + intersect with EPI brain mask (safety)
            run(["fslmaths", str(out_mask_epi), "-thr", "0.5", "-bin", str(out_mask_epi)], env=fsl_env)
            run(["fslmaths", str(out_mask_epi), "-mul", str(gs_epi), str(out_mask_epi)], env=fsl_env)

        _apply_t1_to_epi_nn(wm_t1, wm_epi)
        _apply_t1_to_epi_nn(vent_t1, vent_epi)

        run(["fslmaths", str(wm_epi), "-mul", str(slice_slab), str(wm_epi)], env=fsl_env)
        run(["fslmaths", str(vent_epi), "-mul", str(slice_slab), str(vent_epi)], env=fsl_env)

        # -----------------------------------------
        # 3) Final sanity check: shapes must match EPI volume shape
        # -----------------------------------------
        ref_shape = nib.load(str(epi_ref)).shape[:3]
        for p in (gs_epi, wm_epi, vent_epi):
            shp = nib.load(str(p)).shape[:3]
            if shp != ref_shape:
                raise RuntimeError(f"Mask shape mismatch for {p.name}: {shp} vs ref {ref_shape}")

        log.info(f"✓ RTP masks ready:\n  GS={gs_epi}\n  WM={wm_epi}\n  Vent={vent_epi}")
        return gs_epi, wm_epi, vent_epi

    def _warp_epi_mean_to_mni(self, run_dir: Path, epi_mean: Path):
        mni = self.cfg.mni_template
        warp_t1_mni = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        composite = self.trans_dir / "epi2t1_Composite.h5"
        out_epi_mni = self.trans_dir / "epi_in_MNI.nii"

        if out_epi_mni.exists():
            print(f"✓ epi_in_MNI already exists for {self.trans_dir.name} — skipping warp")
            return

        if not self.decoder_template.exists():
            raise FileNotFoundError(
                f"Decoder template not found at {self.decoder_template} "
                f"(expected relative to {self.cfg.root})"
            )

        if not composite.exists():
            raise FileNotFoundError(
                f"EPI→T1 composite transform not found at {composite}. "
                f"Make sure _register_epi_to_t1 ran at least once."
            )

        cmd = [
            "bash", "-c",
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsApplyTransforms \
              -d 3 \
              -i {epi_mean} \
              -r {self.decoder_template} \
              -o {out_epi_mni} \
              -t {warp_t1_mni} \
              -t {composite} \
              -n Linear --float 1
            """
        ]
        #-r {mni} \
        run(cmd)

    def qc_plot_epi_in_mni(
            self,
            run_dir: Path,
            decoder_path: Path | None = None,
            out_png: Path | None = None,
            dpi: int = 300,
    ):
        """High-quality QC plot."""

        epi_mni = self.trans_dir / "epi_in_MNI.nii"
        if not epi_mni.exists():
            raise FileNotFoundError(f"epi_in_MNI not found at {epi_mni}")

        if out_png is None:
            out_png = self.trans_dir / "qc_epi_in_MNI.png"

        try:
            import nibabel as nib
            import matplotlib.pyplot as plt
            from nilearn import plotting, image
        except ImportError:
            log.warning("Nilearn not installed; skipping QC plot.")
            return

        epi_img = nib.load(str(epi_mni))
        mni_template = nib.load(str(self.cfg.mni_template))

        # Default background
        bg_img = mni_template
        stat_img = epi_img

        # Overlay decoder if provided
        if decoder_path is not None and Path(decoder_path).exists():
            dec_img = nib.load(str(decoder_path))
            dec_resampled = image.resample_to_img(
                dec_img, epi_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True
            )
            bg_img = epi_img
            stat_img = dec_resampled

        # Auto cut coordinates
        cut_coords = plotting.find_xyz_cut_coords(stat_img)

        # High-quality figure
        fig = plt.figure(figsize=(9, 9), dpi=dpi)
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_img,
            display_mode="ortho",
            cut_coords=cut_coords,
            threshold=None,
            annotate=True,
            axes=fig.add_subplot(111),
            title="EPI in MNI (and decoder)" if decoder_path else "EPI in MNI",
        )

        display.savefig(str(out_png), dpi=dpi)
        plt.close(fig)

    def export_decoder_roi_txt(
            self,
            stat_img,
            reference_img,
            out_txt: Path,
            bias_value: float | None = None,
    ):
        """
        Export ROI in the same format as the MATLAB ROI_NPS.txt script.

        stat_img      = nibabel image (decoder in MNI or EPI space)
                        (equivalent to `decoder` in MATLAB)
        reference_img = nibabel image (usually epi_mean / wmeanVol)
                        (equivalent to `wmeanVol` in MATLAB)
        out_txt       = output txt path (e.g. ROI_DECODER.txt)
        bias_value    = final bias term; if None -> 0.0

        File format (same as MATLAB):
          line 1: nx ny nz
          line 2: N_voxels
          next N lines: x y z decoder(x,y,z) wmeanVol(x,y,z)
          last line: 0 0 0 bias 0
        """

        import numpy as np

        dec = np.asanyarray(stat_img.get_fdata())
        ref = np.asanyarray(reference_img.get_fdata())

        # ---- voxel selection: same logic as MATLAB ----
        # MATLAB: find(decoder ~= 0 & ~isnan(decoder))
        # Here we treat tiny values as non-zero as well,
        # to reproduce lines that *print* as 0.000000 but are actually ~1e-7.
        eps = 1e-12
        mask_nonzero = np.logical_and(np.abs(dec) > eps, ~np.isnan(dec))
        coords = np.array(np.where(mask_nonzero)).T  # 0-based indices (i,j,k)

        nx, ny, nz = dec.shape

        with open(out_txt, "w") as f:
            # Header: dimensions
            f.write(f"{nx} {ny} {nz}\n")

            # Number of voxels
            f.write(f"{coords.shape[0]}\n")

            # Each voxel: x y z decoder(x,y,z) wmeanVol(x,y,z)
            # IMPORTANT: +1 to go from 0-based (Python) to 1-based (MATLAB / ROI_NPS.txt)
            for (i, j, k) in coords:
                x = i + 1
                y = j + 1
                z = k + 1
                f.write(
                    f"{x} {y} {z} "
                    f"{dec[i, j, k]:.6f} {ref[i, j, k]:.6f}\n"
                )

            # Final bias line (MATLAB: if exist(bias) ... else ... end)
            if bias_value is None:
                bias_value = 0.0
            f.write(f"0 0 0 {bias_value:.6f} 0\n")
