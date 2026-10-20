import json
import socket
import time

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from decnef.physio import FileBuffer, Receiver, ReceiverConfig
from decnef.smoothing import MaskedGaussianSmoother, smooth_file


def test_receiver_round_trip_over_local_socket(tmp_path):
    rx = Receiver(ReceiverConfig(host="127.0.0.1", port=0, timeout=0.2, handshake_tr=1.4, subject="s", day="d", run="r",
                                 output_path=tmp_path / "rx.csv"))
    rx.start()
    try:
        with socket.create_connection(("127.0.0.1", rx.port), timeout=2) as c:
            c.settimeout(2)
            handshake = json.loads(c.makefile().readline())
            assert handshake["kind"] == "handshake" and handshake["tr"] == 1.4
            for v in (1, 2):
                c.sendall((json.dumps({"kind": "retrots", "volume_idx": v, "n_regressors": 3,
                                       "regressors": [v, v + 0.5, v + 1], "timestamp": 100.0 + v}) + "\n").encode())
            assert rx.wait_for_volume(2, timeout=2)
            retro = rx.get_retrots(1.4, 3, 0.0)                # vol 3 never arrives -> zero row, marked missing
            assert retro.shape == (3, 3) and retro[1, 1] == 2.5 and not retro[2].any()
            assert rx.was_missing(3) and not rx.was_missing(2)
            assert rx.get_trigger_timestamp(2) == 102.0
            rx.send_run_start()
            msg = json.loads(c.makefile().readline())
            assert msg["kind"] == "run_start" and msg["run"] == "r"
    finally:
        rx.stop()
    rows = (tmp_path / "rx.csv").read_text().splitlines()
    assert rows[0] == "volume_idx,timestamp,reg_01,reg_02,reg_03" and rows[2].startswith("2,102.000000,2.000000")

    fb = FileBuffer(tmp_path / "rx.csv", timeout=0.1, poll_interval=0.01)
    assert fb.get_retrots(1.4, 2, 0.0)[0, 0] == 1.0 and fb.was_missing(5) is False
    fb.get_retrots(1.4, 5, 0.0)
    assert fb.was_missing(5)


def test_smoother_matches_normalized_convolution_and_mask_only(tmp_path):
    rng = np.random.default_rng(0)
    mask = np.zeros((8, 8, 8), dtype=bool); mask[2:6, 1:7, 3:7] = True
    data = rng.normal(size=mask.shape)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    sm = MaskedGaussianSmoother(mask, affine, fwhm_mm=4.0)
    out = sm.apply(data).copy()
    sigma = 4.0 / (np.sqrt(8 * np.log(2)) * np.array([2.0, 2.0, 3.0]))
    num = gaussian_filter(np.where(mask, data, 0.0), sigma, mode="constant", cval=0.0, truncate=4.0)
    den = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0.0, truncate=4.0)
    expected = np.where(mask, num / np.where(den < 1e-6, 1.0, den), 0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
    masked = MaskedGaussianSmoother(mask, affine, 0.0).apply(data)
    assert masked[~mask].max() == 0 and np.array_equal(masked[mask], data[mask])
    src = tmp_path / "v.nii"
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), src)
    dst = smooth_file(sm, src, tmp_path / "s.nii")
    assert dst.exists() and not (tmp_path / ".s.nii").exists()
    with pytest.raises(ValueError, match="grids differ"):
        smooth_file(sm, src, tmp_path / "x.nii") if False else smooth_file(
            MaskedGaussianSmoother(mask, np.eye(4), 1.0), src, tmp_path / "x.nii")
