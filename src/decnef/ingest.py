"""Volume ingest: DICOM file names, DICOM -> NIfTI conversion, and the incoming-folder watcher.

DICOM names are `SSS_RRRRRR_VVVVVV.dcm` (series, run, scan); the scan number is
the volume index. Watchdog `created`/`closed` events (and a startup scan of
the folder) feed a priority queue ordered by scan number.
"""
from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler

from .tools import run

log = logging.getLogger(__name__)
DCM2NIIX_RETRIES = 3
DCM2NIIX_RETRY_SLEEP_S = 0.2


def parse_dicom_name(name: str) -> Optional[tuple[int, int, int]]:
    """'001_000004_000003.dcm' -> (series=1, run=4, scan=3); None if the name does not match."""
    parts = Path(name).stem.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def list_run_dicoms(folder: Path, run_id: int) -> list[Path]:
    """Existing `*.dcm` files of one run, in scan order (processed before going online)."""
    out = []
    for path in sorted(Path(folder).glob("*.dcm")):
        parsed = parse_dicom_name(path.name)
        if parsed is not None and parsed[1] == run_id:
            out.append(path)
    return out


def convert_dicom(dicom_path: Path, raw_dir: Path, volume_idx: int) -> Path:
    """DICOM -> `raw_dir/vol_<idx>.nii` with dcm2niix (unchanged flags); returns the NIfTI path.
    Already-converted volumes are reused. Raises after DCM2NIIX_RETRIES attempts."""
    raw_dir = Path(raw_dir)
    raw_nii = raw_dir / f"vol_{volume_idx:05d}.nii"
    if raw_nii.exists():
        return raw_nii
    for attempt in range(1, DCM2NIIX_RETRIES + 1):
        run(["dcm2niix", "-z", "n", "-s", "y", "-a", "y", "-b", "y", "-v", "0",
             "-f", f"vol_{volume_idx:05d}", "-o", str(raw_dir), str(dicom_path)])
        produced = sorted(raw_dir.glob(f"vol_{volume_idx:05d}*.nii*"))
        if produced:
            return produced[0]
        log.warning("[DICOM] vol %05d: dcm2niix produced no output (attempt %d/%d)", volume_idx, attempt, DCM2NIIX_RETRIES)
        time.sleep(DCM2NIIX_RETRY_SLEEP_S)
    raise RuntimeError(f"dcm2niix produced no output for volume {volume_idx} ({dicom_path})")


class IncomingWatcher(FileSystemEventHandler):
    """Collects this run's DICOMs from filesystem events into a scan-ordered queue.
    `on_first_online_volume` is called once when the first volume arrives after
    `set_online()` (used to tell the physio receiver the run started)."""

    def __init__(self, run_id: int, on_first_online_volume=None):
        super().__init__()
        self.run_id = int(run_id)
        self._pending: queue.PriorityQueue = queue.PriorityQueue()
        self._pending_scans: set[int] = set()
        self._seen_scans: set[int] = set()
        self._online = False
        self._run_started = False
        self._on_first_online_volume = on_first_online_volume

    # watchdog callbacks
    def on_created(self, event):
        if not event.is_directory:
            self.enqueue(Path(event.src_path))

    def on_closed(self, event):
        if not event.is_directory:
            self.enqueue(Path(event.src_path))

    def set_online(self) -> None:
        self._online = True

    def enqueue(self, path: Path) -> None:
        parsed = parse_dicom_name(path.name)
        if parsed is None or parsed[1] != self.run_id:
            return
        scan = parsed[2]
        if self._online and not self._run_started:
            self._run_started = True
            if self._on_first_online_volume is not None:
                self._on_first_online_volume()
        if scan in self._seen_scans or scan in self._pending_scans:
            return
        self._pending.put((scan, path))
        self._pending_scans.add(scan)

    def next_pending(self, timeout: float = 0.0) -> Optional[tuple[int, Path]]:
        try:
            scan, path = self._pending.get(timeout=timeout) if timeout > 0 else self._pending.get_nowait()
        except queue.Empty:
            return None
        self._pending_scans.discard(scan)
        self._seen_scans.add(scan)
        return scan, path

    @property
    def pending_scans(self) -> set[int]:
        return set(self._pending_scans)
