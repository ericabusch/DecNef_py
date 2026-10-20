"""Live QC viewer: a daemon process that shows the latest volume (matplotlib).

The pipeline only publishes (volume_idx, path) over a small queue; rendering
failures never affect the run. Enabled with `qc.volume_viewer`.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
STREAM_KINDS = ("raw", "mc", "reg", "score_input")


@dataclass
class ViewerConfig:
    enabled: bool = False
    kind: str = "mc"                 # which per-volume file to show (see STREAM_KINDS)
    every_n: int = 1
    max_queue: int = 2
    window_title: str = "RT volume QC"
    cmap: str = "gray"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    poll_interval_s: float = 0.05

    @classmethod
    def from_settings(cls, qc) -> "ViewerConfig":
        return cls(enabled=qc.volume_viewer, kind=qc.stream_kind, every_n=qc.every_n, max_queue=qc.max_queue,
                   window_title=qc.window_title)


class ViewerHandle:
    def __init__(self, config: ViewerConfig):
        self.config = config
        self._queue = None
        self._stop_event = None
        self._process = None

    def start(self) -> None:
        if not self.config.enabled or self._process is not None:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=self.config.max_queue)
        self._stop_event = ctx.Event()
        self._process = ctx.Process(target=_viewer_main, args=(self.config, self._queue, self._stop_event), daemon=True)
        self._process.start()
        log.info("[VOLQC] started %s viewer (pid=%s)", self.config.kind, self._process.pid)

    def publish(self, volume_idx: int, path: Optional[Path]) -> None:
        if not self.config.enabled or self._queue is None or path is None or volume_idx % self.config.every_n != 0:
            return
        try:
            while True:   # drop stale frames so the viewer never lags behind real time
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait((int(volume_idx), str(path)))
        except queue.Full:
            pass
        except Exception as exc:
            log.warning("[VOLQC] could not publish volume %05d: %s", volume_idx, exc)

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = self._queue = self._stop_event = None


def _viewer_main(config: ViewerConfig, q, stop_event) -> None:
    vlog = logging.getLogger("decnef.viewer")
    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
    except Exception as exc:
        vlog.warning("[VOLQC] viewer disabled; could not import GUI dependencies: %s", exc)
        return
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.2), num=config.window_title)
    try:
        fig.canvas.manager.set_window_title(config.window_title)
    except Exception:
        pass
    fig.tight_layout()
    last_path = None
    while not stop_event.is_set():
        try:
            volume_idx, path_text = q.get(timeout=config.poll_interval_s)
        except queue.Empty:
            plt.pause(config.poll_interval_s)
            continue
        path = Path(path_text)
        if path == last_path or not path.exists():
            continue
        last_path = path
        try:
            data = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
            data = np.squeeze(data) if data.ndim > 3 else data
            if data.ndim != 3:
                raise ValueError(f"expected 3D volume, got shape {data.shape}")
            _draw_volume(fig, axes, data, volume_idx, path.name, config)
        except Exception as exc:
            vlog.warning("[VOLQC] could not render %s: %s", path, exc)
    plt.close(fig)


def _draw_volume(fig, axes, data, volume_idx, label, config) -> None:
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [config.percentile_low, config.percentile_high])
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    mids = [s // 2 for s in data.shape]
    slices = [np.rot90(data[mids[0], :, :]), np.rot90(data[:, mids[1], :]), np.rot90(data[:, :, mids[2]])]
    for ax, slc, title in zip(axes, slices, ["sag", "cor", "axi"]):
        ax.clear()
        ax.imshow(slc, cmap=config.cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{config.kind} vol {volume_idx:05d}: {label}", fontsize=9)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(0.001)
