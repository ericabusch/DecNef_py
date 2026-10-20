"""Optional PsychoPy helpers for launchers (install the 'task' extra).

Only the generic pieces live here: the window, releasing the mouse, the start
('s') / stop ('q' or Escape) keys, and the acquisition-speed log. Paradigms
(what is drawn, trial structure) belong to the launcher; see
examples/nf_events_launcher.py.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
START_KEYS = ("s",)
STOP_KEYS = ("q", "escape")


def release_mouse(win) -> None:
    try:
        win.mouseVisible = True
    except Exception as exc:
        log.warning("could not show the mouse cursor: %s", exc)
    handles = [getattr(win, "winHandle", None)]
    backend = getattr(win, "backend", None)
    if backend is not None:
        handles.append(getattr(backend, "winHandle", None))
    for handle in handles:
        if handle is not None and hasattr(handle, "set_exclusive_mouse"):
            try:
                handle.set_exclusive_mouse(False)
            except Exception as exc:
                log.warning("could not release exclusive mouse control: %s", exc)


def build_window(visual, color, size=(1000, 700), prefer_second_screen: bool = True):
    """A PsychoPy window: fullscreen on the second screen when there is one, else
    windowed on the first; falls back to a safe windowed setup if the first attempt fails."""
    kwargs = {"size": size, "color": color, "units": "pix", "screen": 0, "fullscr": False}
    if prefer_second_screen:
        try:
            import pyglet

            screens = pyglet.canvas.get_display().get_screens()
            if len(screens) > 1:
                kwargs.update({"size": (screens[1].width, screens[1].height), "screen": 1, "fullscr": True})
        except Exception as exc:
            log.warning("could not detect a second screen; using the default window: %s", exc)
    try:
        win = visual.Window(**kwargs)
    except Exception as exc:
        log.warning("window setup failed (%s); retrying with safe defaults", exc)
        win = visual.Window(size=size, color=color, units="pix", screen=0, fullscr=False)
    release_mouse(win)
    return win


def pressed(keys, which) -> bool:
    return any(k in keys for k in which)


def wait_for_start(win, event, core, draw, poll=None, interval: float = 0.02) -> Optional[float]:
    """Draw `draw()` each frame until 's' (returns the trigger time) or a stop key (returns None).
    `poll()` is called each frame so queued events are consumed meanwhile."""
    while True:
        if poll is not None:
            poll()
        draw()
        win.flip()
        keys = event.getKeys()
        if pressed(keys, START_KEYS):
            return time.time()
        if pressed(keys, STOP_KEYS):
            return None
        core.wait(interval)


class AcquisitionSpeedLog:
    """acquisition_speed_rt.csv: estimated trigger -> DICOM pickup -> score latencies per volume."""

    FIELDS = ["volume_idx", "estimated_trigger_timestamp", "watchdog_timestamp", "analysis_timestamp",
              "trigger_to_watchdog_s", "watchdog_to_analysis_s"]

    def __init__(self, path: Path, tr_seconds: float, first_trigger_timestamp: float):
        self.path = Path(path)
        self.tr = float(tr_seconds)
        self.t0 = float(first_trigger_timestamp)

    def record(self, event) -> None:
        if event.volume_idx is None or event.watchdog_timestamp is None:
            return
        analysis = event.analysis_timestamp if event.analysis_timestamp is not None else event.timestamp
        if analysis is None:
            return
        est = self.t0 + (event.volume_idx - 1) * self.tr
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if not exists:
                w.writeheader()
            w.writerow({"volume_idx": event.volume_idx, "estimated_trigger_timestamp": f"{est:.6f}",
                        "watchdog_timestamp": f"{event.watchdog_timestamp:.6f}", "analysis_timestamp": f"{analysis:.6f}",
                        "trigger_to_watchdog_s": f"{event.watchdog_timestamp - est:.6f}",
                        "watchdog_to_analysis_s": f"{analysis - event.watchdog_timestamp:.6f}"})
