"""Optional PsychoPy helpers for launchers (install the 'task' extra).

Only the generic pieces live here: the window, releasing the mouse, the start
('s') / stop ('q' or Escape) keys, and the acquisition-speed log. Paradigms
(what is drawn, trial structure) belong to the launcher; see
examples/nf_events_launcher.py.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
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


def parse_xrandr_monitors(text: str) -> list:
    """[(x, y, width, height)] from `xrandr --listmonitors` output (one monitor per line, e.g.
    ` 0: +*HDMI-0 1920/598x1080/336+0+0  HDMI-0`)."""
    found = []
    for m in re.finditer(r"^\s*\d+:\s+\S+\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)", text, re.MULTILINE):
        w, h, x, y = (int(v) for v in m.groups())
        found.append((x, y, w, h))
    return found


def xrandr_screens() -> list:
    """Monitors from `xrandr --listmonitors` (X11 only); [] when xrandr is missing or fails."""
    try:
        out = subprocess.run(["xrandr", "--listmonitors"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("xrandr not usable: %s", exc)
        return []
    return parse_xrandr_monitors(out.stdout) if out.returncode == 0 else []


def list_screens() -> list:
    """[(x, y, width, height)] of the attached screens, index = PsychoPy `screen`.

    pyglet is asked first. On Linux, pyglet (via Xinerama) often reports one screen spanning
    the whole desktop even when two monitors are attached; then the monitors from
    `xrandr --listmonitors` are used instead (xrandr lists the primary monitor first)."""
    import pyglet

    screens = [(int(sc.x), int(sc.y), int(sc.width), int(sc.height)) for sc in pyglet.canvas.get_display().get_screens()]
    if sys.platform.startswith("linux") and len(screens) < 2:
        monitors = xrandr_screens()
        if len(monitors) > len(screens):
            log.info("pyglet reports %s; using the %d monitors from xrandr instead: %s", screens, len(monitors), monitors)
            return monitors
    return screens


def window_kwargs(screens: list, color, size=(1000, 700), prefer_second_screen: bool = True,
                  screen: Optional[int] = None, borderless: Optional[bool] = None, platform: str = sys.platform) -> dict:
    """PsychoPy `Window` arguments for the chosen screen.

    `screen` picks a screen index explicitly; otherwise the second screen is used when
    `prefer_second_screen` and one exists, else the first. A window on a non-primary screen
    (or an explicit `screen`) fills that screen. On Linux a fullscreen window asked for on
    screen 1 is placed on the primary screen by pyglet/X11, so there the window is created
    borderless at the screen's own offset and size instead (`borderless=None` = Linux only);
    the returned `screen` is then 0 (pyglet's origin screen) and `pos` the monitor's offset."""
    base = {"size": tuple(size), "color": color, "units": "pix", "screen": 0, "fullscr": False}
    if screen is None:
        idx = 1 if (prefer_second_screen and len(screens) > 1) else 0
        if prefer_second_screen and len(screens) < 2:
            log.warning("one screen found (%s); fullscreen on a second screen is not possible, using a %sx%s window",
                        screens, size[0], size[1])
    else:
        idx = int(screen)
        if not 0 <= idx < len(screens):
            raise ValueError(f"screen {idx} requested but only {len(screens)} screen(s) found: {screens}")
    if idx == 0 and screen is None:
        return base
    x, y, w, h = screens[idx]
    if borderless is None:
        borderless = platform.startswith("linux")
    if borderless:
        # PsychoPy adds the chosen pyglet screen's own offset to `pos`, and on Linux pyglet may
        # know only one screen: place the window on screen 0 at the monitor's absolute position.
        x0, y0 = screens[0][0], screens[0][1]
        return {**base, "size": (w, h), "pos": (x - x0, y - y0), "screen": 0, "fullscr": False, "allowGUI": False}
    return {**base, "size": (w, h), "screen": idx, "fullscr": True}


def build_window(visual, color, size=(1000, 700), prefer_second_screen: bool = True,
                 screen: Optional[int] = None, borderless: Optional[bool] = None):
    """A PsychoPy window: fullscreen on the second screen when there is one (or on `screen`),
    else windowed on the first; falls back to a safe windowed setup if the first attempt fails.
    See `window_kwargs` for how the screen is chosen; the screens found and the choice are logged."""
    kwargs = {"size": size, "color": color, "units": "pix", "screen": 0, "fullscr": False}
    if prefer_second_screen or screen is not None:
        try:
            screens = list_screens()
            kwargs = window_kwargs(screens, color, size, prefer_second_screen, screen, borderless)
            log.info("screens %s -> window kwargs %s", screens,
                     {k: v for k, v in kwargs.items() if k in ("screen", "size", "pos", "fullscr")})
            if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and (kwargs["screen"] > 0 or "pos" in kwargs):
                log.warning("Wayland session: the compositor may ignore the window position and put the window "
                            "on the primary screen; log in with 'Ubuntu on Xorg' for the projector")
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
