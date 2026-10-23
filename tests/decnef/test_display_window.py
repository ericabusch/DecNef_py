"""decnef.display.window_kwargs: which screen the launcher window goes to, and how (no PsychoPy needed)."""
import pytest

from decnef.display import window_kwargs

ONE = [(0, 0, 1920, 1080)]
TWO = [(0, 0, 1920, 1080), (1920, 0, 1920, 1200)]


def test_one_screen_gives_a_window_on_screen_0():
    kw = window_kwargs(ONE, color=[0, 0, 0], size=(1000, 700), prefer_second_screen=True)
    assert kw["screen"] == 0 and kw["fullscr"] is False and kw["size"] == (1000, 700)


def test_second_screen_fullscreen_on_mac_and_windows():
    for platform in ("darwin", "win32"):
        kw = window_kwargs(TWO, color=[0, 0, 0], prefer_second_screen=True, platform=platform)
        assert kw["screen"] == 1 and kw["fullscr"] is True and kw["size"] == (1920, 1200) and "pos" not in kw


def test_second_screen_is_borderless_at_its_offset_on_linux():
    kw = window_kwargs(TWO, color=[0, 0, 0], prefer_second_screen=True, platform="linux")
    assert kw == {"size": (1920, 1200), "pos": (1920, 0), "color": [0, 0, 0], "units": "pix", "screen": 1,
                  "fullscr": False, "allowGUI": False}
    assert window_kwargs(TWO, color=[0, 0, 0], prefer_second_screen=True, platform="linux", borderless=False)["fullscr"] is True


def test_explicit_screen_wins_and_is_checked():
    kw = window_kwargs(TWO, color=[0, 0, 0], prefer_second_screen=False, screen=1, platform="darwin")
    assert kw["screen"] == 1 and kw["fullscr"] is True
    kw0 = window_kwargs(TWO, color=[0, 0, 0], prefer_second_screen=True, screen=0, platform="darwin")
    assert kw0["screen"] == 0 and kw0["fullscr"] is True and kw0["size"] == (1920, 1080)
    with pytest.raises(ValueError):
        window_kwargs(ONE, color=[0, 0, 0], screen=1)


def test_prefer_second_screen_false_stays_windowed():
    kw = window_kwargs(TWO, color=[0, 0, 0], size=(800, 600), prefer_second_screen=False, platform="linux")
    assert kw["screen"] == 0 and kw["fullscr"] is False and kw["size"] == (800, 600)
