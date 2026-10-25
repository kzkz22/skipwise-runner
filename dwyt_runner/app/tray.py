"""The tray icon (plan 15b, decision 18.7): the runner's state at a glance, the window on a click, and the
way to quit. Closing the window only hides it while the icon is there.

The icon is the dwyt logo with a coloured dot. Colours: green = waiting for work, blue = working,
orange = the GPU is being handed over between the
language model and Whisper (do not use the local models now), red = stopped with an error, grey = not
paired. Where no tray exists (some Linux desktops) the app runs on without it; a new start shows the window.
"""
from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from typing import Any
from ..i18n import gettext as _

log = logging.getLogger("dwyt-runner")

COLOURS = {"green": (34, 160, 90), "blue": (37, 99, 235), "orange": (234, 128, 20), "red": (210, 45, 45),
           "grey": (130, 130, 140)}
POLL_S = 2.0


def state_of(s: dict[str, Any]) -> tuple[str, str]:
    """(colour, tooltip) from Runtime.status()."""
    if not s.get("paired"):
        return "grey", _("Skipwise futtató – nincs párosítva")
    if s.get("error"):
        return "red", (_("Skipwise futtató – leállt: ") + s["error"])[:120]
    if not s.get("running"):
        return "grey", _("Skipwise futtató – nem fut")
    if (s.get("swap") or {}).get("phase") in ("waiting", "swapping", "whisper"):
        return "orange", _("Skipwise futtató – modellcsere: most ne használd a helyi modelleket")
    cur = s.get("current") or {}
    if cur:
        label = cur.get("kind_label") or "munka"
        stage = cur.get("stage_label") or ""
        return "blue", _("Skipwise futtató – {label}").format(label=label) + (f": {stage}" if stage else "")
    return "green", _("Skipwise futtató – vár a következő munkára")


def image(colour: str, size: int = 64):
    """The dwyt logo with a status dot in the lower right corner (white ring, so it reads on the blue tile)."""
    from PIL import ImageDraw

    from .logo import image as logo
    img = logo(size)
    d = ImageDraw.Draw(img)
    r = size * 0.25
    cx = cy = size - r - size * 0.01
    ring = max(1.0, size * 0.06)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
    d.ellipse((cx - r + ring, cy - r + ring, cx + r - ring, cy + r - ring), fill=COLOURS[colour] + (255,))
    return img


def on_main(fn: Callable[[], None]) -> None:
    """macOS allows AppKit (the menu-bar icon) only on the main thread: changes from the polling thread are handed
    to the run loop there; elsewhere they run at once."""
    if sys.platform == "darwin":
        try:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(fn)
            return
        except ImportError:
            pass
    fn()


class Tray:
    """pystray icon kept in step with the runtime by a small polling thread."""

    def __init__(self, status: Callable[[], dict[str, Any]], show: Callable[[], None],
                 quit: Callable[[], None]) -> None:
        self.status, self.show, self.quit = status, show, quit
        self.icon = None
        self.colour = ""
        self.stopped = threading.Event()

    def start(self) -> bool:
        """True when a tray icon is up; False where there is none."""
        try:
            import pystray
        except Exception as exc:  # noqa: BLE001 - not installed, or no backend on this desktop
            log.info("no tray icon: %s", exc)
            return False
        colour, tip = state_of(self.status())
        menu = pystray.Menu(
            pystray.MenuItem(_("Ablak megnyitása"), lambda *_: self.show(), default=True),
            pystray.MenuItem(lambda _item: self._tip(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(_("Kilépés"), lambda *_: self.quit()),
        )
        try:
            self.icon = pystray.Icon("dwyt-runner", image(colour), tip, menu)
            self.icon.run_detached()
        except Exception as exc:  # noqa: BLE001
            log.info("no tray icon: %s", exc)
            self.icon = None
            return False
        self.colour, self.tooltip = colour, tip
        threading.Thread(target=self._follow, daemon=True, name="tray").start()
        return True

    def _tip(self) -> str:
        return getattr(self, "tooltip", "").removeprefix(_("Skipwise futtató – ")).capitalize()

    def _follow(self) -> None:
        while not self.stopped.wait(POLL_S):
            try:
                self.update(self.status())
            except Exception:  # noqa: BLE001 - the icon must never take the app down
                log.debug("tray update", exc_info=True)

    def update(self, s: dict[str, Any]) -> None:
        on_main(lambda: self._update(s))

    def _update(self, s: dict[str, Any]) -> None:
        colour, tip = state_of(s)
        if self.icon is None:
            return
        if colour != self.colour:
            if colour == "red":
                self._notify(tip.removeprefix(_("Skipwise futtató – ")))
            self.icon.icon = image(colour)
            self.colour = colour
        if tip != getattr(self, "tooltip", ""):
            self.icon.title = tip
            self.tooltip = tip
            self.icon.update_menu()

    def notify(self, text: str) -> None:
        on_main(lambda: self._notify(text))

    def _notify(self, text: str) -> None:
        try:
            if self.icon is not None and getattr(self.icon, "HAS_NOTIFICATION", False):
                self.icon.notify(text, _("Skipwise futtató"))
        except Exception:  # noqa: BLE001
            log.debug("tray notification", exc_info=True)

    def stop(self) -> None:
        self.stopped.set()
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:  # noqa: BLE001
                pass
