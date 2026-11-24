# -*- coding: utf-8 -*-
"""Screen grab + mouse/keyboard input bound to the game window.

Two ways, chosen call by call:
- the game is the front window: the real mouse and keyboard (as always);
- the game is behind other windows (the user is working elsewhere): the picture via PrintWindow and the
  input posted to the game window as messages, so nothing the user is doing gets disturbed.
"""
import threading
import time

import mss
import numpy as np
from pynput.keyboard import Controller as KB, Key
from pynput.mouse import Button, Controller as Mouse

from . import bg, win

MODS = {"ctrl": Key.ctrl, "shift": Key.shift}


class GameIO:
    def __init__(self, title):
        self.title = title
        self.mouse = Mouse()
        self.kb = KB()
        self.sct = None
        self.hwnd = None
        self.rect = None
        self.found_at = 0.0
        self.fg, self.fg_at = False, 0.0
        self.last_pt, self.last_pt_at = None, 0.0
        self.held_keys = set()
        self.dragging = None                      # "fg" / "bg" while the left button is held

    # ---- window ---------------------------------------------------------------------------------
    def locate(self, fresh=True):
        """fresh=False reuses the window found within the last half second (for fast loops)."""
        if not fresh and self.rect is not None and time.time() - self.found_at < 0.5:
            return True
        self.found_at = time.time()
        self.hwnd = win.find_window(self.title)
        if not self.hwnd:
            self.rect = None
            return False
        if win.is_minimized(self.hwnd):
            bg.restore_behind(self.hwnd)          # a minimised window shows nothing: bring it back, behind
            time.sleep(0.3)
            if win.is_minimized(self.hwnd):
                self.rect = None
                return False
        r = win.client_rect(self.hwnd)
        self.rect = r if r and r[2] >= 50 and r[3] >= 50 else None
        return self.rect is not None

    def front(self):
        """Is the game the front window right now (cached for a fraction of a second)?"""
        now = time.time()
        if now - self.fg_at > 0.2:
            self.fg_at = now
            self.fg = bool(self.hwnd) and win.is_foreground(self.hwnd)
        return self.fg

    def usable(self):
        return self.locate(fresh=False)

    def focus(self):
        """Kept for old callers: nothing needs the game in front any more."""
        return self.locate()

    # ---- picture --------------------------------------------------------------------------------
    def _sct(self):
        """The screen grabber only works in the thread that made it (the phone page reads the screen from
        its own threads): one per thread."""
        loc = self.__dict__.setdefault("_tls", threading.local())
        if getattr(loc, "sct", None) is None:
            loc.sct = mss.mss()
        return loc.sct

    def grab(self):
        if not self.locate():
            return None
        if self.front():
            l, t, w, h = self.rect
            try:
                shot = self._sct().grab({"left": l, "top": t, "width": w, "height": h})
                return np.ascontiguousarray(np.array(shot)[:, :, :3])
            except Exception:
                return bg.capture(self.hwnd)          # the screen grabber failed: read the window itself
        if self.last_pt is not None and time.time() - self.last_pt_at < 2.0:
            # the game may have dropped the hover when the real cursor is elsewhere: point again first
            bg.move(self.hwnd, *self.last_pt, held=self.dragging == "bg")
            time.sleep(0.05)
        return bg.capture(self.hwnd)

    def grab_region(self, x0, y0, x1, y1):
        """Part of the client area only. -> (image, client w, client h)"""
        if not self.locate(fresh=False):
            return None, 0, 0
        l, t, w, h = self.rect
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return None, w, h
        if self.front():
            try:
                shot = self._sct().grab({"left": l + x0, "top": t + y0, "width": x1 - x0, "height": y1 - y0})
                return np.ascontiguousarray(np.array(shot)[:, :, :3]), w, h
            except Exception:
                pass
        img = bg.capture(self.hwnd)
        if img is None or img.shape[:2] != (h, w):
            return None, w, h
        return np.ascontiguousarray(img[y0:y1, x0:x1]), w, h

    def size(self):
        return (self.rect[2], self.rect[3]) if self.locate(fresh=False) else (0, 0)

    # ---- mouse ----------------------------------------------------------------------------------
    def move(self, x, y):
        if not self.rect:
            return
        self.last_pt, self.last_pt_at = (int(x), int(y)), time.time()
        if self.front():
            self.mouse.position = (self.rect[0] + int(x), self.rect[1] + int(y))
        else:
            bg.move(self.hwnd, int(x), int(y), held=self.dragging == "bg")

    def click(self, x, y, mod=None):
        if not self.rect:
            return
        if not self.front():
            self.last_pt, self.last_pt_at = (int(x), int(y)), time.time()
            bg.click(self.hwnd, int(x), int(y), mod)
            return
        key = MODS.get(mod)
        self.move(x, y)
        time.sleep(0.08)
        if key:
            self.kb.press(key)
            time.sleep(0.06)
        self.mouse.press(Button.left)
        time.sleep(0.07)
        self.mouse.release(Button.left)
        if key:
            time.sleep(0.05)
            self.kb.release(key)

    def tap(self, x, y):
        """A quick plain left click (walking by mouse)."""
        if not self.rect:
            return
        if not self.front():
            self.last_pt, self.last_pt_at = (int(x), int(y)), time.time()
            bg.click(self.hwnd, int(x), int(y))
            return
        self.move(x, y)
        time.sleep(0.02)
        self.mouse.press(Button.left)
        time.sleep(0.03)
        self.mouse.release(Button.left)

    def drag_start(self, x, y):
        """Hold the left button down at (x, y): the character keeps running toward the pointer."""
        if not self.rect:
            return False
        if not self.front():
            self.dragging = "bg"
            self.last_pt, self.last_pt_at = (int(x), int(y)), time.time()
            bg.move(self.hwnd, int(x), int(y))
            bg.button(self.hwnd, int(x), int(y), True)
            return True
        self.dragging = "fg"
        self.move(x, y)
        time.sleep(0.02)
        self.mouse.press(Button.left)
        return True

    def drag_end(self):
        try:
            if self.dragging == "bg" and self.hwnd:
                pt = self.last_pt or (0, 0)
                bg.button(self.hwnd, pt[0], pt[1], False)
            elif self.dragging == "fg":
                self.mouse.release(Button.left)
        except Exception:
            pass
        self.dragging = None

    # ---- keyboard -------------------------------------------------------------------------------
    def key_down(self, k):
        self.locate(fresh=False)
        if self.hwnd and not self.front():
            bg.key_down(self.hwnd, k)
            self.held_keys.add(("bg", k))
        else:
            self.kb.press(k)
            self.held_keys.add(("fg", k))

    def key_up(self, k):
        for way in ("bg", "fg"):
            if (way, k) in self.held_keys:
                self.held_keys.discard((way, k))
                try:
                    if way == "bg":
                        if self.hwnd:
                            bg.key_up(self.hwnd, k)
                    else:
                        self.kb.release(k)
                except Exception:
                    pass

    def key(self, k):
        self.key_down(k)
        time.sleep(0.06)
        self.key_up(k)

    def close(self):
        for way, k in list(self.held_keys):
            self.key_up(k)
        if self.dragging:
            self.drag_end()
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None
