# -*- coding: utf-8 -*-
"""Keeps a record of a session: every log line, plus a picture of the game at each step, so a problem
can be looked at afterwards from what the screen really showed."""
import os
import threading
import time

import cv2
import mss
try:
    # mss copies the screen with CAPTUREBLT, which makes Windows hide and redraw the mouse pointer on every
    # copy: with several copies a second the pointer stutters. Without it the game picture is the same.
    import importlib
    for _m in ("mss.windows", "mss.windows.gdi"):          # (mss 9 / mss 10)
        try:
            _mod = importlib.import_module(_m)
            if hasattr(_mod, "CAPTUREBLT"):
                _mod.CAPTUREBLT = 0
        except Exception:
            pass
except Exception:
    pass
import numpy as np

from . import win

KEEP_SESSIONS = 5
MAX_PICTURES = 400


class Recorder:
    def __init__(self, base, title):
        self.root = os.path.join(base, "records")
        self.title = title
        self.dir = None
        self.count = 0
        self.last = {}
        self.lock = threading.Lock()

    def start(self):
        os.makedirs(self.root, exist_ok=True)
        self.dir = os.path.join(self.root, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.count = 0
        old = sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))
        for d in old[:-KEEP_SESSIONS]:
            try:
                for f in os.listdir(os.path.join(self.root, d)):
                    os.remove(os.path.join(self.root, d, f))
                os.rmdir(os.path.join(self.root, d))
            except OSError:
                pass

    def note(self, msg):
        """One log line; a picture too, unless the same line came less than 5 s ago."""
        if not self.dir:
            return
        now = time.time()
        stamp = time.strftime("%H%M%S") + "_%03d" % (int(now * 1000) % 1000)
        pic = ""
        if now - self.last.get(msg, 0) >= 5 and self.count < MAX_PICTURES:
            self.last[msg] = now
            img = self._grab()
            if img is not None:
                pic = stamp + ".jpg"
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])     # full size: needed to read it back
                if ok:
                    buf.tofile(os.path.join(self.dir, pic))
                    self.count += 1
                else:
                    pic = ""
        with self.lock:
            with open(os.path.join(self.dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write("%s  %s%s\n" % (time.strftime("%H:%M:%S"), msg, ("   [" + pic + "]") if pic else ""))

    def _grab(self):
        try:
            hwnd = win.find_window(self.title)
            if not hwnd or win.is_minimized(hwnd):
                return None
            if not win.is_foreground(hwnd):
                from . import bg
                return bg.capture(hwnd)
            l, t, w, h = win.client_rect(hwnd)
            with mss.mss() as sct:
                shot = sct.grab({"left": l, "top": t, "width": w, "height": h})
            return np.ascontiguousarray(np.array(shot)[:, :, :3])
        except Exception:
            return None
