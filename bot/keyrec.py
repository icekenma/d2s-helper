# -*- coding: utf-8 -*-
"""Records the arrow keys the user presses (with their timing) and plays them back to the game."""
import json
import os
import threading
import time

from pynput import keyboard

ARROW_NAMES = {keyboard.Key.up: "up", keyboard.Key.down: "down", keyboard.Key.left: "left",
               keyboard.Key.right: "right"}


class KeyRecorder:
    def __init__(self, path):
        self.path = path
        self.events = []            # [(seconds from the first key, "down"/"up", "up"/"down"/"left"/"right")]
        self.listener = None
        self.t0 = None
        self.held = set()
        self.lock = threading.Lock()
        self.load()

    # ---- file --------------------------------------------------------------------------------
    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.events = [tuple(e) for e in json.load(f).get("events", [])]
        except Exception:
            self.events = []

    def save(self):
        if not self.events:                            # never replace a recording with nothing
            return
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"events": self.events}, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def clear(self):
        self.events = []
        try:
            os.remove(self.path)
        except OSError:
            pass

    # ---- recording ---------------------------------------------------------------------------
    @property
    def recording(self):
        return self.listener is not None

    def start(self):
        self.prev = list(self.events)                  # kept if this recording ends up empty
        self.events, self.t0, self.held = [], None, set()
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        with self.lock:
            now = time.time()
            for k in list(self.held):                  # a key still down when recording ends
                self.events.append((round(now - (self.t0 or now), 3), "up", k))
            self.held.clear()
        if not self.events:
            self.events = getattr(self, "prev", [])    # started and stopped without a key: keep the old one
        self.save()

    def _press(self, key):
        name = ARROW_NAMES.get(key)
        if not name:
            return
        with self.lock:
            if name in self.held:                      # auto-repeat while held: not a new press
                return
            now = time.time()
            if self.t0 is None:
                self.t0 = now                          # the wait before the first key is not kept
            self.held.add(name)
            self.events.append((round(now - self.t0, 3), "down", name))

    def _release(self, key):
        name = ARROW_NAMES.get(key)
        if not name:
            return
        with self.lock:
            if name not in self.held:
                return
            self.held.discard(name)
            self.events.append((round(time.time() - self.t0, 3), "up", name))
            if not self.held:                          # all keys up: keep what we have even if the app closes
                try:
                    self.save()
                except OSError:
                    pass

    def summary(self):
        downs = sum(1 for e in self.events if e[1] == "down")
        length = self.events[-1][0] if self.events else 0.0
        return downs, length
