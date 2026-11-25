# -*- coding: utf-8 -*-
"""Collects set / unique / jewel markers from the minimap and labelled floor loot after the boss,
then smith and portal; uses the waypoint in town. Walks by mouse clicks (or the arrow keys)."""
import os
import threading
import time

import cv2
import numpy as np
from pynput.keyboard import Key

from . import scorer, sorter, spot, win

KIND_KO = {"set": "세트", "unique": "유니크", "chest": "상자", "jewel": "주얼"}
ARROWS = {"up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right}


def _timer_res(on):
    """Windows sleeps in 15.6 ms steps unless asked for 1 ms: needed for exact key timing."""
    try:
        import ctypes
        w = ctypes.windll.winmm
        (w.timeBeginPeriod if on else w.timeEndPeriod)(1)
    except Exception:
        pass


def region(w, h):
    """Part of the client area where the minimap is drawn: (x0, y0, x1, y1)."""
    return int(w * 0.68), int(h * 0.153), w, int(h * 0.60)


def _blobs(mask):
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(stats[i], cent[i]) for i in range(1, n)]


def find_player(img, s):
    """The player is the solid white plus sign. -> (x, y) or None."""
    white = (img.min(axis=2) >= 250)
    best = None
    for st, c in _blobs(white):
        x, y, w, h, area = st
        if not (4.5 * s <= w <= 11 * s and 4.5 * s <= h <= 11 * s) or abs(w - h) > 0.3 * max(w, h):
            continue
        blob = white[y:y + h, x:x + w]
        q = max(1, int(min(w, h) * 0.28))
        corners = blob[:q, :q].sum() + blob[:q, -q:].sum() + blob[-q:, :q].sum() + blob[-q:, -q:].sum()
        if corners > 0:
            continue
        mid_row = blob[h // 2].mean()
        mid_col = blob[:, w // 2].mean()
        if mid_row < 0.85 or mid_col < 0.85:
            continue
        score = area
        if best is None or score > best[0]:
            best = (score, (x + w / 2.0, y + h / 2.0))
    return None if best is None else best[1]


def _star(n, p):
    ys, xs = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    return (np.abs((xs - c) / c) ** p + np.abs((ys - c) / c) ** p) <= 1.0


_STARS = [_star(24, p) for p in (0.5, 0.62, 0.75, 0.9)]


def find_targets(img, s, kinds):
    out = []
    b, g, r = [img[:, :, i].astype(np.int32) for i in range(3)]
    if "set" in kinds:        # pure green diamond outline
        m = (g >= 240) & (r <= 30) & (b <= 30)
        for st, c in _blobs(m):
            if 4 * s <= st[2] <= 26 * s and 4 * s <= st[3] <= 14 * s:
                out.append(("set", float(c[0]), float(c[1])))
    if "chest" in kinds:      # small orange square outline
        m = (r >= 240) & (abs(g - 168) <= 14) & (b <= 30)
        for st, c in _blobs(m):
            if 2.5 * s <= st[2] <= 7 * s and 2.5 * s <= st[3] <= 7 * s and st[4] >= 6 * s:
                out.append(("chest", float(c[0]), float(c[1])))
    if "jewel" in kinds:      # solid blue four-point star (the thin blue plus signs are other things)
        m = ((b >= 240) & (abs(g - 105) <= 16) & (abs(r - 105) <= 16)).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        for st, c in _blobs(m):
            x, y, w, h, area = st
            if not (5 * s <= w <= 13 * s and 5 * s <= h <= 13 * s) or abs(w - h) > 0.4 * max(w, h):
                continue
            if dist[y:y + h, x:x + w].max() < 1.0 * s:
                continue
            blob = cv2.resize(m[y:y + h, x:x + w], (24, 24), interpolation=cv2.INTER_NEAREST).astype(bool)
            if max((blob & t).sum() / float((blob | t).sum()) for t in _STARS) < 0.5:
                continue
            out.append(("jewel", x + w / 2.0, y + h / 2.0))
    if "unique" in kinds:     # translucent tan four-point star (filled, unlike the thin tan plus signs)
        m = ((r - g >= 8) & (r - g <= 40) & (g - b >= 30) & (g - b <= 80) & (r >= 118) & (r <= 225)).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        for st, c in _blobs(m):
            x, y, w, h, area = st
            if not (5.5 * s <= w <= 12 * s and 5.5 * s <= h <= 12 * s) or abs(w - h) > 0.35 * max(w, h):
                continue
            fill = area / float(w * h)
            if not (0.28 <= fill <= 0.62):
                continue
            if dist[y:y + h, x:x + w].max() < 1.25 * s:
                continue
            cy, cx = y + h // 2, x + w // 2
            if not m[cy, cx]:
                continue
            blob = cv2.resize(m[y:y + h, x:x + w], (24, 24), interpolation=cv2.INTER_NEAREST).astype(bool)
            if max((blob & t).sum() / float((blob | t).sum()) for t in _STARS) < 0.62:
                continue                # gold HUD text and scenery are tan too, but not star-shaped
            out.append(("unique", x + w / 2.0, y + h / 2.0))
    return out


def marker_ink(img, kind, x, y, s):
    """How much of a marker's colour is still drawn around (x, y). The player's own cross covers part of
    a marker he is standing on, which breaks the shape test but leaves most of its colour visible.
    -> fraction of what a whole marker would show (about 1.0 = fully there, 0 = gone)."""
    r0 = int(7 * s)
    xi, yi = int(round(x)), int(round(y))
    win_ = img[max(0, yi - r0):yi + r0 + 1, max(0, xi - r0):xi + r0 + 1].astype(np.int32)
    if win_.size == 0:
        return 0.0
    b, g, r = win_[:, :, 0], win_[:, :, 1], win_[:, :, 2]
    if kind == "set":
        n, full = int(((g >= 240) & (r <= 30) & (b <= 30)).sum()), 26.0 * s
    elif kind == "jewel":
        n, full = int(((b >= 240) & (abs(g - 105) <= 16) & (abs(r - 105) <= 16)).sum()), 34.0 * s * s
    else:
        n = int(((r - g >= 8) & (r - g <= 40) & (g - b >= 30) & (g - b <= 80) & (r >= 118) & (r <= 225)).sum())
        full = 34.0 * s * s
    return n / full


def portal_marks(img, s, me=None):
    """Every red oval outline on the minimap that could be the exit portal. -> [(x, y)]"""
    b, g, r = [img[:, :, i].astype(np.int32) for i in range(3)]
    m = (r >= 240) & (abs(g - 77) <= 14) & (abs(b - 77) <= 14)
    out = []
    for st, c in _blobs(m):
        x, y, w, h, area = st
        if 2.2 * s <= w <= 8 * s and 4.5 * s <= h <= 14 * s and 1.3 <= h / float(w) <= 2.7:
            if me is not None and (abs(x + w / 2.0 - me[0]) > 75 * s or abs(y + h / 2.0 - me[1]) > 75 * s):
                continue
            out.append((x + w / 2.0, y + h / 2.0, area))
    return out


def find_portal_mark(img, s, me=None):
    """The exit portal on the minimap: a small upright red oval outline. -> (x, y) or None."""
    marks = portal_marks(img, s, me)
    if not marks:
        return None
    best = max(marks, key=lambda m: m[2])
    return best[0], best[1]


def find_portal_screen(frame, s):
    """The red portal in the game view: a tall red oval ring with a glowing rim. -> (x, y) or None."""
    h, w = frame.shape[:2]
    k = 4
    small = cv2.resize(frame, (w // k, h // k), interpolation=cv2.INTER_AREA)
    b, g, r = [small[:, :, i].astype(np.int32) for i in range(3)]
    m = ((r >= 170) & (g <= 120) & (b <= 90) & (r - g >= 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    best = None
    for st, c in _blobs(m):
        x, y, bw, bh, area = st
        fw, fh = bw * k / s, bh * k / s          # size in layout units
        if not (28 <= fw <= 118 and 50 <= fh <= 178 and 1.15 <= fh / fw <= 2.6):
            continue
        if area * k * k / (s * s) < 900 or area / float(bw * bh) < 0.12:
            continue
        if best is None or bw * bh > best[0]:
            best = (bw * bh, ((x + bw / 2.0) * k, (y + bh / 2.0) * k))
    return None if best is None else best[1]


NPCS = ["찰시", "파라", "흐라틀리", "할부", "라주크"]      # the smith who appears beside the portal, by act


def find_labels(frame, s):
    """Grey name labels drawn in the game view (chests, plain items, townsfolk).
    -> list of (x0, y0, x1, y1) in frame pixels, HUD strips excluded."""
    h, w = frame.shape[:2]
    v = frame.astype(np.int16)
    mx, mn = v.max(axis=2), v.min(axis=2)
    m = ((mx - mn <= 8) & (mx >= 135) & (mx <= 172)).astype(np.uint8)
    m[: int(40 * s), :] = 0
    m[int(h - 115 * s):, :] = 0
    kx, ky = max(3, int(16 * s)) | 1, max(3, int(3 * s)) | 1
    wide = cv2.dilate(m, np.ones((ky, kx), np.uint8))
    out = []
    for st, c in _blobs(wide):
        x, y, bw, bh, area = st
        if not (9 * s <= bh <= 24 * s and 24 * s <= bw <= 260 * s):
            continue
        if m[y:y + bh, x:x + bw].mean() < 0.14:
            continue
        out.append((int(x), int(y), int(x + bw), int(y + bh)))
    return out


LOOT_WORDS = ["정수", "열쇠", "부적"]
SMITHS = ["찰시", "흐라틀리", "할부", "라주크"]      # smiths whose name picture is known (acts 1, 3, 4, 5)


def classify_box(frame, box, s):
    """'npc' (a smith), 'waypoint', 'chest', or 'item' (any other name label). Compares the label's
    picture with known words, so it needs no OCR program."""
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    if x0 > 0.72 * w and y1 < 0.28 * h:
        return ""                                   # the run info in the top right corner
    mask = spot.label_mask(frame[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2], s)
    cols = np.where(mask.sum(axis=0) > 0)[0]
    mw = int(cols[-1] - cols[0] + 1) if len(cols) else 0      # width of the lettering itself
    for n in SMITHS:
        if mw <= spot.width(n) * 1.3 + 6 and spot.has(mask, n, 0.68):
            return "npc"
    if spot.has(mask, "웨이포인트", 0.68):
        return "waypoint"
    if any(spot.has(mask, n, 0.7) for n in LOOT_WORDS if n in spot._words):
        return "loot"                               # floor items worth a detour (no minimap marker)
    if mw <= spot.width("상자") * 1.3 + 6 and spot.has(mask, "상자", 0.68):
        return "chest"
    return "item"


SECTORS = [("right",), ("right", "down"), ("down",), ("down", "left"), ("left",), ("left", "up"), ("up",),
           ("up", "right")]            # every 45 degrees, clockwise from east (screen y grows downward)


def direction(dx, dy, prev=None):
    """Arrow keys to hold for a screen-space vector. With prev (the keys held now) the choice only
    changes once the heading is clearly in the next sector, so the walk does not jitter."""
    if dx == 0 and dy == 0:
        return set()
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    if prev:
        for i, keys in enumerate(SECTORS):
            if set(keys) == set(prev):
                off = abs((ang - i * 45.0 + 180.0) % 360.0 - 180.0)
                if off <= 32.0:
                    return set(keys)
    return set(SECTORS[int(((ang + 22.5) % 360.0) // 45.0)])


def find_wp_flames(frame, s, top=0):
    """A waypoint seen without its name label: the two blue flames on its stone pad, side by side.
    -> point on the pad to click, or None."""
    h, w = frame.shape[:2]
    v = frame.astype(np.int16)
    b, g, r = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    m = ((b >= 220) & (g >= 150) & (b - r >= 50) & (r <= 200)).astype(np.uint8)
    m[: top + int(30 * s), :] = 0
    m[int(h - 115 * s):, :] = 0
    blobs = []
    for st, c in _blobs(cv2.dilate(m, np.ones((5, 5), np.uint8))):
        x, y, bw, bh, area = st
        if 10 * s <= bw <= 32 * s and 10 * s <= bh <= 36 * s and area >= 25 * s * s:
            blobs.append((x + bw / 2.0, y + bh / 2.0))
    best = None
    for i, a in enumerate(blobs):
        for bb in blobs[i + 1:]:
            dx, dy = abs(a[0] - bb[0]) / s, abs(a[1] - bb[1]) / s
            if 85 <= dx <= 135 and dy <= 12:
                p = ((a[0] + bb[0]) / 2.0, (a[1] + bb[1]) / 2.0 + 25 * s)
                d = (p[0] - w / 2.0) ** 2 + (p[1] - h / 2.0) ** 2
                if best is None or d < best[0]:
                    best = (d, p)
    return None if best is None else best[1]


LOOT_SEARCH = ["정수", "열쇠", "부적"]      # floor items without a minimap marker that are always worth taking
EXPLORE = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]



ROUTE_IDLE_MAX = 1.5     # s: a longer pause with no key held is cut short (the recording was left running)
ROUTE_MAX = 90.0         # s: never play longer than this


def squeeze_route(events):
    """-> (events, seconds cut): pauses with nothing held shortened, nothing before the first key, at most
    ROUTE_MAX long. Moves themselves keep their exact timing."""
    out, held, shift, prev = [], set(), 0.0, None
    for t, kind, name in events:
        t = float(t)
        if prev is None:
            shift = t - 0.1
        elif not held and t - prev > ROUTE_IDLE_MAX:
            shift += (t - prev) - 0.3
        prev = t
        nt = t - shift
        if nt > ROUTE_MAX:
            break
        out.append((nt, kind, name))
        if kind == "down":
            held.add(name)
        else:
            held.discard(name)
    total = float(events[-1][0]) if events else 0.0
    return out, max(0.0, total - (out[-1][0] if out else 0.0))

class Picker:
    """Call step() several times a second from one thread; call release() when stopping.

    Phases: town (use the waypoint) -> hunt (hands off) -> collect (after the boss: minimap markers and
    floor essences / keys / charms) -> smith (only when the bags hold something) -> portal.
    """

    def __init__(self, io, opts, log, state=None):
        self.io, self.log = io, log
        self.opts = opts
        self.state = state if state is not None else {}
        self.kinds = set(k for k in ("set", "unique", "jewel") if opts.get(k, False))   # chests are never opened
        self.mouse = bool(opts.get("mouse_move", True))
        self.held = set()
        self.dragging = False
        self.target = None          # (kind, x, y) in minimap-region pixels
        self.since = 0.0
        self.ignore = []            # (x, y, until)
        self.count = 0
        self.history = []           # target lists of the last two frames
        self.labels, self.label_time = [], 0.0
        self.idle_since = None
        self.over_since = None      # when the hunt was seen to be over in this run
        self.stage_start = None     # when collecting finished and the wrap-up began
        self.npc_since, self.npc_seen, self.npc_note = None, False, 0.0
        self.gave_up = False
        self.last_click = 0.0
        self.wp_time, self.wp_list_tab, self.wp_main_tab, self.wp_miss = 0.0, None, None, 0
        self.wp_frontier = False
        self.town_label, self.town_since, self.explore_i, self.explore_at = 0.0, None, 0, 0.0
        self.moved_from, self.moved_at = None, 0.0
        self.last_gain = 0.0
        self.npc_look, self.npc_hits, self.gate_at = 0.0, [], 0.0
        self.tries, self.arrived, self.clean_seen = {}, None, 0.0
        self.vel, self.prev_me, self.prev_at, self.near_since, self.ring = None, None, 0.0, None, 0
        self.frame, self.portal, self.slow_at, self.busy = None, None, 0.0, False
        self.panel_at, self.top, self.forced = 0.0, 0, False
        self.portal_note, self.inv_open_since = 0.0, None
        self.panel_since, self.panel_esc = None, 0.0
        self.wd_at, self.wd_note = 0.0, 0.0
        self.route_played, self.route_note, self.route_end = False, False, None
        self.state["route_done"], self.state["route_len"] = False, 0.0
        self.mark_off, self.mark_at, self.back_note, self.psweep = None, 0.0, 0.0, None
        self.portal_w, self.home, self.prev_rel, self.cur_rel = None, None, None, None
        self.lost_goal, self.other_since = None, None
        self.ess_t0, self.ess_done, self.ess_seen = None, False, False
        self.ess_mem, self.ess_note = [], 0.0
        self.sweep_pts, self.sweep_done, self.sweep_i, self.sweep_t0, self.sweep_at = None, False, 0, 0.0, 0.0
        self.ss_done, self.ss_home, self.tab_tried, self.ss_pts = False, None, 0, None
        self.exit_at, self.exit_tries = 0.0, 0
        self.pos, self.odo_prev, self.odo_win, self.goal_mm, self.click_pos, self.click_mm = None, None, None, (0.0, 0.0), None, (0.0, 0.0)
        self.why, self.why_since, self.why_note = "", 0.0, 0.0
        self.portal_first, self.portal_last, self.target_seen = 0.0, 0.0, 0.0
        self.nomap_note, self.nomap_since, self.hunt_tab, self.hunt_tab_at = 0.0, None, 0, 0.0
        self.town_fix, self.town_fix_n = 0.0, 0
        self.title_at, self.title_click = 0.0, 0.0
        self.reroll_n = 0
        self.fast_counted = False
        self.act_done, self.act_t0, self.act_votes, self.act_at, self.rerolling = False, None, [], 0.0, False
        self.act_name = ""
        # mouse walking
        self.ratio = 20.5           # screen pixels per minimap pixel (learnt while walking)
        self.click_at, self.click_me, self.click_d, self.clamped = 0.0, None, None, True
        self.still_since, self.last_me, self.mouse_ok, self.mouse_fail, self.block_fail = 0.0, None, False, 0, 0
        # where we are
        self.terror_last, self.terror_ever, self.run_text = 0.0, False, False
        self.wrong_note = 0.0
        self.portal_click, self.gone_since = 0.0, None
        # wrap-up
        self.inv_check, self.need_smith = None, True
        self.loot_look, self.loot_hits, self.loot_tries, self.loot_goal = 0.0, [], {}, None

    # ---- keys and mouse ---------------------------------------------------------------------
    def _hold(self, keys):
        for k in self.held - keys:
            self.io.key_up(ARROWS[k])
        for k in keys - self.held:
            self.io.key_down(ARROWS[k])
        self.held = set(keys)

    def release(self):
        self._hold(set())
        self._undrag()

    def _undrag(self):
        if self.dragging:
            self.dragging = False
            self.io.drag_end()

    def _toward(self, x, y, frame, s, stop_at=5.0):
        h, w = frame.shape[:2]
        dx, dy = x - w / 2.0, y - h / 2.0
        if abs(dx) < stop_at * s and abs(dy) < stop_at * s:
            self.release()
            return True
        self._hold(direction(dx, dy, self.held))
        return False

    def _safe(self, x, y, w, h, s, portal_ok=False):
        """Pull a click point back inside the playfield (clear of the bottom bar) and away from the
        portal. -> (x, y, clamped)"""
        cx, cy = w / 2.0, h / 2.0
        dx, dy = x - cx, y - cy
        k = 1.0
        lim = ((0.05 * w - cx, 0.95 * w - cx), (0.08 * h - cy, h - 125 * s - cy))
        for d, (lo, hi) in ((dx, lim[0]), (dy, lim[1])):
            if d > hi:
                k = min(k, hi / d)
            elif d < lo:
                k = min(k, lo / d)
        x, y = cx + dx * k, cy + dy * k
        if not portal_ok:
            x, y = self._away(x, y, w, h, s)
        return x, y, k < 0.999

    def _portal_zones(self, w, h, s):
        """Screen areas covered by the exit portal (a click or a held button there would take it):
        [(centre x, centre y, half width, half height)]."""
        pts = []
        if self.portal is not None:
            pts.append((self.portal[0], self.portal[1] + 60 * s))          # its ring centre -> its foot
        elif self.ss_home is not None and self.pos is not None and not self.ss_done:
            # not recognised on this frame (half off the screen, covered): where it must be
            cy = self.top + (h - self.top) / 2.0
            pts.append((w / 2.0 - self.pos[0] - self.ss_home[0], cy - self.pos[1] - self.ss_home[1] + 60 * s))
        if self.mark_off is not None and time.time() - self.mark_at < 0.6:
            pts.append((w / 2.0 + self.mark_off[0] * self.ratio, h / 2.0 + self.mark_off[1] * self.ratio))
        return [(px, py - 55 * s, 85 * s, 115 * s) for px, py in pts]

    def _away(self, x, y, w, h, s):
        """Turn a pointer/click point off the portal, keeping its distance from the character."""
        zones = self._portal_zones(w, h, s)

        def bad(px, py):
            return any(abs(px - zx) < hw and abs(py - zy) < hh for zx, zy, hw, hh in zones)

        if not zones or not bad(x, y):
            return x, y
        cx, cy = w / 2.0, h / 2.0
        dx, dy = x - cx, y - cy
        for deg in (35, -35, 70, -70, 105, -105, 140, -140, 180):
            a = np.radians(deg)
            nx = cx + dx * np.cos(a) - dy * np.sin(a)
            ny = cy + dx * np.sin(a) + dy * np.cos(a)
            if not bad(nx, ny) and 0.05 * w < nx < 0.95 * w and 0.08 * h < ny < h - 125 * s:
                return nx, ny
        return cx + dx * 0.3, cy + dy * 0.3

    def _walk_to(self, x, y, frame, s, now, portal_ok=False):
        """Mouse: click the ground at a screen point (the character walks there by himself).
        portal_ok: the click is meant to bring him to the portal (no steering away from it)."""
        h, w = frame.shape[:2]
        x, y, clamped = self._safe(x, y, w, h, s, portal_ok)
        self._world_click(x, y, portal_ok=portal_ok)
        self.click_at = now
        return clamped

    def _panel_open(self, frame):
        """Any game window over the world view (smith/stash, inventory, waypoint list)?"""
        try:
            norm = sorter.Geo(frame).normalize(frame)
        except Exception:
            return True
        return (sorter.seen(norm, "panel") >= 0.6 or sorter.seen(norm, "inv") >= 0.7
                or sorter.seen(norm, "stash") >= 0.7 or sorter.seen(norm, "wp") >= 0.75)

    def _world_click(self, x, y, portal_ok=False):
        """A click meant for the ground, an item label, the smith or the portal. Never sent while any
        window is open: there it would land on the equipment or inventory (it could pick up a worn item).
        -> True when sent."""
        self._undrag()
        f = self.io.grab()
        if f is not None and not portal_ok:
            x, y = self._away(x, y, f.shape[1], f.shape[0], (f.shape[0] - self.top) / 600.0)
        if f is None or self._panel_open(f):
            self.panel_at = time.time()
            self.release()
            return False
        self.io.tap(int(x), int(y))
        return True

    # ---- screen reading ---------------------------------------------------------------------
    def _labels(self, frame, s, now):
        """Readable name labels on screen, refreshed about twice a second: [(kind, cx, bottom)]."""
        if now - self.label_time >= 0.35:
            self.label_time = now
            out = []
            for box in find_labels(frame, s)[:10]:
                kind = classify_box(frame, box, s)
                if kind:
                    out.append((kind, (box[0] + box[2]) / 2.0, box[3]))
            self.labels = out
        return self.labels

    def _new_run(self):
        self.over_since, self.stage_start = None, None
        self.portal_first, self.gate_at = 0.0, 0.0
        self.target, self.idle_since, self.ignore, self.history = None, None, [], []
        self.tries, self.arrived = {}, None
        self.npc_since, self.npc_seen, self.gave_up = None, False, False
        self.inv_check, self.need_smith = None, True
        self.loot_tries, self.loot_goal, self.loot_hits = {}, None, []
        self.portal_click, self.gone_since = 0.0, None
        self.town_since = None
        self.forced = False
        self.route_played, self.route_note, self.route_end = False, False, None
        self.state["route_done"], self.state["route_len"] = False, 0.0
        self.sweep_pts, self.sweep_done = None, False
        self.ss_done, self.ss_home, self.tab_tried, self.ss_pts = False, None, 0, None
        self.nomap_since, self.hunt_tab = None, 0
        self.fast_counted = False
        self.act_done, self.act_t0, self.act_votes, self.act_at, self.rerolling = False, None, [], 0.0, False
        self.act_name = ""
        self.exit_at, self.exit_tries = 0.0, 0
        self.ess_t0, self.ess_done, self.ess_seen = None, False, False
        self.ess_mem, self.ess_note = [], 0.0
        self.pos, self.odo_prev = None, None
        self.mark_off, self.psweep, self.portal_w, self.home, self.prev_rel = None, None, None, None, None
        self.state["over"] = False

    # ---- town -------------------------------------------------------------------------------
    def _wp_rows(self, frame, geo, n=9, row1=None, pitch=None):
        """What each row of the open waypoint list says: [(reads '완료한', reads '공포 영역')]."""
        row1 = sorter.WP_ROW1 if row1 is None else row1
        pitch = sorter.WP_PITCH if pitch is None else pitch
        out = []
        for k in range(n):
            ny = row1 + pitch * k
            x0, y0 = geo.raw(132, ny - 13)
            x1, y1 = geo.raw(330, ny + 11)
            m = spot.area_mask(frame[max(0, y0):y1, max(0, x0):x1], geo.s, gold=True)
            out.append((spot.score(m, "완료한"), spot.score(m, "목록공포")))
        return out

    def _wp_menu(self, frame, s, now):
        """Waypoint list open: open the terror zone list and click the configured stage, checking what
        each row says before clicking it. -> True when the list is showing."""
        geo = sorter.Geo(frame)
        norm = geo.normalize(frame)
        if sorter.seen(norm, "wp") < 0.75 or sorter.seen(norm, "panel") >= 0.6 or sorter.seen(norm, "inv") >= 0.7:
            self.wp_list_tab = None
            return False
        self.release()
        if now - self.wp_time < 0.9:
            return True
        self.io.move(*geo.raw(600, 330))              # keep the cursor off the list (it recolours a row)
        time.sleep(0.12)
        frame = self.io.grab()
        if frame is None:
            return True
        rows = self._wp_rows(frame, geo)
        stage = int(self.state.get("target", self.opts.get("town_stage", 0)) or 0)
        main = sorter.seen(geo.normalize(frame), "wp_main") >= 0.75
        self.wp_time = now
        if main:
            self.wp_list_tab = None
            # '완료한 단계 다시 하기': the row that reads it clearly better than all the others
            # (the newest stage's entry is two lines tall, so this row is not at a fixed place)
            order = sorted(range(len(rows)), key=lambda k: -rows[k][0])
            best = order[0]
            done = [best] if rows[best][0] >= 0.55 and rows[best][0] - rows[order[1]][0] >= 0.2 else []
            if stage <= 0 or self.wp_frontier:
                # the newest open stage: the first '공포 영역' row above '완료한 단계 다시 하기'
                stop_at = done[0] if done else len(rows)
                fr = [k for k in range(stop_at) if rows[k][1] >= 0.6 and rows[k][0] < 0.55]
                if fr:
                    self.wp_frontier = False
                    self.wp_miss = 0
                    self.io.click(*geo.raw(sorter.WP_X, sorter.WP_ROW1 + sorter.WP_PITCH * fr[0]))
                    self.log("웨이포인트: 새로 열린 공포 영역(%d번째 줄) 시작" % (fr[0] + 1))
                    return True
                if stage <= 0:
                    self.wp_miss += 1
                    if self.wp_miss % 5 == 1:
                        self.log("웨이포인트: '공포 영역' 줄을 찾지 못함 (일치 %.2f) - 다시 확인"
                                 % max(r[1] for r in rows))
                    return True
            if done:
                self.wp_miss = 0
                self.io.click(*geo.raw(sorter.WP_X, sorter.WP_ROW1 + sorter.WP_PITCH * done[0]))
                self.log("웨이포인트: '완료한 단계 다시 하기' 열기")
                return True
            # this tab lists ordinary areas: look through the other tabs for the terror zone entries
            self.wp_miss += 1
            self.wp_main_tab = 1 if self.wp_main_tab is None else (self.wp_main_tab + 1) % 5
            self.io.click(*geo.raw(sorter.WP_TAB1 + sorter.WP_TAB_PITCH * self.wp_main_tab, sorter.WP_TAB_Y))
            self.log("웨이포인트: 이 탭에서 '완료한 단계 다시 하기'를 못 찾음 (일치 %.2f) → %d번째 탭 확인"
                     % (rows[best][0], self.wp_main_tab + 1))
            return True
        # the stage list (tabs 1-10, 11-20, 21-28)
        per = sorter.WP_PER_TAB
        rows = self._wp_rows(frame, geo, per, sorter.WP_LIST_ROW1, sorter.WP_LIST_PITCH)
        if stage <= 0 or self.wp_frontier:
            self.io.key(Key.esc)                      # the newest stage is on the first page: close and reopen
            self.wp_list_tab = None
            self.log("웨이포인트: 단계 목록이 열려 있어 닫고 첫 화면에서 다시 고름")
            return True
        tab, row = (stage - 1) // per, (stage - 1) % per
        lit = sorter.wp_lit_tab(geo.normalize(frame))
        if self.wp_list_tab != tab or lit != tab:
            # click the stage's tab and do not go on until that tab really is the open one
            # (otherwise row 1 of the 1~9 tab = stage 1 would be clicked for stage 10)
            self.wp_tab_tries = (getattr(self, "wp_tab_tries", 0) + 1) if self.wp_list_tab == tab else 1
            self.wp_list_tab = tab
            if self.wp_tab_tries > 6:
                self.wp_tab_tries = 0
                self.wp_list_tab = None
                self.io.key(Key.esc)
                self.log("웨이포인트: %d~%d단계 탭이 열리지 않음 (열린 탭: %s) - 목록을 닫고 다시"
                         % (tab * per + 1, tab * per + per, "모름" if lit is None else "%d~%d" % (lit * per + 1, lit * per + per)))
                return True
            self.io.click(*geo.raw(sorter.WP_TAB1 + sorter.WP_TAB_PITCH * tab, sorter.WP_TAB_Y))
            self.log("웨이포인트: %d~%d단계 탭 열기" % (tab * per + 1, tab * per + per))
            return True
        self.wp_tab_tries = 0
        if rows[row][1] < 0.6:
            self.wp_miss += 1
            if self.wp_miss >= 2:
                # that stage is not among the cleared ones (e.g. lost again by a death): play the newest
                # open stage instead, the top line of the first page
                self.wp_miss, self.wp_frontier, self.wp_list_tab = 0, True, None
                self.io.key(Key.esc)
                self.log("웨이포인트: 공포 영역 %d은 아직 깬 단계 목록에 없음 → 목록 맨 위 새 단계로 진행" % stage)
                return True
            self.wp_list_tab = None                   # click the tab again, then look once more
            self.log("웨이포인트: %d번째 줄이 비어 있거나 '공포 영역'이 아님 (일치 %.2f)" % (row + 1, rows[row][1]))
            return True
        self.wp_miss = 0
        self.io.click(*geo.raw(sorter.WP_X, sorter.WP_LIST_ROW1 + sorter.WP_LIST_PITCH * row))
        self.log("웨이포인트: 공포 영역 %d 시작" % stage)
        self.wp_list_tab = None
        return True

    def _wp_label(self, frame, s, now):
        """In town: click the waypoint under its name label. -> True when the label is on screen."""
        wps = [l for l in self._labels(frame, s, now) if l[0] == "waypoint"]
        if not wps:
            return False
        self.release()
        if now - self.wp_time > 3:
            self._world_click(wps[0][1], wps[0][2] + 74 * s)
            self.wp_time = now
            self.log("마을: 웨이포인트 클릭")
        return True

    def _town(self, frame, s, now, me):
        """Not in a terror zone and the hunt is not on: find and use the waypoint."""
        if not self.opts.get("town_start"):
            return
        if now - self.terror_last < 3:
            self.town_since = self.other_since = None
            return
        if not self.run_text:
            self.other_since = None
        if self.run_text:                             # some other run (not a terror zone): the engine
            self.town_since = None                    # goes back to town when it ends
            if self.other_since is None:
                self.other_since = now
            if self.terror_ever and now - self.other_since > 15 and now - self.wrong_note > 60:
                self.wrong_note = now
                self.log("공포 영역이 아닌 판에 들어와 있음: 이 판이 끝나면 마을로 돌아가 다시 고름")
            return
        if self.town_since is None:
            self.town_since = now
        if now - self.town_label >= 0.8:
            self.town_label = now
            if self._wp_label(frame, s, now):
                self.explore_at = 0.0
                return
            pad = find_wp_flames(frame, s, self.top)
            if pad is not None:
                self.explore_at = 0.0
                if now - self.wp_time > 3 and self._world_click(pad[0], pad[1]):
                    self.wp_time = now
                    self.log("마을: 웨이포인트(파란 불꽃) 클릭")
                return
        # no waypoint in view: walk around the town a little to bring it on screen (the minimap is not
        # needed for that: the walk goes by the arrow keys)
        if now - self.town_since < 5 or now - self.wp_time < 4:
            return
        if now - self.town_since > 25 and now - self.town_fix > 30:
            # long without a waypoint in view: a window or the big map may be covering the town
            self.town_fix = now
            self.release()
            if sorter.seen(sorter.Geo(frame).normalize(frame), "panel") >= 0.8:
                self.io.key(Key.esc)
                self.log("마을: 25초 넘게 웨이포인트가 안 보임 - Esc로 열린 창 닫기")
            else:
                self.io.key(Key.tab)
                self.log("마을: 25초 넘게 웨이포인트가 안 보임 - Tab으로 지도 표시 바꿔 봄")
            self.town_fix_n += 1
            return
        if now - self.explore_at >= 1.6:
            self.explore_at = now
            dx, dy = EXPLORE[self.explore_i % len(EXPLORE)]
            self.explore_i += 1
            if self.explore_i % len(EXPLORE) == 1:
                self.log("마을: 웨이포인트 이름표가 안 보여 주변을 둘러보는 중")
            h, w = frame.shape[:2]
            if self.mouse:
                self._walk_to(w / 2.0 + dx * 0.3 * w, h / 2.0 + dy * 0.28 * h, frame, s, now)
            else:
                self._hold(direction(dx, dy))
        elif not self.mouse and now - self.explore_at >= 1.0:
            self.release()

    # ---- smith ------------------------------------------------------------------------------
    def _go_smith(self, frame, s, now, me, portal, mark, why):
        """Walk onto the smith; their window opens by itself and the engine tidies up."""
        self.state["want_smith"] = time.time()            # this time the window is wanted
        if self.npc_since is None:
            self.npc_since, self.npc_seen = now, False
            self.log("대장장이에게 이동 (%s)" % why)
        waited = now - self.npc_since
        if now - self.npc_look >= 0.4:
            self.npc_look = now
            self.npc_hits = [("npc", x, y) for _, _, x, y in spot.search(frame, s, SMITHS)]
        npc = self.npc_hits
        if not npc and waited > 20:
            npc = [l for l in self._labels(frame, s, now) if l[0] == "item"]   # a smith whose name is not known yet
        if npc:
            h, w = frame.shape[:2]
            t = min(npc, key=lambda l: (l[1] - w / 2.0) ** 2 + (l[2] - h / 2.0) ** 2)
            if not self.npc_seen:
                self.npc_seen = True
            wobble = (40, 25, 55, 0)[int(waited // 5) % 4]     # the trigger spot is at their feet
            if self.mouse:
                if now - self.click_at >= 0.6:
                    self._world_click(t[1], t[2] + wobble * s)      # clicking the smith walks up to him
                    self.click_at = now
                self.state["npc_dir"] = ()
            else:
                self._toward(t[1], t[2] + wobble * s, frame, s, stop_at=1.5)
                self.state["npc_dir"] = tuple(self.held) or self.state.get("npc_dir", ())
            return
        self.npc_seen = False
        if waited > 15 and now - self.npc_note > 20:
            self.npc_note = now
            self.log("대장장이 이름표가 화면에 보이지 않음: 포탈 주변을 돌며 찾는 중")
        ang = (waited % 16) / 16.0 * 6.283                     # the smith stands somewhere beside the portal
        if portal is not None:
            px, py = portal[0] + 150 * s * np.cos(ang), portal[1] + 110 * s * np.sin(ang)
            if self.mouse:
                if now - self.click_at >= 0.8:
                    self._walk_to(px, py, frame, s, now)
            else:
                self._toward(px, py, frame, s, stop_at=6.0)
        elif mark is not None:
            self._mark_walk(me, mark, frame, s, now)
        elif self._portal_vec(now) is not None:
            v = self._portal_vec(now)
            self._head(frame, s, now, v[0], v[1])
        else:
            self.release()

    def _mark_walk(self, me, pt, frame, s, now, portal_ok=True):
        """Head for a minimap point (keys or mouse): a click a short way toward it, in the middle part of
        the screen (never on the skill icons along the top)."""
        if self.mouse:
            if now - self.click_at >= 0.35:
                h, w = frame.shape[:2]
                dx, dy = (pt[0] - me[0]) * self.ratio, (pt[1] - me[1]) * self.ratio
                d = float(np.hypot(dx, dy)) or 1.0
                k = min(1.0, 0.28 * min(w, h) / d)
                self._walk_to(w / 2.0 + dx * k, h / 2.0 + dy * k, frame, s, now, portal_ok=portal_ok)
        else:
            self._hold(direction(pt[0] - me[0], pt[1] - me[1], self.held))

    # ---- wrap-up ----------------------------------------------------------------------------
    def _count_inventory(self):
        """Open the inventory (C), count the filled cells, close it. -> count, or -1 if it did not open."""
        frame = self.io.grab()
        if frame is None:
            return -1
        geo = sorter.Geo(frame)
        opened = False
        if sorter.seen(geo.normalize(frame), "inv") < 0.7:
            self.io.key("c")
            opened = True
        norm = None
        for _ in range(6):
            time.sleep(0.15)
            f = self.io.grab()
            if f is not None and sorter.seen(geo.normalize(f), "inv") >= 0.7:
                norm = geo.normalize(f)
                break
        if norm is None:
            time.sleep(0.5)                           # it may still open late: never leave it open
            f = self.io.grab()
            if f is not None and sorter.seen(geo.normalize(f), "inv") >= 0.7:
                self.io.key(Key.esc)
                time.sleep(0.2)
            return -1
        self.io.move(*geo.raw(*sorter.PARK))
        time.sleep(0.12)
        f = self.io.grab()
        if f is not None:
            norm = geo.normalize(f)
        n = sum(1 for row in sorter.read_inventory(norm) for v in row if v)
        if opened:
            self.io.key("c")
            time.sleep(0.25)
            f = self.io.grab()
            if f is not None and sorter.seen(geo.normalize(f), "inv") >= 0.7:
                self.io.key(Key.esc)
                time.sleep(0.2)
        return n

    def _finish(self, frame, s, now, me, portal, mark):
        """Nothing left to pick up: the smith only if the bags hold something, then the portal."""
        if self.stage_start is None:
            self.stage_start, self.gave_up = now, False
            self.npc_since = None
            self.log("주울 것 없음: 마무리")
        if (self.opts.get("auto_clean") and self.inv_check is None and self.route_end is not None
                and self.state.get("cleaned", 0) > self.route_end):
            self.inv_check, self.need_smith = self.state["cleaned"], False   # tidied at the end of the route
            self.log("방향키 재생 끝에 대장장이 정리 완료: 바로 포탈로")
        if self.opts.get("auto_clean") and self.inv_check is None:
            self.release()
            if self.last_gain > max(self.over_since or 0, self.state.get("cleaned", 0)) + 0.01:
                self.inv_check, self.need_smith = now, True     # picked something up: no need to look
                return
            n = self._count_inventory()
            self.inv_check = time.time()
            self.frame = None                       # the screen changed: grab a fresh one next step
            if n == 0:
                self.need_smith = False
                self.log("인벤토리가 비어 있음: 대장장이 없이 바로 포탈로")
            elif n > 0:
                self.need_smith = True
                self.log("인벤토리 %d칸 사용 중: 대장장이에서 정리" % n)
            else:
                self.need_smith = True
                self.log("인벤토리를 열어 보지 못함: 대장장이에서 정리")
            return
        if (self.opts.get("auto_clean") and self.need_smith and not self.gave_up
                and self.state.get("cleaned", 0) < (self.inv_check or 0)):
            if self.npc_since is not None and now - self.npc_since > 120:
                self.gave_up = True
                self.release()
                self.log("2분 동안 대장장이 창을 열지 못함: 이번만 정리 없이 포탈로 이동")
                return
            self._go_smith(frame, s, now, me, portal, mark, "인벤토리 정리")
            return
        if (self.over_since is not None and self.state.get("cleaned", 0) > self.over_since
                and self.exit_tries < 6):
            self._exit_town(now)                       # tidied up at the smith: Esc -> town -> next run
            return
        self._portal_now(frame, s, now, me, portal, mark)

    def _act_check(self, frame, now):
        """패스트 모드: read the act of the new run a few times; acts 3 and 4 give two essences."""
        if self.act_t0 is None:
            self.act_t0, self.act_votes = now, []
        try:
            act, sc = sorter.read_act(frame, self.top, os.path.dirname(os.path.dirname(os.path.abspath(sorter.__file__))))
        except Exception:
            act, sc = None, {}
        if sc:
            self.act_votes.append(act if act else (0 if max(sc.values()) < 0.7 else None))
        v = [x for x in self.act_votes if x is not None]
        good = sum(1 for x in v if x in (3, 4))
        bad = sum(1 for x in v if x in (0, 1, 2, 5))
        if good >= 2:
            self.act_done, self.reroll_n = True, 0
            a = max((3, 4), key=lambda k: v.count(k))
            self.log("패스트 모드: %d막 - 이 판 진행 (정수 2개)" % a)
        elif bad >= 3 and good == 0:
            self.act_done = True
            k = max(set(v), key=v.count)
            self.act_name = "%d막" % k if k else "1막이나 2막"
            if self.reroll_n >= 10:
                self.reroll_n = 0
                self.log("패스트 모드: %s이지만 10번 연속 다시 뽑아서 이번 판은 그대로 진행" % self.act_name)
            else:
                self.reroll_n += 1
                self.rerolling = True
                self.exit_tries, self.exit_at = 0, 0.0
        elif now - self.act_t0 > 7:
            self.act_done = True
            self.log("패스트 모드: 몇 막인지 확실히 읽지 못해 이번 판은 그대로 진행 (읽은 값: %s)"
                     % (", ".join(str(x) for x in self.act_votes[-8:]) or "없음"))

    def _fast_count(self):
        """패스트 모드: open the inventory, read the burning-essence stacks, close it again."""
        frame = self.io.grab()
        if frame is None:
            return
        geo = sorter.Geo(frame)
        opened = False
        if sorter.seen(geo.normalize(frame), "inv") < 0.7:
            self.io.key("c")
            opened = True
        norm = None
        for _ in range(8):
            time.sleep(0.15)
            f = self.io.grab()
            if f is not None and sorter.seen(geo.normalize(f), "inv") >= 0.7:
                norm = geo.normalize(f)
                break
        if norm is not None:
            self.io.move(*geo.raw(*sorter.PARK))
            time.sleep(0.12)
            f = self.io.grab()
            if f is not None:
                norm = geo.normalize(f)
            snap, bad = sorter.read_burning(self.io, geo, norm, os.path.dirname(os.path.dirname(os.path.abspath(sorter.__file__))))
            if bad:
                self.log("패스트 모드: 불타는 정수 묶음 %d개의 수량을 읽지 못함 (게임 창이 뒤에 있으면 정보창이 안 뜸)" % bad)
            else:
                sorter.record_essences(snap, set())
                self.log("패스트 모드: 불타는 정수 인벤토리에 %d개" % sum(snap.values()))
        else:
            self.log("패스트 모드: 인벤토리가 열리지 않아 정수를 세지 못함")
        if opened:
            self.io.key("c")
            time.sleep(0.25)
            f = self.io.grab()
            if f is not None and sorter.seen(geo.normalize(f), "inv") >= 0.7:
                self.io.key(Key.esc)
                time.sleep(0.2)

    def _exit_town(self, now, first="창고 정리 끝: Esc로 마을로 가서 다음 판"):
        """Esc -> "저장 후 나가기": back to town, where the waypoint starts the next run."""
        if now - self.exit_at < 4.0:
            return
        self.exit_at = now
        self.exit_tries += 1
        self.release()
        if self.exit_tries == 1:
            self.log(first)
        if sorter.leave_via_menu(self.io, self.log, lambda: self.io.key(Key.esc),
                                 {"down": Key.down, "up": Key.up, "enter": Key.enter, "tab": Key.tab},
                                 attempt=self.exit_tries):
            self.state["over"] = False
            self.state["run_reset"] = time.time()
            self.state["town_check"] = time.time() + 40
            self.frame = None
            self.log("마을로 나감: 저장 후 나가기 (다음 판)")
            return
        self.frame = None
        if self.exit_tries >= 6:
            self.log("Esc 메뉴로 마을에 가지 못함 (6번): 포탈로 다음 판")

    def _portal_now(self, frame, s, now, me, portal, mark):
        self.npc_since = None
        if portal is not None:
            self.release()
            h, w = frame.shape[:2]
            geo = sorter.Geo(frame)
            nx, ny = (portal[0] - geo.x0) / geo.s, (portal[1] - geo.top) / geo.s
            far = np.hypot(portal[0] - w / 2.0, portal[1] - h / 2.0) > 0.32 * min(w, h)
            # the buff/skill icons in the top left corner and the bottom bar take clicks themselves
            covered = (nx < 300 and ny < 165) or ny > 470
            if far or covered:
                # walk toward it first; once it is near the middle of the screen it is clicked
                if now - self.click_at >= 0.6:
                    gx = w / 2.0 + (portal[0] - w / 2.0) * 0.7
                    gy = h / 2.0 + (portal[1] + 40 * s - h / 2.0) * 0.7
                    if self.mouse:
                        dx, dy = gx - w / 2.0, gy - h / 2.0
                        d = float(np.hypot(dx, dy)) or 1.0
                        k = min(1.0, 0.28 * min(w, h) / d)
                        self._walk_to(w / 2.0 + dx * k, h / 2.0 + dy * k, frame, s, now, portal_ok=True)
                    else:
                        self._toward(portal[0], portal[1] + 40 * s, frame, s, stop_at=40)
                    if now - self.portal_note > 10:
                        self.portal_note = now
                        self.log("포탈 쪽으로 이동")
                return
            if now - self.last_click > 2.0:
                if self._world_click(portal[0], portal[1] + 15 * s, portal_ok=True):     # the lower half of the ring
                    self.last_click = self.portal_click = now
                    self.log("포탈 클릭: 다음 판으로")
        elif mark is not None:
            self._mark_walk(me, mark, frame, s, now)
        elif self._portal_vec(now) is not None:
            v = self._portal_vec(now)
            self._head(frame, s, now, v[0], v[1])
        else:
            self.release()

    # ---- main -------------------------------------------------------------------------------
    def _title_screen(self, frame, now):
        """The game's main menu (after '저장 후 나가기' in town): '오프라인 플레이', then '시작' to play on."""
        self.title_at = now
        norm = sorter.Geo(frame).normalize(frame)
        off = sorter.seen(norm, "title_offline")
        st = min(sorter.seen(norm, "title_start"), sorter.seen(norm, "title_saves"))
        if off >= 0.82 and off > st + 0.1:
            spot_ = sorter.TITLE_OFFLINE
            what = "게임 첫 화면: '오프라인 플레이' 클릭"
        elif st >= 0.82 and st > off + 0.1:
            spot_ = sorter.TITLE_START
            what = "게임 첫 화면: '시작' 클릭 (이어서 사냥)"
        else:
            return False
        self.release()
        if now - self.title_click > 3:
            self.title_click = now
            self.io.click(*sorter.Geo(frame).raw(*spot_))
            self.log(what)
        return True

    def _slow(self, now):
        """A few times a second: full-screen checks (where we are, waypoint list, the portal)."""
        frame = self.io.grab()
        self.frame = frame
        if frame is None:
            self.portal = None
            return False
        h, w = frame.shape[:2]
        o = self.opts
        top = sorter.Geo(frame).top
        s = (h - top) / 600.0                        # the game picture only (a window menu bar is not game)
        if self.state.get("run_reset", 0) > (self.over_since or 0) and self.over_since is not None:
            self._new_run()
        top = sorter.Geo(frame).top                  # a window menu bar inside the client area is not game
        self.top = top
        crop = frame[top: top + int(0.13 * (h - top)), int(0.60 * w):]
        mask = spot.area_mask(crop, (h - top) / 600.0)
        self.run_text = mask.sum() > 2000             # "N막 - ..." and the kill count of a run
        if self.run_text:
            self.state["run_text_at"] = now
        if self.run_text and spot.score(mask, "공포영역") >= 0.55:
            self.terror_last = now
            self.state["terror_at"] = now
            if not self.terror_ever:
                self.terror_ever = True
                self.state["terror_ever"] = True
        if not self.run_text and now - self.title_at > 1.5 and self._title_screen(frame, now):
            return True
        if o.get("town_start") and self._wp_menu(frame, s, now):
            if self.over_since is not None:
                self._new_run()
            return True
        if self._panel_open(frame):
            self.release()                            # the smith's window etc.: the engine works there
            self.panel_at = now
            norm = sorter.Geo(frame).normalize(frame)
            if sorter.seen(norm, "inv") >= 0.7 and sorter.seen(norm, "panel") < 0.6:
                # the plain inventory / character window, not the smith's: nothing of ours needs it open
                if self.inv_open_since is None:
                    self.inv_open_since = now
                elif now - self.inv_open_since > 2.5:
                    self.io.key(Key.esc)
                    self.inv_open_since = None
                    self.log("열려 있던 인벤토리 창을 닫음")
            else:
                self.inv_open_since = None
            if self.panel_since is None:
                self.panel_since = now
            elif now - self.panel_since > 12 and now - self.panel_esc > 3:
                self.panel_esc = now                  # nobody is working in it: close it
                self.io.key(Key.esc)
                self.log("12초 넘게 열려 있던 창을 Esc로 닫음")
            return True
        self.inv_open_since = None
        self.panel_since = None
        self.portal = find_portal_screen(frame, s)
        if self.over_since is None:
            try:
                if sorter.portal_opened(sorter.Geo(frame).normalize(frame)) >= 0.55:
                    if now - self.gate_at > 30:
                        self.log("'차원문이 열렸습니다' 확인")
                    self.gate_at = now
                    self.over_since = now             # the hunt is over even before the minimap is read
                    self.last_gain = now
                    self.state["over"] = True
                    self.state["over_at"] = now
                    self.log("보스 처치 확인: 사냥 종료, 수집 시작")
            except Exception:
                pass
        return False

    def _left_zone(self, now, portal, mark):
        """After the portal (or whenever the terror zone is clearly gone) the run is over for us."""
        if self.over_since is None:
            return False
        if self.portal_click and now - self.portal_click < 60 and portal is None and mark is None:
            if self.gone_since is None:
                self.gone_since = now
            if now - self.gone_since >= 2.0:
                self.log("포탈로 이동함: 새 판 준비")
                return True
        else:
            self.gone_since = None
        if self.terror_ever and self.terror_last >= self.over_since - 120 and now - self.terror_last > 6 \
                and not self.run_text:
            self.log("공포 영역을 벗어남: 새 판 준비")
            return True
        return False

    def _stalled(self, now, why):
        """While collecting, say why nothing is happening (at most every 15 s)."""
        if self.over_since is None:
            return
        if why != self.why:
            self.why, self.why_since = why, now
            return
        if now - self.why_since >= 6 and now - self.why_note >= 15:
            self.why_note = now
            self.log("줍기 대기 중: " + why)

    def _watchdog(self, now):
        """Whatever went wrong: some time after the boss, close any window and take the portal; in town,
        close menus that stay open. -> True when it pressed something this step."""
        limit = float(self.opts.get("over_limit", 60) or 60)
        base = self.over_since
        if self.route_end is not None:
            base, limit = self.route_end, max(limit, 100.0)
        if base is not None and now - base > limit + 15:
            if self.busy and now - self.wd_at > 3 and now - self.state.get("want_smith", 0) > 8:
                self.wd_at = now
                self.release()
                self.io.key(Key.esc)
                self.busy = False
                self.frame = None
                if now - self.wd_note > 30:
                    self.wd_note = now
                    self.log("보스 처치 후 %d초: 열린 창을 닫고 포탈로 (막힘 대비)" % int(now - self.over_since))
                return True
            self.busy = False                         # never wait on anything any more: portal only
        elif self.over_since is None and self.town_since is not None and now - self.town_since > 45 \
                and now - self.wd_at > 15:
            self.wd_at = now
            self.town_since = now
            self.release()
            self.io.key(Key.esc)
            self.frame = None
            self.log("마을에서 45초 동안 출발하지 못함: 열린 창을 닫고 다시 시도 (막힘 대비)")
            return True
        return False

    def _route(self, now):
        """A recorded arrow-key route: 5 seconds after the boss died, play it back exactly as recorded,
        then go straight on to the tidy-up and the portal. -> True while waiting or playing."""
        events = self.opts.get("route") or []
        base = self.over_since
        if base is None and self.gate_at and now - self.gate_at < 30:
            base = self.gate_at                       # the boss notice was just seen: nothing moves before
        if not events or base is None or self.route_played or self.forced:
            return False
        if now - base < 3.0:
            self.release()
            if not self.route_note:
                self.route_note = True
                self.log("녹음된 방향키가 있음: 보스 처치 3초 뒤에 재생 (그 녹음으로만 이동)")
            return True
        self.release()
        events, cut = squeeze_route(events)
        if not events:
            self.route_played = True
            self.log("녹음된 방향키에 누른 키가 없음: 녹음 없이 진행")
            return False
        self.log("녹음된 방향키 재생 시작 (%d번 누름, %.1f초%s)" % (
            sum(1 for e in events if e[1] == "down"), events[-1][0],
            ", 키를 안 누른 빈 시간 %.0f초는 건너뜀" % cut if cut >= 1 else ""))
        stop = getattr(self, "stop_flag", None)
        # exact timing: the screen is watched on another thread so a key is never let go late
        # (a late key-up = running too far, which is what made the route drift)
        _timer_res(True)
        watch = threading.Event()               # set by the watcher: the smith's window is open
        done = threading.Event()

        def watcher():
            try:
                from .gameio import GameIO
                io2 = GameIO(self.io.title)
            except Exception:
                return
            while not done.is_set():
                try:
                    f = io2.grab()
                    if f is not None:
                        norm = sorter.Geo(f).normalize(f)
                        if sorter.seen(norm, "panel") >= 0.6 or sorter.seen(norm, "shop") >= 0.6:
                            watch.set()
                except Exception:
                    pass
                done.wait(0.25)

        threading.Thread(target=watcher, daemon=True).start()
        held = set()
        rep_at = [0.0]
        start = time.perf_counter()
        try:
            for t, kind, name in events:
                while True:
                    if watch.is_set():
                        # the smith's window popped up on the way: let go, close it, carry on from here
                        t0 = time.perf_counter()
                        for k in list(held):
                            self.io.key_up(k)
                        try:
                            halted = []
                            sorter.restore_worn(self.io, self.log,
                                                lambda: (halted.append(1), self.state.__setitem__("halt_now", True)))
                            if halted:
                                return True
                        except Exception:
                            pass
                        self.io.key(Key.esc)
                        time.sleep(0.3)
                        watch.clear()
                        time.sleep(0.3)
                        if watch.is_set():
                            watch.clear()
                            self.io.key(Key.esc)
                            time.sleep(0.3)
                        for k in list(held):
                            self.io.key_down(k)
                        start += time.perf_counter() - t0
                        self.log("이동 중 대장장이 창이 열려 닫고 계속 이동")
                    left = start + float(t) - time.perf_counter()
                    if left <= 0 or (stop is not None and stop.is_set()):
                        break
                    if held and time.perf_counter() - rep_at[0] > 0.12:
                        rep_at[0] = time.perf_counter()
                        self.io.key_hold()          # (background only: keeps held arrows held)
                    if left > 0.004:
                        time.sleep(min(left - 0.002, 0.01))
                if stop is not None and stop.is_set():
                    break
                k = ARROWS.get(name)
                if k is None:
                    continue
                if kind == "down":
                    if k not in held:
                        self.io.key_down(k)
                        held.add(k)
                else:
                    self.io.key_up(k)
                    held.discard(k)
        finally:
            done.set()
            for k in held:
                self.io.key_up(k)
            for k in ARROWS.values():           # nothing may stay pressed after the route
                self.io.key_up(k)
            _timer_res(False)
        self.route_played = True
        self.route_end = time.time()
        self.state["route_done"] = True                   # a smith window from here on is the tidy-up
        self.state["route_len"] = self.route_end - base
        self.state["want_smith"] = self.route_end
        self.sweep_done = True                        # the route replaces the run-around
        self.frame = None
        self.log("방향키 재생 끝: 정리하고 다음 판으로")
        return True

    def _time_up(self, now):
        """The hunt ended more than the allowed time ago: straight to the portal, whatever is left."""
        limit = float(self.opts.get("over_limit", 60) or 0)
        base = self.over_since
        if self.route_end is not None:
            base, limit = self.route_end, max(limit, 100.0)     # after the route: smith + portal get their own time
        if base is None or limit <= 0 or now - base <= limit:
            return False
        if not self.forced:
            self.forced = True
            self.target, self.arrived, self.loot_goal = None, None, None
            self.release()
            self.log("%s %d초 지남: 남은 줍기·정리를 그만두고 다음 판으로"
                     % ("방향키 재생 끝난 뒤" if self.route_end is not None else "보스 처치 후", int(limit)))
        return True

    def step(self):
        o = self.opts
        if not self.kinds and not o.get("stage_end") and not o.get("town_start"):
            return
        now = time.time()
        if now - self.slow_at >= 0.35 or self.frame is None:
            self.slow_at = now
            self.busy = self._slow(now)
        frame = self.frame
        if frame is None:
            self._stalled(now, "게임 화면을 읽지 못함 (게임 창이 최소화됐거나 그림을 그리지 않음)")
            self.release()
            return
        if self._watchdog(now):
            return
        if self.busy:
            self._stalled(now, "게임 위에 창이 열려 있음 (대장장이·인벤토리·웨이포인트 창이 아닌데 이 문구가 계속 나오면 알려 주세요)")
            return
        if self.over_since is not None and now - self.over_since > 1.0:
            from . import gearscan
            if gearscan.PENDING.get("dir"):
                self.release()
                try:
                    gearscan.scan_skills(self.io, self.log)
                except Exception as e:
                    self.log("스킬 스캔 오류: %s" % e)
                return
        if self.rerolling:
            # 패스트 모드: not act 3 / 4 -> leave at once (Esc -> 저장 후 나가기) and take the waypoint again
            self.release()
            if self.exit_tries < 6:
                self._exit_town(now, "패스트 모드: %s이라 바로 나가서 다시 들어감 (3막·4막만)" % self.act_name)
                return
            self.rerolling = False
        if (self.opts.get("fast_mode") and self.over_since is None and not self.act_done and self.run_text
                and now - self.terror_last < 2 and now - self.act_at >= 0.3):
            self.act_at = now
            self._act_check(frame, now)
            if self.rerolling:
                return
        if self.opts.get("fast_mode") and self.over_since is not None and not self.forced:
            # 패스트 모드: the essences come into the bags by themselves when the boss dies. Pick up
            # nothing, no route, no smith, no inventory: Esc -> town -> next run.
            self.release()
            if now - self.over_since < 2.0:
                return
            if not self.fast_counted and self.opts.get("fast_count"):
                self.fast_counted = True
                self._fast_count()
                self.frame = None
                return
            if self.exit_tries < 6:
                self._exit_town(now, "패스트 모드: 보스 처치 -> Esc로 마을로 가서 다음 판")
                return
        if self._route(now):
            return
        h, w = frame.shape[:2]
        s = (h - self.top) / 600.0
        x0, y0, x1, y1 = region(w, h)
        quick = getattr(self.io, "grab_region", None)
        if quick is not None:
            img, cw, ch = quick(x0, y0, x1, y1)        # fresh minimap every step: this is what steers
            if img is None or (cw, ch) != (w, h):
                self.release()
                self.frame = None
                return
        else:
            img = frame[y0:y1, x0:x1]
        me = find_player(img, s)
        portal = self.portal
        mark = self._track_mark(img, s, me, now) if me is not None else None
        if self._left_zone(now, portal, mark):
            self._new_run()
        if me is None:
            self.release()
            self.target = None
            self._stalled(now, "미니맵에서 내 위치(흰 십자)를 찾지 못함 (미니맵이 꺼져 있으면 Tab으로 켜 주세요)")
            if self._time_up(now):
                self._finish(frame, s, now, (0.0, 0.0), portal, None)
                return
            if self.route_played and not self.forced:
                self._finish(frame, s, now, (0.0, 0.0), portal, None)
                return
            if self.over_since is not None and not self.forced:
                # essences first, minimap or not: their names are on the screen itself
                if self.pos is None:
                    self.pos, self.odo_prev, self.still_since = (0.0, 0.0), None, now
                    self.ratio = 11.3 * s
                self._odo(frame, s, now)
                self.cur_rel = None
                if self.ess_t0 is None:
                    self.ess_t0 = now
                    self.log("줍기 시작: 화면에서 정수부터 찾음")
                if self.loot_goal is not None or now - self.loot_look >= 0.4:
                    if self._loot(frame, s, now):
                        self.idle_since = None
                        self.stage_start = None
                        if not self.ess_done:
                            self.ess_t0 = now
                        return
                if not self.ess_done and now - self.ess_t0 < 1.3:
                    return
                self.ess_done = True
                if not self.ss_done:
                    self._screen_sweep(frame, s, now)
                    return
                if self.tab_tried == 0:
                    # the item marks are on the minimap: turn it on (Tab) to look for more
                    self.tab_tried = 1
                    self.release()
                    self.io.key(Key.tab)
                    self.frame = None
                    self.log("미니맵이 꺼져 있음: Tab으로 켜서 아이템 표시 확인")
                    return
                if self.tab_tried == 1:
                    self.tab_tried = 2
                    self.log("미니맵을 읽지 못함 (이 맵은 미니맵이 없음): 표시 줍기 없이 정리로")
            if self.over_since is not None and now - self.over_since > 6 and o.get("stage_end"):
                # no minimap: skip the markers, but still tidy up and take the portal (both found on screen)
                if self.idle_since is None:
                    self.idle_since = now
                self._finish(frame, s, now, (0.0, 0.0), portal, None)
                return
            if self.over_since is None and portal is not None and (not self.terror_ever or now - self.terror_last < 5):
                # no minimap to read and the boss notice was missed: a portal standing on screen ends the hunt
                if now - self.portal_last > 1.0:
                    self.portal_first = now
                self.portal_last = now
                if now - self.portal_first >= 5.0:
                    self.over_since = now
                    self.last_gain = now
                    self.state["over"] = True
                    self.state["over_at"] = now
                    self.log("미니맵은 못 읽었지만 화면에 포탈이 계속 보여 사냥 종료로 판단")
                    return
            if self.over_since is None:
                hunting = self.terror_ever and now - self.terror_last < 5
                if hunting:
                    if self.nomap_since is None:
                        self.nomap_since = now
                    if self.hunt_tab == 0 and now - self.nomap_since > 20:
                        # the minimap is most likely switched off (Tab): switch it on once this run
                        self.hunt_tab, self.hunt_tab_at = 1, now
                        self.release()
                        self.io.key(Key.tab)
                        self.log("사냥 중 미니맵이 안 보여 Tab으로 켬 (보스 처치 확인과 정리에 필요)")
                    elif self.hunt_tab == 1 and now - self.hunt_tab_at > 4:
                        self.hunt_tab = 2                 # still nothing: it was not that, put it back
                        self.io.key(Key.tab)
                        self.log("Tab을 눌러도 미니맵의 내 위치가 안 보임: 원래대로 되돌림 "
                                 "(보스 처치는 '차원문이 열렸습니다' 문구와 화면의 포탈로 확인)")
                self._town(frame, s, now, None)
            return
        self.nomap_since = None
        if self.hunt_tab == 1:
            self.hunt_tab = 3
            self.log("미니맵 켜짐: 내 위치 확인")
        if self.over_since is None:
            if now - self.gate_at < 30:
                self.over_since = now
                self.last_gain = now
                self.state["over"] = True
                self.state["over_at"] = now
                self.log("보스 처치 확인: 사냥 종료, 수집 시작")
            elif portal is not None and mark is not None and (not self.terror_ever or now - self.terror_last < 5):
                # the notice was missed (e.g. the helper started later): both portal signs for a while
                if now - self.portal_last > 1.0:
                    self.portal_first = now
                self.portal_last = now
                if now - self.portal_first >= 4.0:
                    self.over_since = now
                    self.last_gain = now
                    self.state["over"] = True
                    self.state["over_at"] = now
                    self.log("차원문 문구는 못 봤지만 포탈이 계속 보여 사냥 종료로 판단")
        over = self.over_since is not None
        if o.get("collect_after", True) and not over:
            self.release()
            self.target = None
            self._town(frame, s, now, me)
            return
        if self.pos is None:
            self.pos, self.odo_prev, self.still_since = (0.0, 0.0), None, now
            self.ratio = 11.3 * s                         # screen pixels per minimap pixel (refined later)
        self._odo(frame, s, now)
        if self._time_up(now):
            self._finish(frame, s, now, me, portal, mark)
            return
        # inventory filled up while collecting: visit the smith before picking up more
        if over and o.get("auto_clean") and self.state.get("inv_full", 0) > max(self.state.get("cleaned", 0),
                                                                                self.over_since):
            self.target = None
            self._go_smith(frame, s, now, me, portal, mark, "인벤토리가 가득 참")
            return
        if self.stage_start is None:
            self.npc_since = None
        if self.state.get("cleaned", 0) > self.clean_seen:
            self.clean_seen = self.state.get("cleaned", 0)      # room in the bags again: everything is worth
            self.ignore, self.tries, self.loot_tries = [], {}, {}  # another go
            self.inv_check = None
        self._collect(frame, img, s, now, me, portal, mark)

    # ---- where the character is: odometry from the scrolling game view -----------------------
    # The minimap scrolls with the character (the player marker never moves), so positions are kept
    # in "world" screen pixels: (0, 0) is where the character stood when collecting began, and the
    # character's position is followed by measuring how far the game picture slides.
    def _odo(self, frame, s, now):
        h, w = frame.shape[:2]
        k = max(2, int(round(w / 640.0)))
        # the game world only: no skill icons (top), no bottom bar, no minimap (right), no run info
        x0, y0, x1, y1 = int(0.12 * w), int(0.25 * h), int(0.66 * w), int(0.78 * h)
        quick = getattr(self.io, "grab_region", None)
        if quick is not None:
            img, cw, ch = quick(x0, y0, x1, y1)
            if img is None or (cw, ch) != (w, h):
                return
        else:
            img = frame[y0:y1, x0:x1]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // k, g.shape[0] // k), interpolation=cv2.INTER_AREA).astype(np.float32)
        cy, cx = int((0.5 * h - y0) / k), int((0.5 * w - x0) / k)
        r = int(45 * s / k)
        g[max(0, cy - 2 * r):cy + r, max(0, cx - r):cx + r] = float(g.mean())   # the character himself
        if self.odo_prev is not None and self.odo_prev.shape == g.shape:
            if self.odo_win is None or self.odo_win.shape != g.shape:
                self.odo_win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            (sx, sy), resp = cv2.phaseCorrelate(self.odo_prev, g, self.odo_win)
            if resp > 0.04 and abs(sx) < g.shape[1] / 3.0 and abs(sy) < g.shape[0] / 3.0:
                mx, my = -sx * k, -sy * k                 # the scene slides the other way
                if abs(mx) + abs(my) > 1.5 * k:
                    self.pos = (self.pos[0] + mx, self.pos[1] + my)
                    self.still_since = now
        self.odo_prev = g

    def _bounds_ok(self, wx, wy, frame):
        """Inside what the screen showed when collecting began."""
        h, w = frame.shape[:2]
        return abs(wx) <= 0.48 * w and abs(wy) <= 0.44 * h

    # ---- collecting ------------------------------------------------------------------------------
    # The area to collect in (the user's rule): the character is 2 m tall; move only within 10 m up, down,
    # left and right of where he stood when collecting began. A character is about 60 layout units tall,
    # so 10 m = 300 units = 300*s screen pixels = 300*s / (11.3*s) = about 26.5 minimap pixels.
    # Measured on the minimap from the exit portal's red mark, which does not drift.
    @property
    def HALF(self):
        """Half size of the area in minimap pixels: 1 m = 30 layout units = 30 / 11.3 minimap pixels."""
        m = float(self.opts.get("radius_m", 10) or 10)
        return max(1.0, m) * 30.0 / 11.3

    def _in_box(self, rx, ry, margin=0.0):
        if self.home is None:
            return True
        return (abs(rx - self.home[0]) <= self.HALF + margin and abs(ry - self.home[1]) <= self.HALF + margin)

    def _track_mark(self, img, s, me, now):
        """The portal mark, followed from frame to frame: it never jumps (the portal does not move, the
        character only runs a few pixels' worth between two looks), so a red mark far from where it was
        is something else."""
        marks = portal_marks(img, s, me)
        if not marks:
            return None
        if self.mark_off is not None and now - self.mark_at < 2.0:
            px, py = me[0] + self.mark_off[0], me[1] + self.mark_off[1]
            near = min(marks, key=lambda m: (m[0] - px) ** 2 + (m[1] - py) ** 2)
            allow = 4.0 * s + 15.0 * (now - self.mark_at)
            if (near[0] - px) ** 2 + (near[1] - py) ** 2 <= allow ** 2:
                return near[0], near[1]
            return None
        # first sighting: the one nearest to the character (the portal opens where the boss fell)
        near = min(marks, key=lambda m: (m[0] - me[0]) ** 2 + (m[1] - me[1]) ** 2)
        return near[0], near[1]

    def _portal_vec(self, now):
        """Minimap vector from the character to the portal: measured when its mark is in view, else
        reckoned from where it was last seen."""
        if self.mark_off is not None and now - self.mark_at < 0.5:
            return self.mark_off
        if self.portal_w is not None and self.pos is not None:
            return ((self.portal_w[0] - self.pos[0]) / self.ratio, (self.portal_w[1] - self.pos[1]) / self.ratio)
        return None

    def _collect(self, frame, img, s, now, me, portal, mark):
        o = self.opts
        h, w = frame.shape[:2]
        px, py = self.pos
        if mark is not None:
            self.mark_off, self.mark_at = (mark[0] - me[0], mark[1] - me[1]), now
        anchored = self.mark_off is not None and now - self.mark_at < 0.6
        if anchored and self.home is None:
            self.home = (-self.mark_off[0], -self.mark_off[1])      # where I stand, seen from the portal
        rel = (-self.mark_off[0], -self.mark_off[1]) if anchored else None
        self.cur_rel = rel
        if anchored:
            if self.prev_rel is None or np.hypot(rel[0] - self.prev_rel[0], rel[1] - self.prev_rel[1]) > 0.8:
                self.still_since, self.prev_rel = now, rel          # moving (measured on the minimap)
        elif self.home is not None and not self.forced and self.ss_done:
            # the portal mark is not in view: never keep running blind. Stop right here and look again.
            if self.dragging or now - self.click_at < 1.0:
                self.release()
                self._world_click(w / 2.0, h / 2.0)            # a click on his own feet: he stops there
                self.click_at = 0.0
            else:
                self.release()
            self.target, self.loot_goal = None, None
            if now - self.mark_at > 3.0 and not self.sweep_done:
                self.sweep_done = True
                self.log("미니맵에서 포탈 표시를 3초 넘게 못 찾아 돌기를 멈춤")
            if now - self.mark_at < 3.0:
                return
        if self.route_played and not self.forced:
            # the recorded route is the only movement: no essence chasing, sweep or markers after it
            if self.idle_since is None:
                self.idle_since = now
            self._finish(frame, s, now, me, portal, mark)
            return
        # FIRST: essences (then keys / charms) whose name is on screen, before markers, the run-around and
        # even the radius rule. At the very start look for a moment before doing anything else.
        if not self.forced:
            if self.ess_t0 is None:
                self.ess_t0 = now
                self.release()
                self.log("줍기 시작: 화면에서 정수부터 찾음")
            if self.loot_goal is not None or now - self.loot_look >= 0.4:
                if self._loot(frame, s, now):
                    self.idle_since = None
                    self.stage_start = None
                    self.sweep_at = now
                    if not self.ess_done:
                        self.ess_t0 = now             # keep looking until none has shown for a while
                    return
            # an essence seen earlier that went off the screen: go back toward it until its name shows
            self.ess_mem = [m for m in self.ess_mem if now - m[3] < 25]
            if self.ess_mem and self.cur_rel is not None:
                rel_ = self.cur_rel
                m = min(self.ess_mem, key=lambda q: (q[0] - rel_[0]) ** 2 + (q[1] - rel_[1]) ** 2)
                if now - self.ess_note > 8:
                    self.ess_note = now
                    self.log("앞서 본 정수 쪽으로 이동")
                self._head(frame, s, now, m[0] - rel_[0], m[1] - rel_[1])
                self.ess_t0 = now if not self.ess_done else self.ess_t0
                return
            if not self.ess_done:
                if now - self.ess_t0 < 1.3:       # a few looks (labels can be hidden by effects for a moment)
                    self.release()
                    return
                self.ess_done = True
                if not self.ess_seen:
                    self.log("화면에 정수 없음: 주변 줍기로")
        if self.route_played and not self.forced:
            # the recorded route did the collecting: tidy up and leave (after the essences above)
            if self.idle_since is None:
                self.idle_since = now
            self._finish(frame, s, now, me, portal, mark)
            return
        if not self.ss_done and not self.forced:
            # the user's rule: run back and forth over everything the screen showed when the boss died
            self._screen_sweep(frame, s, now)
            self.idle_since = None
            self.stage_start = None
            return
        if self.tab_tried == 1:
            self.tab_tried = 2
            self.log("미니맵 켜짐: 아이템 표시 확인")

        # markers in portal coordinates (minimap pixels from the portal mark, times the scale so the
        # tolerances below read in screen pixels): exact and steady, nothing reckoned
        def world(t):
            return (t[1] - me[0] + rel[0]) * self.ratio, (t[2] - me[1] + rel[1]) * self.ratio

        tol = 3.0 * self.ratio
        self.ignore = [i for i in self.ignore if i[2] > now]
        seen_now = []
        if anchored:
            for t in find_targets(img, s, self.kinds):
                wx, wy = world(t)
                if self._in_box(wx / self.ratio, wy / self.ratio):
                    seen_now.append((t[0], wx, wy, t[1], t[2]))
        steady = [t for t in seen_now
                  if all(any(o_[0] == t[0] and abs(o_[1] - t[1]) < tol and abs(o_[2] - t[2]) < tol for o_ in old)
                         for old in self.history)]
        self.history = (self.history + [seen_now])[-2:]
        cands = [t for t in steady
                 if not any(abs(t[1] - i[0]) < tol and abs(t[2] - i[1]) < tol for i in self.ignore)]

        goal = self.target                                # (kind, wx, wy) in portal coordinates
        hit = None
        if goal is not None and not anchored:
            self.target = goal = None                     # no measuring without the mark: let it go
            self.release()
        if goal is not None:
            hit = min((c for c in seen_now if c[0] == goal[0] and abs(c[1] - goal[1]) < tol
                       and abs(c[2] - goal[2]) < tol),
                      key=lambda c: (c[1] - goal[1]) ** 2 + (c[2] - goal[2]) ** 2, default=None)
            if hit is not None:
                self.target = goal = (goal[0], hit[1], hit[2])
                self.goal_mm = (hit[3] - me[0], hit[4] - me[1])          # minimap offset, exact
                self.target_seen = now
            else:
                # hidden this frame (under my own cross, usually): its place is known exactly
                self.goal_mm = (goal[1] / self.ratio - rel[0], goal[2] / self.ratio - rel[1])
                mdist = float(np.hypot(*self.goal_mm))
                gx, gy = me[0] + self.goal_mm[0], me[1] + self.goal_mm[1]
                if marker_ink(img, goal[0], gx, gy, s) >= (0.3 if mdist < 10 * s else 0.55):
                    self.target_seen = now
                elif now - self.target_seen >= 0.3 and mdist <= max(3.0, 1.5 * s):
                    self.count += 1
                    self.last_gain = now
                    self.inv_check = None
                    self.log("%s 획득" % KIND_KO[goal[0]])
                    self.target = goal = None
                    self.arrived = None
                elif now - self.target_seen >= 1.0:
                    self.lost_goal = goal

        # then run around the portal (or, with no portal mark, where the hunt ended): items are picked
        # up by running over them
        if goal is None and not self.sweep_done and not anchored and now - self.over_since > 4:
            self.sweep_done = True                        # no portal mark to measure by: no running about
            self.log("미니맵에 포탈 표시가 없어 주변 돌기는 건너뜀")
        if goal is None and not self.sweep_done and anchored:
            if self._sweep_portal(frame, s, now, anchored):
                self.idle_since = None
                self.stage_start = None
                return

        if goal is None:
            if not cands:
                if self.idle_since is None:
                    self.idle_since = now
                if o.get("stage_end") and now - self.idle_since >= 0.3:
                    self._finish(frame, s, now, me, portal, mark)
                else:
                    self.release()
                return
            best = min(cands, key=lambda c: (c[1] - px) ** 2 + (c[2] - py) ** 2)
            goal = self.target = (best[0], best[1], best[2])
            self.goal_mm = (best[3] - me[0], best[4] - me[1])
            self.target_seen, self.since, self.arrived = now, now, None
            self.ring, self.click_at, self.click_pos, self.block_fail = 0, 0.0, None, 0
            self.log("%s 쪽으로 이동" % KIND_KO[goal[0]])
        self.idle_since = None
        self.stage_start = None
        key = (int(round(goal[1] / tol)), int(round(goal[2] / tol)))

        def drop(why):
            """Leave this marker for now; after two failed visits leave it for the rest of the run."""
            self.tries[key] = self.tries.get(key, 0) + 1
            others = [c for c in cands if abs(c[1] - goal[1]) > tol or abs(c[2] - goal[2]) > tol]
            last = self.tries[key] >= 2 or not others      # coming back only pays off after a detour
            self.ignore.append((goal[1], goal[2], now + (900 if last else 8)))
            self.log("%s: %s (%s)" % (KIND_KO[goal[0]], why, "이번 판에서는 포기" if last else "다른 것부터 줍고 다시 옴"))
            self.target, self.arrived = None, None
            self.release()

        if self.lost_goal is not None:
            self.lost_goal = None
            drop("미니맵 표시가 멀리서 사라짐")
            return
        if now - self.since > 20:
            drop("20초 안에 줍지 못함")
            return
        if self.mouse:
            self._mouse_step(goal, frame, s, now, drop)
        else:
            self._key_step(goal, s, now, drop)

    # ---- sweep: one lap around the spot where the hunt ended ---------------------------------
    def _head(self, frame, s, now, dx, dy):
        """Run (button held) in the direction of a minimap vector."""
        h, w = frame.shape[:2]
        d = float(np.hypot(dx, dy))
        if d < 1e-6:
            return
        if not self.mouse:
            self._hold(direction(dx, dy, self.held))
            return
        lead = min(max(d * self.ratio, 0.10 * min(w, h)), 0.22 * min(w, h))
        tx, ty = w / 2.0 + dx / d * lead, h / 2.0 + dy / d * lead
        tx, ty, _ = self._safe(tx, ty, w, h, s)
        if not self.dragging:
            f = self.io.grab()
            if f is None or self._panel_open(f):
                return
            tx, ty = self._clear_spot(tx, ty, frame, s, now)
            if self.io.drag_start(int(tx), int(ty)):
                self.dragging = True
                self.still_since = now
        else:
            self.io.move(int(tx), int(ty))

    def _sweep_portal(self, frame, s, now, anchored=True):
        """-> True while sweeping: a square spiral around the starting spot, inside the area, measured on the
        minimap itself (it cannot drift)."""
        if not anchored:
            self.release()                                # mark out of sight for a moment: wait for it
            if now - self.mark_at > 4:
                self.sweep_done = True
                self.log("포탈 표시가 안 보여 돌기를 마침")
                return False
            return True
        rel = (-self.mark_off[0], -self.mark_off[1])
        if self.psweep is None:
            hx, hy = self.home if self.home is not None else rel
            H = self.HALF
            pts, k, sx, sy = [], 1, 8.0, 6.0              # ring spacing in minimap pixels
            while True:
                ax, ay = min(k * sx, H), min(k * sy, H)
                pts += [(hx + ax, hy - ay), (hx - ax, hy - ay), (hx - ax, hy + ay), (hx + ax, hy + ay),
                        (hx + ax, hy - min(ay + sy, H))]
                if ax >= H and ay >= H:
                    break
                k += 1
            self.psweep, self.sweep_i, self.sweep_t0, self.sweep_at = pts, 0, now, now
            self.log("캐릭터 둘레 상하좌우 %gm 안을 안쪽부터 돌며 줍기 시작" % float(self.opts.get("radius_m", 10) or 10))
        if self.sweep_i >= len(self.psweep) or now - self.sweep_t0 > 45:
            self.sweep_done = True
            self.release()
            self.log("둘레 돌기 끝 (%d/%d 구간)" % (min(self.sweep_i, len(self.psweep)), len(self.psweep)))
            return False
        tx, ty = self.psweep[self.sweep_i]
        dx, dy = tx - rel[0], ty - rel[1]
        stuck = now - self.still_since > 1.2 and now - self.sweep_at > 1.2
        if np.hypot(dx, dy) < 2.5 or now - self.sweep_at > 5.0 or stuck:
            self.sweep_i += 1
            self.sweep_at = now
            return True
        # never run across the portal itself: if the straight line passes close to it, go round it
        L2 = dx * dx + dy * dy
        t = max(0.0, min(1.0, (-rel[0] * dx - rel[1] * dy) / L2)) if L2 > 0 else 0.0
        cx_, cy_ = rel[0] + t * dx, rel[1] + t * dy             # nearest point of the path to the portal
        if np.hypot(cx_, cy_) < 6.0 and 0.0 < t < 1.0:
            n = float(np.hypot(cx_, cy_)) or 1.0
            side = (cx_ / n, cy_ / n) if np.hypot(cx_, cy_) > 0.5 else (-dy / np.sqrt(L2), dx / np.sqrt(L2))
            gx, gy = side[0] * 9.0, side[1] * 7.0
            if self.home is not None:                           # the way round stays inside the area
                H = self.HALF
                gx = min(max(gx, self.home[0] - H), self.home[0] + H)
                gy = min(max(gy, self.home[1] - H), self.home[1] + H)
            dx, dy = gx - rel[0], gy - rel[1]
        self._head(frame, s, now, dx, dy)
        return True

    def _sweep(self, frame, s, now):
        """-> True while sweeping. A square spiral outward from where the hunt ended, rings about an
        eighth of the screen height apart, never leaving what the screen showed at that moment. Runs with
        the left button held down: the game picks up what the character runs over. Nearest ground first,
        so if time runs short it is the far edge that is left out."""
        h, w = frame.shape[:2]
        if self.sweep_pts is None:
            bx, by = 0.42 * w, 0.38 * h                     # half-size of the area (inside the start screen)
            d = 0.10 * h                                    # ring spacing
            pts, k = [], 1
            while True:
                rx, ry = min(k * d * w / h * 0.9, bx), min(k * d, by)
                pts += [(rx, -ry), (-rx, -ry), (-rx, ry), (rx, ry), (rx, -ry - d if ry < by else -ry)]
                if rx >= bx and ry >= by:
                    break
                k += 1
            self.sweep_pts = [(min(max(x, -bx), bx), min(max(y, -by), by)) for x, y in pts]
            self.sweep_i, self.sweep_t0, self.sweep_at = 0, now, now
            self.log("주변 달리며 줍기 시작 (보스 처치 때 보이던 화면 안, 안쪽부터 바깥으로)")
        if self.sweep_i >= len(self.sweep_pts) or now - self.sweep_t0 > 35:
            self.sweep_done = True
            self.release()
            self.log("주변 달리기 끝 (%d/%d 구간)" % (min(self.sweep_i, len(self.sweep_pts)), len(self.sweep_pts)))
            return False
        tx_, ty_ = self.sweep_pts[self.sweep_i]
        dx, dy = tx_ - self.pos[0], ty_ - self.pos[1]
        d = float(np.hypot(dx, dy))
        stuck = now - self.still_since > 1.2 and now - self.sweep_at > 1.2
        if d < 0.05 * min(w, h) or now - self.sweep_at > 6.0 or stuck:
            self.sweep_i += 1                                  # corner reached (or blocked): next one
            self.sweep_at = now
            return True
        if not self.mouse:
            self._hold(direction(dx, dy, self.held))
            return True
        # point the cursor at the corner, at least a short stretch ahead (keeps a steady run), but never
        # past the edge of the area
        lead = min(max(d, 0.10 * min(w, h)), d + 0.04 * min(w, h))
        tx, ty = w / 2.0 + dx / d * lead, h / 2.0 + dy / d * lead
        tx, ty, _ = self._safe(tx, ty, w, h, s)
        if not self.dragging:
            f = self.io.grab()
            if f is None or self._panel_open(f):
                return True
            tx, ty = self._clear_spot(tx, ty, frame, s, now)     # the first press must land on bare ground
            if self.io.drag_start(int(tx), int(ty)):
                self.dragging = True
                self.still_since = now
        else:
            self.io.move(int(tx), int(ty))
        return True

    def _screen_sweep(self, frame, s, now):
        """Back and forth in rows over the area the screen showed when the boss died (the character in the
        middle). Position: from the portal on the screen (exact) and, where it is off the screen, from how
        far the picture slid. Items are picked up by running over them."""
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, self.top + (h - self.top) / 2.0
        if self.pos is None:
            self.pos, self.odo_prev, self.still_since = (0.0, 0.0), None, now
        portal = self.portal if now - self.slow_at < 0.4 else None
        if self.ss_pts is None:
            self.pos, self.odo_prev = (0.0, 0.0), None
            self.ss_home = (cx - portal[0], cy - portal[1]) if portal is not None else None
            bx, by = 0.40 * w, 0.34 * (h - self.top)      # half-size: what the screen showed, minus its edges
            d = 0.09 * (h - self.top)                     # row spacing: close rows, nothing in between missed
            rows, y = [], -by
            while y <= by + 1:
                rows.append(y)
                y += d
            pts = []
            for i, y in enumerate(rows):
                xs = (-bx, bx) if i % 2 == 0 else (bx, -bx)
                pts += [(xs[0], y), (xs[1], y)]
            self.ss_pts, self.ss_i, self.ss_t0, self.ss_at, self.ss_box = pts, 0, now, now, (bx, by)
            self.log("화면 안 촘촘히 돌며 줍기 시작 (보스 처치 때 보이던 화면, %d줄)" % len(rows))
        elif portal is not None and self.ss_home is None:
            self.ss_home = (cx - portal[0] - self.pos[0], cy - portal[1] - self.pos[1])   # first sight of it
        elif portal is not None and self.ss_home is not None:
            # the portal does not move: where it is on the screen says exactly where the character is
            p = (cx - portal[0] - self.ss_home[0], cy - portal[1] - self.ss_home[1])
            if np.hypot(p[0] - self.pos[0], p[1] - self.pos[1]) < 0.25 * h:
                self.pos = p
        if self.ss_i >= len(self.ss_pts) or now - self.ss_t0 > 50:
            self.ss_done = True
            self.sweep_done = True                        # nothing else runs around after this
            self.release()
            self.log("화면 안 돌기 끝 (%d/%d 구간, %d초)" % (min(self.ss_i, len(self.ss_pts)), len(self.ss_pts),
                                                     int(now - self.ss_t0)))
            return False
        tx_, ty_ = self.ss_pts[self.ss_i]
        dx, dy = tx_ - self.pos[0], ty_ - self.pos[1]
        d = float(np.hypot(dx, dy))
        stuck = now - self.still_since > 1.2 and now - self.ss_at > 1.2
        if d < 0.04 * min(w, h) or now - self.ss_at > 7.0 or stuck:
            self.ss_i += 1
            self.ss_at = now
            return True
        # never past the edge of the area: the cursor leads only a little beyond the corner
        lead = min(max(d, 0.10 * min(w, h)), d + 0.03 * min(w, h))
        tx, ty = cx + dx / d * lead, cy + dy / d * lead
        tx, ty, _ = self._safe(tx, ty, w, h, s)            # never on the portal, the bars or the icons
        if not self.mouse:
            self._hold(direction(dx, dy, self.held))
            return True
        if not self.dragging:
            f = self.io.grab()
            if f is None or self._panel_open(f):
                return True
            tx, ty = self._clear_spot(tx, ty, frame, s, now)
            if self.io.drag_start(int(tx), int(ty)):
                self.dragging = True
                self.still_since = now
        else:
            self.io.move(int(tx), int(ty))
        return True

    def _clear_spot(self, x, y, frame, s, now):
        """Move a ground click off chests and townsfolk (a click on them would open or talk)."""
        h, w = frame.shape[:2]
        for _ in range(3):
            hit = False
            for kind, lx, ly in self._labels(frame, s, now):
                if kind in ("chest", "npc") and abs(x - lx) < 45 * s and ly - 10 * s < y < ly + 80 * s:
                    hit = True
                    break
            if not hit:
                break
            x, y = w / 2.0 + (x - w / 2.0) * 0.75, h / 2.0 + (y - h / 2.0) * 0.75
        return x, y

    # ---- floor loot by its label -------------------------------------------------------------
    def _loot(self, frame, s, now):
        """-> True while busy with a labelled floor item (essence, key, charm)."""
        h, w = frame.shape[:2]
        if now - self.loot_look >= 0.4:
            self.loot_look = now
            f = self.io.grab()                     # a fresh picture: labels move as the character does
            if f is None or f.shape != frame.shape:
                f = frame
            found = spot.search(f, s, LOOT_SEARCH, need=0.62, any_colour=True, alone=False, every=True)
            found.sort(key=lambda b: b[0] != "정수")    # essences first, so their entry wins below
            raw = [(x, y, n == "정수") for n, _, x, y in found]
            raw += [(l[1], l[2], False) for l in self._labels(f, s, now) if l[0] == "loot"]
            hits = []
            for p in raw:                            # one entry per label (several words can hit one name)
                if not any(abs(p[0] - q[0]) < 90 * s and abs(p[1] - q[1]) < 12 * s for q in hits):
                    hits.append(p)
            # (everything readable on screen counts: an essence is never left behind for being "too far")
            self.loot_hits = [p for p in hits if 0.03 * w < p[0] < 0.97 * w]
            if self.ess_t0 is not None and not self.ess_done and any(p[2] for p in self.loot_hits):
                self.ess_seen = True
            self._remember_essences(w, h, s, now)

        def wkey(pt):
            wx, wy = self._lw(pt, frame)
            return int(round(wx / (60 * s))), int(round(wy / (40 * s)))

        g = self.loot_goal
        if g is not None:
            near = [p for p in self.loot_hits
                    if abs(self._lw(p, frame)[0] - g[0]) < 45 * s and abs(self._lw(p, frame)[1] - g[1]) < 18 * s]
            if not near and now - g[3] > 0.8:
                if self.cur_rel is not None and self.ess_mem:
                    m = min(self.ess_mem, key=lambda q: (q[0] - self.cur_rel[0]) ** 2 + (q[1] - self.cur_rel[1]) ** 2)
                    if np.hypot(m[0] - self.cur_rel[0], m[1] - self.cur_rel[1]) < 8:
                        self.ess_mem.remove(m)
                self.log("바닥 아이템 획득")
                self.last_gain = now
                self.inv_check = None
                self.loot_goal = None
                return False
            if now > g[2]:
                self.loot_goal = None                    # try again from scratch (counts as a try)
                return False
            if near:
                p = min(near, key=lambda q: (q[0] - w / 2.0) ** 2 + (q[1] - h / 2.0) ** 2)
                if self.mouse and now - self.click_at >= 0.9:
                    self._world_click(p[0], p[1] - 5 * s, portal_ok=True)
                    self.click_at = now
                elif not self.mouse:
                    self._toward(p[0], p[1] + 10 * s, frame, s, stop_at=1.5)
            return True
        todo = [p for p in self.loot_hits if self.loot_tries.get(wkey(p), 0) < 3]
        if not todo:
            return False
        # essences before anything else, then the nearest
        p = min(todo, key=lambda q: (not q[2], (q[0] - w / 2.0) ** 2 + (q[1] - h / 2.0) ** 2))
        k = wkey(p)
        self.loot_tries[k] = self.loot_tries.get(k, 0) + 1
        wx, wy = self._lw(p, frame)
        self.loot_goal = (wx, wy, now + 5.0, now)
        self.release()
        if self.mouse:
            self._world_click(p[0], p[1] - 5 * s, portal_ok=True)   # clicking a name label walks over and picks it up
            self.click_at = now
        self.log("바닥의 %s 이름 클릭해서 줍기" % ("정수" if p[2] else "열쇠·부적"))
        return True

    def _remember_essences(self, w, h, s, now):
        """Essence names seen on screen are kept as spots measured from the portal mark, so one that
        scrolls off the screen (the character went for another first) is still gone back for."""
        rel = self.cur_rel
        if rel is None:
            return
        r = self.ratio
        seen = [(rel[0] + (x - w / 2.0) / r, rel[1] + (y + 10 * s - h / 2.0) / r) for x, y, e in self.loot_hits if e]
        for q in seen:
            for i, m in enumerate(self.ess_mem):
                if np.hypot(q[0] - m[0], q[1] - m[1]) < 4:
                    self.ess_mem[i] = [q[0], q[1], 0, m[3]]
                    break
            else:
                self.ess_mem.append([q[0], q[1], 0, now])
        # a remembered spot that is on screen with no essence name near it: count a miss
        for m in list(self.ess_mem):
            sx, sy = w / 2.0 + (m[0] - rel[0]) * r, h / 2.0 + (m[1] - rel[1]) * r
            if 0.05 * w < sx < 0.95 * w and 0.1 * h < sy < 0.8 * h \
                    and not any(np.hypot(q[0] - m[0], q[1] - m[1]) < 6 for q in seen):
                m[2] += 1
                if m[2] >= 3:
                    self.ess_mem.remove(m)

    def _lw(self, pt, frame):
        """World position of a point on screen."""
        h, w = frame.shape[:2]
        return self.pos[0] + pt[0] - w / 2.0, self.pos[1] + pt[1] - h / 2.0

    # ---- walking onto a marker -------------------------------------------------------------
    def _mouse_step(self, goal, frame, s, now, drop):
        h, w = frame.shape[:2]
        mx, my = self.goal_mm                                 # minimap offset of the marker
        dist = float(np.hypot(mx, my))
        moving = now - self.still_since < 0.25
        # after a click: did the character move? (and how far per minimap pixel: the scale)
        if self.click_pos is not None and not moving and now - self.click_at > 0.45 and self.cur_rel is not None:
            moved = float(np.hypot(self.cur_rel[0] - self.click_pos[0], self.cur_rel[1] - self.click_pos[1]))
            want = float(np.hypot(*self.click_mm))
            if moved > 0.6:                                    # minimap pixels, measured from the portal mark
                self.mouse_ok, self.mouse_fail, self.block_fail = True, 0, 0
            elif want > 1.0:
                self.mouse_fail += 1
                self.block_fail += 1
                if not self.mouse_ok and self.mouse_fail >= 4:
                    self.mouse = False
                    self.log("마우스 클릭으로 캐릭터가 움직이지 않음: 화살표 키 방식으로 바꿈")
                    return
                if self.block_fail >= 3:
                    drop("길이 막힘")
                    return
            self.click_pos = None
        if dist <= max(1.1, 0.55 * s):
            # standing on it: a moment for the pick-up, then click the item itself, then a small ring
            if self.arrived is None:
                self.arrived, self.ring = now, 0
                return
            if now - self.arrived < 0.35 + 0.3 * self.ring:
                return
            ring = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
            if self.ring < len(ring):
                ox, oy = ring[self.ring]
                self.ring += 1
                r = 0.8 * self.ratio
                self._world_click(w / 2.0 + mx * self.ratio + ox * r, h / 2.0 + my * self.ratio + oy * r)
                self.click_at, self.click_pos = now, None
                return
            if self.state.get("inv_full", 0) > self.since:
                self.target = None                  # the bags are full: the smith comes first
                return
            drop("바로 위에서도 주워지지 않음")
            return
        self.arrived = None
        far = dist * self.ratio > 0.3 * min(w, h)
        if far:
            if now - self.click_at < 0.3:
                return
        elif self.click_pos is not None and (moving or now - self.click_at < 0.45) and now - self.click_at < 1.5:
            return
        self.clamped = self._walk_to(w / 2.0 + mx * self.ratio, h / 2.0 + my * self.ratio, frame, s, now)
        self.click_pos, self.click_mm = self.cur_rel, (mx, my)
        self.still_since = max(self.still_since, now - 0.2)

    def _key_step(self, goal, s, now, drop):
        mx, my = self.goal_mm
        dist = float(np.hypot(mx, my))
        if self.arrived is not None:
            ring = ("right", "down", "left", "left", "up", "up", "right", "right", "down")
            if now - self.arrived < 0.12:
                self.release()
                return
            if self.ring < len(ring):
                self._hold({ring[self.ring]})
                time.sleep(0.05)
                self.release()
                self.ring += 1
                self.arrived = now
                return
            self.arrived = None
            if self.state.get("inv_full", 0) > self.since:
                self.target = None
                return
            drop("바로 위에서도 주워지지 않음")
            return
        if self.held and now - self.still_since > 1.5:
            drop("길이 막힘")
            return
        if dist > 2.5 * s:
            if not self.held:
                self.still_since = now
            self._hold(direction(mx, my, self.held))
            return
        self.release()
        if dist <= max(0.3 * s, 0.75):
            self.arrived, self.ring = now, 0
            return
        k = ("right" if mx > 0 else "left") if abs(mx) >= abs(my) else ("down" if my > 0 else "up")
        self._hold({k})
        time.sleep(0.04)
        self.release()


class PickerThread(threading.Thread):
    def __init__(self, title, opts, emit, lock, state=None):
        super().__init__(daemon=True)
        self.title, self.opts, self.emit, self.lock = title, opts, emit, lock
        self.state = state
        self.stop_flag = threading.Event()

    def stop(self):
        self.stop_flag.set()

    def in_town(self):
        """Between runs (in town, the hunt not started): a safe moment to restart for an update."""
        p = getattr(self, "p", None)
        return p is not None and p.over_since is None and p.town_since is not None

    def run(self):
        from .gameio import GameIO
        io = GameIO(self.title)
        def log(m):
            if self.state is not None and not m.startswith(("줍기 대기 중", "(")):
                self.state["progress_at"] = time.time()   # waiting notices are not progress
            self.emit("log", m)
        p = Picker(io, self.opts, log, self.state)
        p.stop_flag = self.stop_flag
        self.p = p
        kinds = ", ".join(KIND_KO[k] for k in ("set", "unique", "jewel") if k in p.kinds)
        log("아이템 줍기 시작: %s%s" % (kinds or "대상 없음", ", 끝나면 포탈" if self.opts.get("stage_end") else ""))
        last_err, err_at = "", 0.0
        try:
            while not self.stop_flag.is_set():
                if self.lock.acquire(False):
                    try:
                        p.step()
                    except Exception as e:
                        msg = "%s: %s" % (type(e).__name__, e)
                        if msg != last_err or time.time() - err_at > 10:
                            last_err, err_at = msg, time.time()
                            log("아이템 줍기 오류: %s" % msg)
                        p.release()
                    finally:
                        self.lock.release()
                else:
                    p.release()
                self.stop_flag.wait(0.03)
        finally:
            p.release()
            io.close()
