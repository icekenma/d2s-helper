# -*- coding: utf-8 -*-
"""Inventory sorter for the D2 Survivors town panel.

The game draws its UI as an 800x600 layout scaled by window height and centred
horizontally, so every coordinate here is in that 800x600 space.
"""
import json
import os
import time

import cv2
import numpy as np

from . import scorer
from . import paths

INV_X, INV_Y, CELL, INV_COLS, INV_ROWS = 418, 313, 29, 10, 4
ESSENCE_HOLD = 90       # an essence stack stays in the bags until it holds this many (then: stash)
ESS_NEED = 0.6          # essence picture match
TAB_SHOP, TAB_STASH = (117, 17), (263, 17)              # old window: 파라 / 도박 / 보관함 / 비술 트리
TAB_SHOP2, TAB_STASH2 = (136, 17), (345, 17)            # new window (2026-11): 파라 / 도박 / 보관함, wider tabs
STASH_SUBTABS = {"개인": (130, 51), "공유 1": (185, 51), "공유 2": (240, 51), "공유 3": (295, 51)}   # old: 3 shared
STASH_SUBTABS2 = {"개인": (131, 52), "공유 1": (174, 52), "공유 2": (201, 52), "공유 3": (230, 52),       # new: 6 shared
                  "공유 4": (258, 52), "공유 5": (286, 52), "공유 6": (315, 52)}
STASH_ORDER = ["개인", "공유 1", "공유 2", "공유 3", "공유 4", "공유 5", "공유 6"]
PARK = (400, 500)   # empty stone area between the two panels' bottoms
# name -> (file, expected top-left x, y)
MARKS = {"subtabs2": ("subtabs2.png", 162, 44), "menu_back": ("menu_back.png", 300, 152), "menu_exit": ("menu_exit.png", 280, 360), "confirm": ("confirm.png", 240, 214), "panel": ("panel.png", 332, 9), "panel2": ("panel2.png", 193, 6), "stash": ("stash.png", 200, 447),
         "personal": ("stash_personal.png", 104, 40), "shop": ("shop.png", 100, 66),
         "inv": ("inv.png", 417, 444), "wp": ("wp.png", 354, 447), "wp_main": ("wp_main.png", 340, 62), "forge": ("forge.png", 253, 306), "forge_mode": ("forge_mode.png", 188, 70),
         "title_offline": ("title_offline.png", 330, 337), "title_start": ("title_start.png", 330, 296),
         "title_saves": ("title_saves.png", 330, 337)}
TITLE_OFFLINE, TITLE_START = (400, 348), (400, 307)      # the game's main menu: '오프라인 플레이', then '시작'
FORGE_OK = (214, 321)            # the check button under the anvil
FORGE_MAT_LINE = (150, 91, 340, 110)   # x0, y0, x1, y1 of the material count line
FORGE_VIEW = (150, 60, 400, 360)
COLORS = ["tan", "yellow", "green", "blue", "red"]
ACTION_KO = {"keep": "그대로 둠", "stash": "보관함으로 옮김", "sell": "판매",
             "sell_low": "점수 낮으면 판매, 아니면 보관"}
COLOR_KO = {"tan": "황갈색 칸 (유니크로 추정)", "yellow": "노란 칸 (레어로 추정)", "green": "초록 칸 (세트)",
            "blue": "파란 칸 (매직)", "red": "빨간 칸 (착용 불가)"}
_marks = {}


_snap_base = [None]                       # program folder, for snapshots of screens that could not be read


def load_marks(base):
    _snap_base[0] = base
    for k, (fn, x, y) in MARKS.items():
        fp = os.path.join(base, "assets", fn)
        if not os.path.exists(fp):
            continue
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        _marks[k] = (img, x, y)
    img = cv2.imdecode(np.fromfile(os.path.join(base, "assets", "portal_msg.png"), dtype=np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    _pmsg[0] = (img > 127).astype(np.float32)


class Geo:
    """Maps between raw client pixels and the 800x600 layout."""

    def __init__(self, frame):
        h, w = frame.shape[:2]
        gray = frame[: min(h, 80), :, :].mean(axis=(1, 2))
        top = 0
        while top < len(gray) - 1 and gray[top] > 200:   # a light menu bar inside the client area
            top += 1
        self.top = top
        self.s = (h - top) / 600.0
        self.x0 = (w - 800 * self.s) / 2.0

    def raw(self, nx, ny):
        return int(round(self.x0 + nx * self.s)), int(round(self.top + ny * self.s))

    def normalize(self, frame):
        m = np.float32([[1 / self.s, 0, -self.x0 / self.s], [0, 1 / self.s, -self.top / self.s]])
        return cv2.warpAffine(frame, m, (800, 600), flags=cv2.INTER_AREA)


def seen(norm, name, pad=8, text=False):
    """text=True compares only bright (white) lettering, ignoring the scenery behind it."""
    if name == "panel":                       # the smith's window, old or new look
        return max(_seen("panel", norm, pad, text), _seen("panel2", norm, pad, text))
    return _seen(name, norm, pad, text)


def find_mark(norm, name, pad=60):
    """-> (score, dx, dy): where a mark is now compared with where it used to be."""
    if name not in _marks or norm is None:
        return 0.0, 0, 0
    img, x, y = _marks[name]
    h, w = img.shape[:2]
    y0, x0 = max(0, y - pad), max(0, x - pad)
    roi = norm[y0:y + h + pad, x0:x + w + pad]
    if roi.shape[0] < h or roi.shape[1] < w:
        return 0.0, 0, 0
    _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(roi, img, cv2.TM_CCOEFF_NORMED))
    if not np.isfinite(v):
        return 0.0, 0, 0
    return float(v), x0 + loc[0] - x, y0 + loc[1] - y


def ui_tabs(norm):
    """-> (shop tab, stash tab) for the look of the smith's window on screen."""
    if norm is not None and _seen("panel2", norm) > _seen("panel", norm):
        return TAB_SHOP2, TAB_STASH2
    return TAB_SHOP, TAB_STASH


def stash_shift(norm):
    """Offsets of the stash's sub-tab row and its bottom row (search box, 수집), found on screen."""
    sc, dx, dy = find_mark(norm, "personal")
    top = (dx, dy) if sc >= 0.6 else (0, 0)
    sc2, bx, by = find_mark(norm, "stash")
    bottom = (bx, by) if sc2 >= 0.6 else (0, 0)
    return top, bottom


def stash_tabs(norm):
    """Sub-tab positions of the stash on screen (new look: 개인, 1..6, 고유 / old: 개인, 3 shared, 고유),
    already moved by where the row really is. Includes "고유"."""
    new = norm is not None and "subtabs2" in _marks and _seen("subtabs2", norm, pad=12) >= 0.6
    (dx, dy), _ = stash_shift(norm) if norm is not None else ((0, 0), None)
    tabs = dict(STASH_SUBTABS2 if new else STASH_SUBTABS)
    tabs["고유"] = (356, 52) if new else TAB_UNIQUE
    return {k: (x + dx, y + dy) for k, (x, y) in tabs.items()}


def stash_open(norm):
    return find_mark(norm, "stash")[0] >= 0.7


def _seen(name, norm, pad=8, text=False):
    if name not in _marks:
        return 0.0
    img, x, y = _marks[name]
    h, w = img.shape[:2]
    y0, x0 = max(0, y - pad), max(0, x - pad)
    roi = norm[y0:y + h + pad, x0:x + w + pad]
    if roi.shape[0] < h or roi.shape[1] < w:
        return 0.0
    if text:
        roi = (roi.min(axis=2) > 140).astype(np.float32)
        img = (img.min(axis=2) > 140).astype(np.float32)
        if img.std() == 0 or roi.std() == 0:
            return 0.0
    v = cv2.minMaxLoc(cv2.matchTemplate(roi, img, cv2.TM_CCOEFF_NORMED))[1]
    return float(v) if np.isfinite(v) else 0.0


_again = [None]
_full = [None]


def _gold(img):
    """Mask of the game's gold lettering."""
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    r1 = np.maximum(r, 1)
    return ((r >= 100) & (g / r1 > 0.84) & (g / r1 < 0.98) & (b / r1 > 0.48) & (b / r1 < 0.70)).astype(np.float32)


def _gray(img):
    """Mask of neutral grey lettering (the fading on-screen notices)."""
    v = img.astype(np.int32)
    mx, mn = v.max(axis=2), v.min(axis=2)
    return ((mx - mn <= 6) & (mx >= 56) & (mx <= 215)).astype(np.float32)


def _asset(base_dir, name):
    return cv2.imdecode(np.fromfile(os.path.join(base_dir, "assets", name), dtype=np.uint8), cv2.IMREAD_COLOR)


def _match(roi, tmpl):
    if roi.std() == 0 or tmpl.std() == 0:
        return 0.0
    v = cv2.minMaxLoc(cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED))[1]
    return float(v) if np.isfinite(v) else 0.0


_ess_t = [None]


def _ess_templates(base_dir):
    """Essence cell pictures (26 x 26 each, side by side), upper part only: the stack count sits bottom right."""
    if _ess_t[0] is None:
        ts = []
        try:
            img = _asset(base_dir, "essence_icons.png")
            for i in range(img.shape[1] // 26):
                ts.append(img[0:18, i * 26:i * 26 + 20].astype(np.float32))
        except Exception:
            pass
        _ess_t[0] = ts
    return _ess_t[0]


def essence_cell(norm, col, row, base_dir):
    """True when the cell shows an essence (navy night sky with little crosses over a crystal),
    from the picture alone: no info box needed, so it is never mistaken for part of another item."""
    x, y = INV_X + CELL * col, INV_Y + CELL * row
    inner = norm[y + 2:y + CELL - 1, x + 2:x + CELL - 1]
    if inner.shape[0] < 20 or inner.shape[1] < 20:
        return False
    b, g, r = np.median(inner[0:16].reshape(-1, 3), axis=0)
    if b < 30 or b - r < 20:                   # no navy sky
        return False
    roi = norm[max(0, y - 1):y + CELL + 2, max(0, x - 1):x + CELL + 2].astype(np.float32)
    for t in _ess_templates(base_dir):
        if roi.shape[0] >= t.shape[0] and roi.shape[1] >= t.shape[1]:
            v = cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(v) and v >= ESS_NEED:
                return True
    return False


def burning_cell(norm, col, row):
    """An essence cell whose crystal is red / orange: 불타는 공포의 정수 (the others are blue, ...)."""
    x, y = INV_X + CELL * col, INV_Y + CELL * row
    c = norm[y + 2:y + CELL - 2, x + 2:x + CELL - 2].astype(np.int32)
    if c.size == 0:
        return False
    b, r = c[..., 0], c[..., 2]
    red = ((r > 160) & (r - b > 70)).mean()
    blue = ((b > 150) & (b - r > 40)).mean()
    return red >= 0.07 and blue < 0.04


def read_burning(io, geo, norm, base_dir):
    """With the inventory open: the count of every 불타는 공포의 정수 stack in it (hover -> info box).
    -> ({cell key: count}, number of stacks whose count could not be read)"""
    from . import scorer
    snap, bad = {}, 0
    inv = read_inventory(norm)
    for r in range(INV_ROWS):
        for c in range(INV_COLS):
            if not inv[r][c] or not essence_cell(norm, c, r, base_dir) or not burning_cell(norm, c, r):
                continue
            io.move(*geo.raw(*PARK))
            time.sleep(0.12)
            base = io.grab()
            io.move(*geo.raw(INV_X + CELL * c + CELL // 2, INV_Y + CELL * r + CELL // 2))
            q = None
            for _ in range(3):
                time.sleep(0.18)
                hov = io.grab()
                if base is None or hov is None:
                    continue
                tip = tooltip_crop(base, hov, geo)
                if tip is None:
                    continue
                try:
                    ev = scorer.evaluate(tip, geo.s)
                except Exception:
                    ev = None
                if ev:
                    q = ev.get("qty") if ev.get("qty") is not None else ev.get("qty_raw")
                if q is not None:
                    break
            if q is None:
                bad += 1
            else:
                snap["불타는 정수 %d,%d" % (c + 1, r + 1)] = int(q)
    io.move(*geo.raw(*PARK))
    return snap, bad


_act_digits = []


def read_act(frame, top, base_dir):
    """The act of the run from the run info in the top right ("3막 - 공포 영역 ..."). -> (act or None, scores)
    The digit is compared at the game's own 800x600 scale (the same picture at any window size)."""
    from . import spot
    if not _act_digits:
        try:
            with open(os.path.join(base_dir, "assets", "act_digits.json"), "r", encoding="utf-8") as f:
                for lab, rows in json.load(f):
                    _act_digits.append((int(lab), np.array(rows, np.float32)))
        except (OSError, ValueError):
            return None, {}
    h, w = frame.shape[:2]
    s = (h - top) / 600.0
    crop = frame[top: top + int(0.13 * (h - top)), int(0.60 * w):]
    m = spot.area_mask(crop, s)
    best = (0.0, None)
    for t in spot._words.get("공포영역", ()):
        if m.shape[0] < t.shape[0] or m.shape[1] < t.shape[1]:
            continue
        r = cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, (loc, t.shape))
    if best[0] < 0.55:
        return None, {}
    (x, y), (th, tw) = best[1]
    x0 = max(0, x - 110)
    reg = m[max(0, y - 3):y + th + 3, x0:max(x0 + 1, x - 4)]
    cols = np.where(reg.sum(0) > 0)[0]
    runs = []
    if len(cols):
        st = pv = cols[0]
        for c in cols[1:]:
            if c - pv > 5:
                runs.append((st, pv))
                st = c
            pv = c
        runs.append((st, pv))
    word = [r_ for r_ in runs if 22 <= r_[1] - r_[0] <= 44]
    if not word:
        return None, {}
    a, b = word[-1]
    g = crop.astype(np.float32)
    txt = np.clip((g.max(axis=2) - 90.0) / 120.0, 0, 1) * ((g.max(axis=2) - g.min(axis=2)) <= 45)
    lay = cv2.resize(txt, None, fx=1.0 / s, fy=1.0 / s, interpolation=cv2.INTER_AREA)
    f = 2.0                                     # work scale per layout pixel
    lx0, lx1 = int((x0 + a) / f) - 1, int((x0 + b) / f) + 2
    ly0, ly1 = int(max(0, y - 3) / f), int((y + th + 3) / f) + 1
    wd = lay[max(0, ly0):ly1, max(0, lx0):lx1]
    if wd.size == 0:
        return None, {}
    c2 = np.where(wd.max(0) > 0.35)[0]
    if not len(c2):
        return None, {}
    run = 0                                     # width of the digit: the first columns with ink
    for cx in range(c2[0], wd.shape[1]):
        if wd[:, cx].max() <= 0.35:
            break
        run += 1
    if run <= 2:                                # a thin stroke followed by a gap: "1막"
        return 1, {1: 1.0}
    d = wd[:, c2[0]:c2[0] + 7]
    r2 = np.where(d.max(1) > 0.35)[0]
    if not len(r2):
        return None, {}
    d = cv2.resize(d[max(0, r2[0] - 1):r2[-1] + 2], (7, 9), interpolation=cv2.INTER_LINEAR)
    d = d - d.mean()
    sc = {}
    for lab, t in _act_digits:
        tt = t - t.mean()
        v = float((d * tt).sum() / (np.sqrt((d * d).sum() * (tt * tt).sum()) + 1e-9))
        sc[lab] = max(sc.get(lab, -1.0), v)
    order = sorted(sc, key=lambda k: -sc[k])
    act = order[0] if sc[order[0]] >= 0.8 and (len(order) < 2 or sc[order[0]] - sc[order[1]] >= 0.05) else None
    return act, sc


def run_ended(norm, base_dir):
    """How strongly the lower half shows the run-end hint 'ENTER: 다시 하기' (gold lettering). After a
    death in a new stage the hint reads 'ENTER: 공포 영역 N' instead, so 'ENTER:' alone counts too."""
    if _again[0] is None:
        _again[0] = (_gold(_asset(base_dir, "again.png")), _gold(_asset(base_dir, "enter.png")))
    if menu_text(norm) >= 0.55 or confirm_shown(norm):
        return 0.0                                # the Esc menu (its hint line looks like the end screen's)
    roi = _gold(norm[360:580, 100:700])
    return max(_match(roi, _again[0][0]), _match(roi, _again[0][1]) - 0.1)


_pmsg = [None]


def _gold_dim(img):
    """Mask of dim gold lettering (the big fading announcements)."""
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    r1 = np.maximum(r, 1)
    return ((r >= 70) & (g / r1 > 0.78) & (g / r1 < 0.99) & (b / r1 > 0.40) & (b / r1 < 0.74)).astype(np.float32)


def portal_opened(norm, base_dir=None):
    """How strongly the screen shows the announcement '... 차원문이 열렸습니다' (the boss is dead)."""
    if _pmsg[0] is None and base_dir:
        img = cv2.imdecode(np.fromfile(os.path.join(base_dir, "assets", "portal_msg.png"), dtype=np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        _pmsg[0] = (img > 127).astype(np.float32)
    return _match(_gold_dim(norm[70:260, 80:740]), _pmsg[0])


def died(norm):
    """On the run-end screen: True when the dark red title '사망했습니다' is showing."""
    roi = norm[75:130, 230:570].astype(np.int32)
    r, g, b = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    return int(((r >= 80) & (r <= 140) & (g < 48) & (b < 48) & (r > 2.4 * np.maximum(g, 1))).sum()) >= 200


def inventory_full(norm, base_dir):
    """How strongly the screen shows the grey notice '(C: 공간 확보)'."""
    if _full[0] is None:
        _full[0] = _gray(_asset(base_dir, "full.png"))
    return _match(_gray(norm[120:300, 150:700]), _full[0])


def cell_color(norm, col, row):
    """'' for an empty cell, else one of COLORS or 'other'.

    Uses the most common flat colour in the cell: the game paints a solid tint
    behind every item, while item art is never one flat colour.
    """
    x, y = INV_X + CELL * col, INV_Y + CELL * row
    px = (norm[y + 2:y + CELL - 1, x + 2:x + CELL - 1].reshape(-1, 3) // 4).astype(np.int32)
    keys, counts = np.unique(px[:, 0] * 4096 + px[:, 1] * 64 + px[:, 2], return_counts=True)
    total = float(len(px))
    best, best_n, dark = None, 0, 0
    for k, n in zip(keys, counts):
        b, g, r = (k // 4096) * 4, (k // 64 % 64) * 4, (k % 64) * 4
        mx, mn = max(b, g, r), min(b, g, r)
        if mx <= 24 and mx - mn < 8:
            dark += n
        elif mx - mn >= 14 and n > best_n:
            best, best_n = (b, g, r), n
    if best is None or best_n / total < 0.05:
        return "" if dark / total >= 0.45 else "other"
    hue = int(cv2.cvtColor(np.uint8([[best]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    if hue < 9 or hue > 170:
        return "red"
    if hue < 26:
        return "tan"
    if hue < 40:
        return "yellow"
    if hue < 90:
        return "green"
    if hue < 165:
        return "blue"
    return "other"


def read_inventory(norm):
    return [[cell_color(norm, c, r) for c in range(INV_COLS)] for r in range(INV_ROWS)]


def cell_center(col, row):
    return INV_X + CELL * col + CELL // 2, INV_Y + CELL * row + CELL // 2


def tooltip_crop(base, hover, geo):
    """Largest changed block between two raw frames, or None."""
    d = cv2.absdiff(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hover, cv2.COLOR_BGR2GRAY))
    mask = (d > 22).astype(np.uint8)
    x_lo, x_hi = max(0, int(geo.x0)), int(geo.x0 + 800 * geo.s)
    mask[:, :x_lo] = 0
    mask[:, x_hi:] = 0
    k = max(5, int(9 * geo.s)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [stats[i] for i in range(1, n) if stats[i][4] >= (18 * geo.s) ** 2]
    if not boxes:
        return None
    x = min(b[0] for b in boxes)
    y = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    if x2 - x < 70 * geo.s or y2 - y < 28 * geo.s:
        return None
    p = int(3 * geo.s)
    return hover[max(0, y - p):y2 + p, max(0, x - p):x2 + p].copy()


# worn-item slots of the right-hand panel (800x600 layout): name, centre, half width, half height
WORN_SLOTS = [("무기", (447, 159), 18, 40), ("방패·화살통", (678, 159), 18, 40), ("투구", (561, 93), 18, 18),
              ("갑옷", (561, 178), 20, 30), ("목걸이", (620, 106), 8, 8), ("장갑", (447, 265), 16, 16),
              ("반지1", (506, 249), 8, 8), ("허리띠", (561, 249), 18, 8), ("반지2", (620, 249), 8, 8),
              ("신발", (678, 265), 16, 16)]


def empty_worn(norm):
    """Names of worn-item slots that look empty (dark, no item tint) while the panel is open."""
    out = []
    for name, (x, y), hw, hh in WORN_SLOTS:
        r = norm[y - hh:y + hh, x - hw:x + hw]
        if r.size == 0:
            continue
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        if g.mean() < 30 and sat.mean() < 90:
            out.append(name)
    return out


def _gray_layout(io, geo, pt):
    io.move(*geo.raw(*pt))
    time.sleep(0.25)
    f = io.grab()
    return None if f is None else cv2.cvtColor(geo.normalize(f), cv2.COLOR_BGR2GRAY).astype(np.int16)


def item_on_cursor(io, geo):
    """Is an item hanging on the mouse cursor? (only with a panel open: the spots are fixed stone/UI)
    The cursor rests on spot A, then on spot B: an item on the cursor shows up at A only when the cursor is
    there, a big patch; the bare hand cursor is small."""
    A, B = (400, 500), (230, 585)          # B: far from A (bottom bar), so the item is only at A once
    ga = _gray_layout(io, geo, A)
    gb = _gray_layout(io, geo, B)
    if ga is None or gb is None:
        return False
    x, y = A
    d = np.abs(ga[y - 45:y + 75, x - 45:x + 45] - gb[y - 45:y + 75, x - 45:x + 45]) > 30
    return int(d.sum()) > 1200


_WORN_REF = {}          # slot name -> gray picture of that worn slot, taken when nothing was on the cursor


def _slot_gray(norm, name):
    for k, (x, y), hw, hh in WORN_SLOTS:
        if k == name:
            return cv2.cvtColor(norm[y - hh:y + hh, x - hw:x + hw], cv2.COLOR_BGR2GRAY).astype(np.int16)
    return None


def remember_worn(norm):
    """Pictures of the worn slots as they are now (called when nothing is on the cursor)."""
    for name, _, _, _ in WORN_SLOTS:
        g = _slot_gray(norm, name)
        if g is not None and g.size:
            _WORN_REF[name] = g


def _changed_slots(norm):
    """Worn slots that no longer look like they did (the item has gone from there), most changed first."""
    out = []
    for name, _, _, _ in WORN_SLOTS:
        ref, cur = _WORN_REF.get(name), _slot_gray(norm, name)
        if ref is None or cur is None or ref.shape != cur.shape:
            continue
        d = float(np.abs(ref - cur).mean())
        if d > 12:
            out.append((d, name))
    return [n for d, n in sorted(out, reverse=True)]


def restore_worn(io, log, halt=None):
    """Safety net before any window is closed: an item picked up by a stray click (e.g. the worn bow) hangs
    on the cursor, and closing the window would drop it on the ground. It is put back on where a worn slot
    is now empty; only if that cannot be done is it laid in an empty bag cell and the helper stopped.
    -> True when an item was on the cursor."""
    f = io.grab()
    if f is None:
        return False
    geo = Geo(f)
    n = geo.normalize(f)
    if seen(n, "inv") < 0.7:
        return False                               # no equipment panel on screen
    if not item_on_cursor(io, geo):
        remember_worn(geo.normalize(io.grab()))    # all is well: keep a picture of the worn slots
        return False
    n = geo.normalize(io.grab())
    order = _changed_slots(n)                      # the slot it came from (compared with before)
    order += [k for k in empty_worn(n) if k not in order]
    if "무기" not in order:
        order.append("무기")                        # the weapon is the one a stray click usually takes
    pos = dict((k, p) for k, p, _, _ in WORN_SLOTS)
    for name in order:
        before = _slot_gray(n, name)
        io.click(*geo.raw(*pos[name]))
        time.sleep(0.45)
        if not item_on_cursor(io, geo):
            log("마우스에 들려 있던 아이템을 비어 있던 착용 칸(%s)에 다시 착용함" % name)
            remember_worn(geo.normalize(io.grab()))
            return True
        n2 = geo.normalize(io.grab())
        after = _slot_gray(n2, name)
        if before is not None and after is not None and float(np.abs(before - after).mean()) > 12:
            io.click(*geo.raw(*pos[name]))         # it swapped with what was worn there: swap back
            time.sleep(0.45)
        n = geo.normalize(io.grab())
    # could not be put on: into the bags, and stop so nothing else happens to it
    inv = read_inventory(n)
    spot = next(((c, r) for r in range(INV_ROWS) for c in range(INV_COLS) if not inv[r][c]), None)
    if spot is None:
        log("마우스에 아이템이 들려 있는데 착용 칸도 인벤토리 빈칸도 없음 - 자동 진행을 멈춤")
    else:
        io.click(*geo.raw(*cell_center(*spot)))
        time.sleep(0.4)
        log("마우스에 들려 있던 아이템을 착용하지 못해 인벤토리 빈칸(%d,%d)에 내려놓고 자동 진행을 멈춤"
            % (spot[0] + 1, spot[1] + 1))
    if halt:
        halt()
    return True


def forged_tip(tip):
    """An item the player has reinforced (강화): its boosted lines are drawn in orange ("... ×4").
    Such an item is the player's own work: never registered, sold or moved by the sorter."""
    if tip is None:
        return False
    b, g, r = [tip[:, :, i].astype(np.int32) for i in range(3)]
    orange = (r > 190) & (g > 105) & (g < 190) & (b < 70) & (r - g > 50)
    return int(orange.sum()) > 60


def same_tip(a, b):
    """True when two info-box crops show the same text (allows a few pixels of shift)."""
    if abs(a.shape[0] - b.shape[0]) > 4 or abs(a.shape[1] - b.shape[1]) > 4:
        return False
    ga = (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) > 110).astype(np.float32)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (ga.shape[1], ga.shape[0]))
    gb = (gb > 110).astype(np.float32)
    if ga.sum() < 20 or gb.sum() < 20:
        return ga.sum() < 20 and gb.sum() < 20
    core = ga[3:-3, 3:-3]
    if core.std() == 0:
        return False
    v = cv2.minMaxLoc(cv2.matchTemplate(gb, core, cv2.TM_CCOEFF_NORMED))[1]
    return bool(np.isfinite(v) and v > 0.93)


def save_snap(base_dir, img, label):
    d = os.path.join(base_dir, "snaps")
    os.makedirs(d, exist_ok=True)
    fn = time.strftime("%Y%m%d_%H%M%S") + "_%03d.jpg" % (int(time.time() * 1000) % 1000)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    buf.tofile(os.path.join(d, fn))
    with open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"file": fn, "rule": label, "time": time.strftime("%m-%d %H:%M")},
                           ensure_ascii=False) + "\n")
    files = sorted(x for x in os.listdir(d) if x.endswith(".jpg"))
    for old in files[:-300]:
        try:
            os.remove(os.path.join(d, old))
        except OSError:
            pass
    return fn


def score_tip(base_dir, tip, geo, log):
    """Scores one info-box image, saves it for the phone, and writes the result to the log."""
    try:
        ev = scorer.evaluate(tip, geo.s)
    except Exception as e:
        ev = None
        log("점수 계산 오류: %s" % e)
    if not ev:
        save_snap(base_dir, tip, "아이템 정보")
        return None
    label = ev["name"] or ev["quality"]
    if ev["score"] is not None:
        label = "%s %d점" % (label, ev["score"])
    fn = save_snap(base_dir, tip, label)
    if ev["quality"] in ("유니크", "세트"):
        log("%s%s" % (label, " (%s)" % ev["note"] if ev["note"] else ""))
        try:
            with open(os.path.join(base_dir, "items.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": ev["name"],
                                    "quality": ev["quality"], "score": ev["score"], "note": ev["note"],
                                    "file": fn}, ensure_ascii=False) + "\n")
        except OSError:
            pass
    return ev


def score_hover(io, base_dir, log, stop):
    """Scores the item the mouse is pointing at right now."""
    with_tip = io.grab()
    if with_tip is None:
        return "게임 창을 찾지 못함"
    geo = Geo(with_tip)
    back = io.mouse.position
    io.move(*geo.raw(*PARK))
    time.sleep(0.35)
    plain = io.grab()
    io.mouse.position = back
    tip = None if plain is None else tooltip_crop(plain, with_tip, geo)
    if tip is None:
        return "정보창을 찾지 못함 (아이템에 마우스를 올린 채로 실행하세요)"
    ev = score_tip(base_dir, tip, geo, log)
    if not ev:
        return "정보창을 읽지 못함"
    if ev["score"] is None:
        return "%s %s: %s" % (ev["quality"], ev["name"], ev["note"])
    return "점수 확인 끝"


def run(io, opts, base_dir, log, stop):
    """io: object with grab() -> raw BGR frame or None, move(x, y), click(x, y, mod).
    Coordinates passed to io are raw client pixels. Returns a short result text."""
    frame = io.grab()
    if frame is None:
        return "게임 창을 찾지 못해 정리를 건너뜀"
    geo = Geo(frame)
    if seen(geo.normalize(frame), "panel") < 0.6:
        return "마을의 상점·보관함 창이 열려 있지 않아 정리를 건너뜀"

    def shot():
        f = io.grab()
        return None if f is None else geo.normalize(f)

    def tap(pt, mod=None, wait=0.35):
        io.click(*geo.raw(*pt), mod=mod)
        time.sleep(wait)

    def park():
        io.move(*geo.raw(*PARK))
        time.sleep(0.2)

    actions = opts.get("actions", {})
    park()
    norm = shot()
    inv = read_inventory(norm)
    n_items = sum(1 for row in inv for c in row if c)
    log("인벤토리에서 칸 %d개가 차 있음" % n_items)

    saved = 0
    if opts.get("save_tips", True) and n_items:
        base = io.grab()
        tips = []
        for r in range(INV_ROWS):
            for c in range(INV_COLS):
                if stop.is_set() or not inv[r][c]:
                    continue
                io.move(*geo.raw(*cell_center(c, r)))
                time.sleep(0.3)
                hov = io.grab()
                tip = None if hov is None else tooltip_crop(base, hov, geo)
                if tip is None or any(same_tip(tip, t) for t in tips):
                    continue
                tips.append(tip)
                score_tip(base_dir, tip, geo, log)
                saved += 1
        park()

    thr = int(opts.get("min_score", 60))
    kept = sold = 0
    kinds = set(actions.get(c) for row in inv for c in row if c)
    if kinds & {"sell", "sell_low"} and not stop.is_set():
        tap(ui_tabs(shot())[0], wait=0.5)
        park()
        if seen(shot(), "shop") < 0.7:
            log("잡화 화면을 확인하지 못해 판매를 건너뜀")
        else:
            sold = sell_sweep(io, geo, actions, thr, base_dir, log, stop)
    if kinds & {"stash", "sell_low"} and not stop.is_set():
        tap(ui_tabs(shot())[1], wait=0.5)
        park()
        tabs = stash_tabs(shot())
        tap(tabs.get(opts.get("stash_tab", "개인"), tabs["개인"]), wait=0.4)
        park()
        if not stash_open(shot()):
            log("보관함 화면을 확인하지 못해 보관을 건너뜀")
        else:
            kept = sweep_cells(io, geo, lambda col: actions.get(col) in ("stash", "sell_low"), "ctrl",
                               "보관", log, stop)
    return "정리 끝: 보관 %d개, 판매 %d개, 정보창 저장 %d장" % (kept, sold, saved)


def _park(io, geo):
    io.move(*geo.raw(*PARK))
    time.sleep(0.2)


def _cells(io, geo):
    f = io.grab()
    return None if f is None else read_inventory(geo.normalize(f))


def sweep_cells(io, geo, wanted, mod, word, log, stop, judge=None):
    """Modifier-click every inventory cell whose colour is wanted (and that judge() accepts).
    -> number of clicks after which the inventory had fewer filled cells."""
    moved, skip, failed = 0, set(), 0
    for _ in range(INV_COLS * INV_ROWS):
        if stop.is_set():
            break
        cur = _cells(io, geo)
        if cur is None:
            break
        target = next(((c, r) for r in range(INV_ROWS) for c in range(INV_COLS)
                       if cur[r][c] and wanted(cur[r][c]) and (c, r) not in skip), None)
        if target is None:
            break
        if judge is not None and not judge(target, cur[target[1]][target[0]]):
            skip.add(target)
            continue
        before = sum(1 for row in cur for v in row if v)
        io.click(*geo.raw(*cell_center(*target)), mod=mod)
        time.sleep(0.35)
        _park(io, geo)
        now = _cells(io, geo)
        if now is not None and sum(1 for row in now for v in row if v) < before:
            moved += 1
        else:
            skip.add(target)
            failed += 1
    if failed:
        log("%s 실패한 칸 %d개 (자리가 없거나 옮길 수 없는 아이템)" % (word, failed))
    return moved


def sell_sweep(io, geo, actions, thr, base_dir, log, stop):
    """Shift-click 'sell' colours; for 'sell_low' colours sell only items scoring under thr."""
    def judge(cell, color):
        f = io.grab()
        if f is None or essence_cell(geo.normalize(f), cell[0], cell[1], base_dir):
            return False                        # essences are never sold
        if actions.get(color) != "sell_low":
            return True
        _park(io, geo)
        base = io.grab()
        io.move(*geo.raw(*cell_center(*cell)))
        time.sleep(0.3)
        hov = io.grab()
        _park(io, geo)
        tip = None if base is None or hov is None else tooltip_crop(base, hov, geo)
        if tip is None:
            return False
        ev = score_tip(base_dir, tip, geo, log)
        return bool(ev and ev["score"] is not None and ev["score"] < thr)

    return sweep_cells(io, geo, lambda col: actions.get(col) in ("sell", "sell_low"), "shift",
                       "판매", log, stop, judge)


def free_space(io, opts, base_dir, log, stop):
    """Inventory window already open during a run: sell what the colour settings allow.
    -> (text, sold count)"""
    frame = io.grab()
    if frame is None:
        return "게임 창을 찾지 못함", 0
    geo = Geo(frame)
    _park(io, geo)
    f = io.grab()
    if f is None or seen(geo.normalize(f), "inv") < 0.7:
        return "인벤토리 창이 열리지 않아 공간 확보를 건너뜀", 0
    sold = sell_sweep(io, geo, opts.get("actions", {}), int(opts.get("min_score", 60)), base_dir, log, stop)
    if sold == 0:
        return "공간 확보: 팔 수 있는 아이템이 없음 (인벤토리 정리 탭에서 판매할 색을 지정하세요)", 0
    return "공간 확보: %d개 판매" % sold, sold


def inventory_open(io):
    f = io.grab()
    return f is not None and seen(Geo(f).normalize(f), "inv") >= 0.7


WP_TAB1, WP_TAB_PITCH, WP_TAB_Y = 119, 63, 75        # waypoint tabs (I~V / 1-10, 11-20, 21-28)
WP_X, WP_ROW1, WP_PITCH = 230, 135, 36.1              # first page (보통/악몽/지옥/공포 영역 N/완료한 단계 다시 하기)
WP_PER_TAB, WP_LIST_ROW1, WP_LIST_PITCH = 10, 136, 32.1   # stage list: ten stages per tab, closer rows


def wp_lit_tab(norm):
    """Which of the waypoint list's five tabs is the open one (it is drawn lighter). None if unclear."""
    g = cv2.cvtColor(norm[64:86], cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = [float(g[:, int(WP_TAB1 + WP_TAB_PITCH * i) - 25:int(WP_TAB1 + WP_TAB_PITCH * i) + 25].mean())
         for i in range(5)]
    order = sorted(range(5), key=lambda i: -v[i])
    return order[0] if v[order[0]] - v[order[1]] >= 12 else None     # waypoint list: click x, first row centre, row spacing
BTN_COLLECT = (265, 461)        # "수집": the game's own register-everything-new button
TAB_UNIQUE = (355, 51)          # the "고유" (collection) sub-tab of the stash


def _fingerprint(norm, col, row):
    x, y = INV_X + CELL * col, INV_Y + CELL * row
    small = cv2.resize(norm[y + 2:y + CELL - 1, x + 2:x + CELL - 1], (6, 6), interpolation=cv2.INTER_AREA)
    return (small // 24).tobytes()


_rec_n = [0]
RUN_INFO = {"stage": 0}                  # set by the program: the stage being repeated (for the essence log)
ESS_LOG = "burning_log.jsonl"           # 불타는 공포의 정수 only


def record_essences(snap, stashed):
    """One line per tidy-up: the count of every essence stack in the bags (before any goes to the stash).
    The phone works out how many came in per run / per hour / per stage from these lines."""
    try:
        with open(paths.p(ESS_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "stage": int(RUN_INFO.get("stage") or 0),
                                "stacks": snap, "stashed": sorted(stashed)}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def record_item(base_dir, action, ev, tip, tab=""):
    """Keep a line (and the info-box picture) for every item that was registered or stashed:
    collected/index.jsonl, read by the phone page. Sold items are never written here."""
    try:
        d = paths.p("collected")
        os.makedirs(d, exist_ok=True)
        now = time.time()
        _rec_n[0] += 1
        img = ""
        if tip is not None:
            img = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_%03d.jpg" % (_rec_n[0] % 1000)
            ok, buf = cv2.imencode(".jpg", tip, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                buf.tofile(os.path.join(d, img))
            else:
                img = ""
        ev = ev or {}
        line = {"t": round(now, 1), "action": action, "tab": tab,
                "name": ev.get("name") or ev.get("seen_name") or "", "raw": ev.get("raw") or "", "kind": ev.get("kind") or "",
                "quality": ev.get("quality") or "", "color": ev.get("color") or "", "score": ev.get("score"),
                "img": img}
        with open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


NEED_FRONT = "__need_front__"


def cleanup(io, opts, base_dir, log, stop):
    """The tidy-up below; with the game behind other windows its info boxes may not show at all: then the
    game is brought to the front for the tidy-up only, and the window that had the front gets it back."""
    text, done = _cleanup(io, opts, base_dir, log, stop, probe=True)
    if text != NEED_FRONT:
        return text, done
    prev_fg = None
    try:
        import ctypes
        from . import win
        prev_fg = ctypes.windll.user32.GetForegroundWindow()
        if win.focus(io.hwnd):
            log("게임 창이 뒤에 있어 정보창이 안 뜨는 아이템이 있음: 정리하는 동안만 게임 창을 앞으로 가져옴")
            time.sleep(0.5)
        else:
            prev_fg = None
            log("게임 창을 앞으로 가져오지 못함: 정보창 없이 정리 (읽지 못한 아이템은 보관함으로)")
        return _cleanup(io, opts, base_dir, log, stop, probe=False)
    finally:
        if prev_fg:
            try:
                win.focus(prev_fg)
            except Exception:
                pass


def _cleanup(io, opts, base_dir, log, stop, probe=False):
    """Charsi's window is open: register/replace uniques and sets in the collection tab, move
    essences and keepers to the stash, sell the rest. Items that cannot be read go to the stash
    instead of being sold. -> (text, done)"""
    class _Either:                              # the caller's stop, or this tidy-up's own abort
        def __init__(self, outer):
            self.outer, self.mine = outer, False

        def is_set(self):
            return self.mine or (self.outer is not None and self.outer.is_set())

        def set(self):
            self.mine = True                       # (never stops the whole helper)

    stop = _Either(stop)
    frame = io.grab()
    if frame is None:
        return "게임 창을 찾지 못함", False
    geo = Geo(frame)
    if seen(geo.normalize(frame), "panel") < 0.6:
        return "대장장이 창이 열려 있지 않음", False
    keep_min = int(opts.get("keep_min", 0))
    words = [w.strip() for w in str(opts.get("keep_words", "정수,주얼,작은 부적")).split(",") if w.strip()]
    swap_better = bool(opts.get("swap_better", False))
    reg_path = paths.p("collection.json")
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        registry = {}
    tips, known, tries = [], {}, {}
    count = {"register": 0, "swap": 0, "stash": 0, "sell": 0}

    def norm_now():
        f = io.grab()
        return None if f is None else geo.normalize(f)

    ess_cells = set()                           # cells whose picture is an essence (latest look)

    burn_cells = set()                          # essence cells holding 불타는 공포의 정수

    def ess_scan(n, inv):
        ess_cells.clear()
        burn_cells.clear()
        for r_ in range(INV_ROWS):
            for c_ in range(INV_COLS):
                if inv[r_][c_] and essence_cell(n, c_, r_, base_dir):
                    ess_cells.add((c_, r_))
                    if burning_cell(n, c_, r_):
                        burn_cells.add((c_, r_))

    shift = {"bottom": (0, 0)}

    def open_stash(name):
        """Stash tab of the smith's window (old or new look), then one of its sub-tabs by name
        ("개인", "공유 1".., "고유"). -> opened? (False also when this stash has no such tab)"""
        n0 = norm_now()
        if n0 is None or not stash_open(n0):
            tap(ui_tabs(n0)[1], wait=0.5)
            _park(io, geo)
        tabs = stash_tabs(norm_now())
        if name not in tabs:
            return False
        tap(tabs[name], wait=0.45)
        _park(io, geo)
        n_ = norm_now()
        if n_ is None or not stash_open(n_):
            g_ = io.grab()
            if g_ is not None:
                save_snap(base_dir, g_, "보관함 화면 인식 실패")
            return False
        shift["bottom"] = stash_shift(n_)[1]
        return True

    def open_shop():
        tap(ui_tabs(norm_now())[0], wait=0.5)
        _park(io, geo)

    last_ev = {}               # top-left cell -> info of the item there (latest look)
    tip_of = {}                # cell -> the info-box picture seen for its item (for the collection log)
    tip_by_fp = {}
    base_img = [None]          # the screen with the cursor parked: reused until something changes
    assumed = {}               # cell -> (top-left cell of its item, info): not hovered separately
    footprint = {}             # top-left cell -> the cells its item is taken to cover

    park_ref = [None]

    def park_patch():
        """The empty stone under the parked cursor (an item held on the cursor would show there)."""
        _park(io, geo)
        n_ = norm_now()
        if n_ is None:
            return None
        x, y = PARK
        return n_[y - 28:y + 28, x - 40:x + 40].astype(np.int16)

    def holding():
        if park_ref[0] is None:
            return False
        p_ = park_patch()
        return p_ is not None and float(np.abs(p_ - park_ref[0]).mean()) > 14.0

    def tap(pt, mod=None, wait=0.4):
        io.click(*geo.raw(*pt), mod=mod)
        time.sleep(wait)
        base_img[0] = None
        if mod and holding():
            # Ctrl / Shift did not reach the game: the click picked the item up instead of moving it.
            # Put it straight back where it was, and stop: an item on the cursor must never be dropped
            # anywhere else (on the ground or on the shop it would be lost or sold).
            io.click(*geo.raw(*pt))
            time.sleep(0.4)
            if holding():
                io.click(*geo.raw(*pt))
                time.sleep(0.4)
            log("Ctrl/Shift 클릭이 안 먹어서 아이템이 마우스에 들림 → 제자리에 되돌리고 정리를 멈춤")
            stop.set()

    def acted(cell):
        """An item was clicked: the cells taken to be part of it are looked at again if still filled."""
        for x in footprint.pop(cell, [cell]):
            if x in assumed and assumed[x][0] == cell:
                del assumed[x]

    def inspect(cell, fresh=False):
        """Info of the item in a cell. One hover per item: the rest of an item (same background colour
        in a block from its top-left cell) takes the same info until something proves otherwise."""
        ess = cell in ess_cells
        if not fresh and cell in assumed and not ess:
            return assumed[cell][1]
        assumed.pop(cell, None)                 # looked at for itself from here on
        if base_img[0] is None:
            _park(io, geo)
            base_img[0] = io.grab()
        base = base_img[0]
        if base is None:
            return None
        nb = geo.normalize(base)
        fp = _fingerprint(nb, *cell)
        if fp in known and not fresh and not ess:      # an essence is always read itself (its count)
            ev = known[fp]
            tip_of[cell] = tip_by_fp.get(fp)
        else:
            io.move(*geo.raw(*cell_center(*cell)))
            time.sleep(0.18)
            hov = io.grab()
            tip = None if hov is None else tooltip_crop(base, hov, geo)
            ev = None
            if tip is not None:
                for t, e in tips:
                    if same_tip(tip, t):
                        ev = e
                        break
                else:
                    try:
                        ev = scorer.evaluate(tip, geo.s, words)
                        if ev is not None:
                            ev["forged"] = forged_tip(tip)
                    except Exception as e:
                        log("정보창 판독 오류: %s" % e)
                    tips.append((tip, ev))
                    if ev and opts.get("save_tips"):
                        label = ev["name"] or ev["raw"] or ev["kind"] or ev["quality"]
                        if ev["score"] is not None:
                            label = "%s %d점" % (label, ev["score"])
                        save_snap(base_dir, tip, label)
            known[fp] = ev
            tip_by_fp[fp] = tip
            tip_of[cell] = tip
        # the rest of this item: a block of the same background colour (at most 2 wide, 4 tall)
        inv = read_inventory(nb)
        c, r = cell
        col = inv[r][c]
        cells = [cell]
        if col and col != "other" and not ess:   # an essence is one cell; never part of another block
            wdt = 2 if (c + 1 < INV_COLS and inv[r][c + 1] == col and (c + 1, r) not in footprint
                        and (c + 1, r) not in ess_cells) else 1
            hgt = 1
            while r + hgt < INV_ROWS and hgt < 4 and all(inv[r + hgt][c + i] == col and (c + i, r + hgt) not in ess_cells
                                                          for i in range(wdt)):
                hgt += 1
            cells = [(c + i, r + j) for j in range(hgt) for i in range(wdt)]
        footprint[cell] = cells
        for x in cells[1:]:
            if x not in assumed and x not in footprint:
                assumed[x] = (cell, ev)
        return ev

    def title(ev):
        return ev["name"] or ev["raw"] or ev["kind"] or ev["quality"]

    def decide(ev, cell=None):
        # essences: never sold; a stack stays in the bags until it holds ESSENCE_HOLD (90), then to the stash.
        # Known by the cell picture even when the info box was not read, or by the box (name / stack count).
        if (cell is not None and cell in ess_cells) or (ev and (ev.get("kind") == "정수" or (
                ev.get("qty") is not None and not ev.get("keep") and ev.get("color") not in ("gold", "green")))):
            q = ev.get("qty") if ev else None
            if q is None and ev and cell is not None and cell in ess_cells:
                q = ev.get("qty_raw")
            return "essence" if q is not None and q >= ESSENCE_HOLD else "hold"
        if not ev:
            return "leave"                      # no info box could be read: never sell blind
        if ev.get("forged"):
            return "leave"                      # reinforced by the player (강화): never touched
        if ev.get("keep_name"):
            return "keep"                       # one of the best items (바람살, 신념, ...): always to the stash
        if ev["color"] == "yellow" and not ev["keep"] and cell is not None:
            # a rare quiver of arrows / bolts is 1 cell wide and 3 tall: kept even when its base name
            # cannot be read (a rare dagger, scepter or javelin of that size is kept too: no harm done)
            cells = footprint.get(cell, [])
            if len(cells) == 3 and len({c for c, r in cells}) == 1:
                ev["kind"], ev["keep"] = "레어 화살", True
        if ev["keep"]:
            return "essence"                    # essences, jewels, small charms, skill grand charms
        if ev["color"] in ("gold", "green"):
            if ev.get("charm"):
                return "keep"                   # unique charms are kept even when one is registered
            if not ev["name"]:
                return "sell"                   # "수집" already registered every new one: this is a spare
            if not ev["stored"] and tries.get(ev["name"], 0) < 1:
                return "register"
            sc, old = ev["score"], registry.get(ev["name"])
            if swap_better and sc is not None and tries.get(ev["name"], 0) < 3 and (old is None or sc > old):
                return "swap"
            return "sell"                       # already in the collection: never stashed
        if ev["color"] in ("blue", "yellow", "white"):
            return "sell"
        return "leave"

    def walk(handler):
        """Visit every filled cell once; handler(cell, ev, decision) may click."""
        done = set()
        for _ in range(INV_COLS * INV_ROWS * 2):
            if stop.is_set():
                return
            n = norm_now()
            if n is None:
                return
            inv = read_inventory(n)
            ess_scan(n, inv)
            cell = next(((c, r) for r in range(INV_ROWS) for c in range(INV_COLS)
                         if inv[r][c] and (c, r) not in done), None)
            if cell is None:
                return
            done.add(cell)
            ev = inspect(cell)
            if cell in footprint:
                last_ev[cell] = ev
            handler(cell, ev, decide(ev, cell))

    def filled(cell):
        n = norm_now()
        return n is not None and bool(read_inventory(n)[cell[1]][cell[0]])

    park_ref[0] = park_patch()                  # what the parked cursor sees with nothing held
    n0 = norm_now()
    inv0 = read_inventory(n0) if n0 is not None else None
    if inv0 is not None and not any(v for row in inv0 for v in row):
        if probe:
            record_essences({}, set())            # no 불타는 정수 in the bags this run
        return "자동 정리: 인벤토리가 비어 있어 할 일 없음", True
    # look at every item once first, so that only the steps that are needed get done
    plan = {}

    ess_snap, ess_out, ess_bad = {}, set(), []

    def survey(cell, ev, d):
        plan[d] = plan.get(d, 0) + 1
        if d not in ("hold", "essence"):
            return
        nm = (ev.get("name") or ev.get("raw") or "") if ev else ""
        burning = cell in burn_cells or "불타" in nm.replace(" ", "")
        if not burning:
            return                              # only 불타는 공포의 정수 is counted
        key = "불타는 정수 %d,%d" % (cell[0] + 1, cell[1] + 1)
        q = (ev.get("qty") if ev.get("qty") is not None else ev.get("qty_raw")) if ev else None
        if q is None:
            ess_bad.append(key)
            return
        ess_snap[key] = int(q)
        if d == "essence":
            ess_out.add(key)

    walk(survey)
    if stop.is_set():
        return "자동 정리 중단", False
    if probe and plan.get("leave") and not io.front():
        return NEED_FRONT, False                # info boxes missing: the game is behind other windows
    if ess_bad:
        log("불타는 정수 묶음 %d개의 수량을 읽지 못함: 이번 판 정수 개수는 세지 않음" % len(ess_bad))
    else:
        record_essences(ess_snap, ess_out)
    if ess_snap and not ess_bad:
        log("불타는 정수: 인벤토리에 %d개 (%s)" % (sum(ess_snap.values()),
                                           ", ".join("%s칸 %d개" % (k.split()[-1], v) for k, v in ess_snap.items())))
    if plan.get("hold"):
        log("정수 %d묶음은 인벤토리에 그대로 둠 (%d개 이상인 묶음만 보관함으로, 정수는 절대 판매 안 함)" % (plan["hold"], ESSENCE_HOLD))
    gold_green = inv0 is not None and any(v in ("tan", "green") for row in inv0 for v in row)
    stash_kinds = ("essence", "keep", "leave")

    # 1) collection tab: register new ones, try a swap when this copy may be better
    forged_here = any(ev_ and (ev_.get("forged") or ev_.get("keep_name")) for ev_ in last_ev.values())
    if forged_here:
        log("강화한 아이템이나 보관 목록의 아이템이 인벤토리에 있음: "
            "게임의 '수집' 버튼은 쓰지 않고 하나씩만 등록")
    if not gold_green and not plan.get("register") and not plan.get("swap"):
        pass
    else:
        if not open_stash("고유"):
            log("보관함 고유 탭을 확인하지 못해 등록을 건너뜀")
        else:
            seen_items = {c_: tip_of.get(c_) for c_ in footprint}
            evs = {c_: ev_ for c_, ev_ in last_ev.items()}
            if not forged_here:
                tap((BTN_COLLECT[0] + shift["bottom"][0], BTN_COLLECT[1] + shift["bottom"][1]), wait=0.9)  # the game registers every new unique and set itself
            _park(io, geo)
            n_after = norm_now()
            inv_after = read_inventory(n_after) if n_after is not None else None
            if inv_after is not None:
                for c_, ev_ in evs.items():
                    if ev_ and ev_.get("color") in ("gold", "green") and not inv_after[c_[1]][c_[0]]:
                        record_item(base_dir, "등록", ev_, seen_items.get(c_))
            assumed.clear()                          # items left the bags: guesses about blocks are void
            footprint.clear()
            _park(io, geo)
            log("고유 탭: '수집'으로 새 유니크·세트 일괄 등록")

            def collect(cell, ev, d):
                if d not in ("register", "swap"):
                    return
                if cell in assumed:              # never register/swap on a guess: look at it itself
                    ev = inspect(cell, fresh=True)
                    d = decide(ev)
                    if d not in ("register", "swap"):
                        return
                name, sc = ev["name"], ev["score"]
                tries[name] = tries.get(name, 0) + 1
                tap(cell_center(*cell), mod="ctrl")
                acted(cell)
                _park(io, geo)
                if not filled(cell):
                    count["register" if d == "register" else "swap"] += 1
                    record_item(base_dir, "등록" if d == "register" else "교체 등록", ev, tip_of.get(cell))
                    if sc is not None:
                        registry[name] = sc
                    log("%s %s" % (name, "등록" if d == "register" else "교체 등록 (%s점)" % sc))
                    return
                back = inspect(cell, fresh=True)     # something is in the cell: the previous copy came back
                if back and back["name"] == name and back["score"] is not None and sc is not None:
                    if back["score"] > sc:
                        tap(cell_center(*cell), mod="ctrl")      # the old one was better: put it back
                        acted(cell)
                        _park(io, geo)
                        registry[name] = back["score"]
                        log("%s: 기존 것이 더 좋아서 되돌림 (기존 %d점, 새것 %d점)" % (name, back["score"], sc))
                    elif back["score"] < sc:
                        registry[name] = sc
                        count["swap"] += 1
                        log("%s: 더 좋은 것으로 교체 (기존 %d점 → %d점)" % (name, back["score"], sc))
                    else:
                        registry[name] = sc
                        log("%s: 기존 것과 점수가 같음 (%d점)" % (name, sc))
                else:
                    log("%s: 등록을 눌렀지만 칸이 비지 않음 (그대로 둠)" % name)

            walk(collect)

    # 2) stash: essences to the personal tab, keepers to the chosen tab
    def stash_pass(tab_name, kinds):
        """-> how many items would not go in (the tab is full)."""
        if not open_stash(tab_name):
            log("보관함 %s 탭을 확인하지 못함" % tab_name)
            return 0
        refused = [0]

        def put(cell, ev, d):
            if d in kinds:
                tap(cell_center(*cell), mod="ctrl")
                acted(cell)
                _park(io, geo)
                if not filled(cell):
                    count["stash"] += 1
                    log("%s → 보관함 %s" % (title(ev) if ev else "읽지 못한 아이템", tab_name))
                    record_item(base_dir, "보관", ev, tip_of.get(cell), tab_name)
                else:
                    refused[0] += 1

        walk(put)
        return refused[0]

    def stash_all(first_tab, kinds):
        """Fill first_tab, then spill over into the tabs after it."""
        order = list(STASH_ORDER)
        start = order.index(first_tab) if first_tab in order else 0
        for tab in order[start:]:
            if stop.is_set():
                return
            left_over = stash_pass(tab, kinds)
            if not left_over:
                return
            log("보관함 %s 탭이 가득 참 (%d개 남음): 다음 탭에 넣음" % (tab, left_over))
        log("보관함 탭이 모두 가득 차서 일부를 넣지 못함")

    keep_tab = opts.get("stash_tab", "개인")
    if not stop.is_set() and any(plan.get(k) for k in stash_kinds):
        if keep_tab == "개인":
            stash_all("개인", ("essence", "keep", "leave"))
        else:
            stash_all("개인", ("essence", "leave"))
            stash_all(keep_tab, ("keep",))

    # 3) sell what is left and was positively identified as not worth keeping
    n5 = norm_now()
    left_now = n5 is not None and any(v for row in read_inventory(n5) for v in row)
    if not stop.is_set() and left_now:
        open_shop()
        n3 = norm_now()
        if n3 is None or seen(n3, "shop") < 0.7:
            log("찰시 판매 화면을 확인하지 못해 판매를 건너뜀")
        else:
            def sell(cell, ev, d):
                if d == "sell":
                    n_ = norm_now()             # last look right before selling: an essence is never sold
                    if n_ is None or essence_cell(n_, cell[0], cell[1], base_dir):
                        log("칸 %d,%d는 정수 그림이라 판매하지 않음" % (cell[0] + 1, cell[1] + 1))
                        return
                    tap(cell_center(*cell), mod="shift")
                    acted(cell)
                    _park(io, geo)
                    if not filled(cell):
                        count["sell"] += 1
                        log("%s 판매" % title(ev))

            walk(sell)
    # 4) something looked at again only now (it shared a colour block with another item) may be a keeper
    n6 = norm_now()
    if not stop.is_set() and n6 is not None and any(v for row in read_inventory(n6) for v in row):
        late = {}

        def check(cell, ev, d):
            late[d] = late.get(d, 0) + 1

        walk(check)
        if any(late.get(k) for k in ("essence", "keep")):
            stash_all("개인", ("essence", "keep"))
    try:
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=1)
    except OSError:
        pass
    n4 = norm_now()
    left = sum(1 for row in read_inventory(n4) for v in row if v) if n4 is not None else -1
    return "자동 정리 끝: 개별 등록 %d, 교체 %d, 보관 %d, 판매 %d, 남은 칸 %d" % (
        count["register"], count["swap"], count["stash"], count["sell"], left), True


def material_short(norm):
    """The material line turns red when there is not enough of it."""
    x0, y0, x1, y1 = FORGE_MAT_LINE
    roi = norm[y0:y1, x0:x1].astype(int)
    red = (roi[:, :, 2] > 150) & (roi[:, :, 1] < 90) & (roi[:, :, 0] < 90)
    return int(red.sum()) > 25


def enhance(io, count, base_dir, log, stop):
    """Press the anvil's check button up to `count` times while the screen says '강화 확률'."""
    frame = io.grab()
    if frame is None:
        return "게임 창을 찾지 못해 강화를 건너뜀"
    geo = Geo(frame)
    io.move(*geo.raw(*PARK))
    time.sleep(0.3)
    done = 0
    why = "지정한 횟수를 채움"
    for i in range(int(count)):
        if stop.is_set():
            why = "중지함"
            break
        frame = io.grab()
        if frame is None:
            why = "게임 창이 사라짐"
            break
        norm = geo.normalize(frame)
        if seen(norm, "forge") < 0.7:
            why = "모루 창이 보이지 않음"
            break
        if seen(norm, "forge_mode", text=True) < 0.6:
            why = "화면에 '강화 확률'이 보이지 않음 (다른 대장간 메뉴이거나 모루가 비어 있음)"
            break
        if material_short(norm):
            why = "재료가 부족함 (재료 수가 빨간색)"
            break
        io.click(*geo.raw(*FORGE_OK))
        io.move(*geo.raw(*PARK))
        time.sleep(1.0)
        done += 1
        after = io.grab()
        if after is not None:
            x0, y0 = geo.raw(FORGE_VIEW[0], FORGE_VIEW[1])
            x1, y1 = geo.raw(FORGE_VIEW[2], FORGE_VIEW[3])
            save_snap(base_dir, after[y0:y1, x0:x1], "강화 %d회째" % done)
    return "강화 끝: %d번 시도, 멈춘 이유: %s" % (done, why)


MENU_EXIT = (400, 380)                    # "저장 후 나가기", the last line of the Esc menu
MENU_PENTAS = ((259, 378), (542, 378))    # the red pentagrams beside it while the cursor is on it


MENU_ROWS = (172, 224, 276, 329, 381)      # the five lines of the Esc menu (the last: 저장 후 나가기)


def lit_row(norm):
    """Which menu line has the pentagrams beside it: 0..4, or None."""
    if norm is None:
        return None
    roi = norm[110:430].astype(np.int32)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    m = ((r > 150) & (g < 70) & (b < 70)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((5, 5), np.uint8)))
    blobs = [(cen[i][0], cen[i][1] + 110) for i in range(1, n) if 150 <= st[i][4] <= 2500
             and 15 <= st[i][2] <= 60 and 15 <= st[i][3] <= 60]
    for l in blobs:
        for r_ in blobs:
            if l[0] < 390 < 410 < r_[0] and abs(l[1] - r_[1]) < 10:
                yy = (l[1] + r_[1]) / 2.0
                i = min(range(len(MENU_ROWS)), key=lambda k: abs(MENU_ROWS[k] - yy))
                if abs(MENU_ROWS[i] - yy) < 20:
                    return i
    return None


def esc_menu_open(norm):
    """True when the Esc menu is up, whichever line is highlighted: a red pentagram on each side of one
    line (left and right of the middle, on the same height)."""
    if norm is None:
        return False
    if _again[0] is not None and run_ended(norm, None) >= 0.5:
        return False                              # the run-end screen has pentagrams too
    roi = norm[110:430].astype(np.int32)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    m = ((r > 150) & (g < 70) & (b < 70)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((5, 5), np.uint8)))
    blobs = [(cen[i][0], cen[i][1], st[i][4]) for i in range(1, n) if 150 <= st[i][4] <= 2500
             and 15 <= st[i][2] <= 60 and 15 <= st[i][3] <= 60]
    left = [bl for bl in blobs if 120 < bl[0] < 390]
    right = [bl for bl in blobs if 410 < bl[0] < 680]
    pair = any(abs(l[1] - r_[1]) < 10 and abs((l[0] + r_[0]) / 2.0 - 400) < 40 for l in left for r_ in right)
    return pair and lit_row(norm) is not None     # beside one of the menu's five lines


def out_of_stage(frame):
    """No run information at the top right ("N막 - 공포 영역 ...", kills): town, or the loading screen."""
    from . import spot
    h, w = frame.shape[:2]
    top = Geo(frame).top
    crop = frame[top: top + int(0.13 * (h - top)), int(0.60 * w):]
    return spot.area_mask(crop, (h - top) / 600.0).sum() < 1500


def _textmap(img):
    """Lettering brighter than what is around it: the same whatever the brightness and the scene behind."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g > cv2.blur(g, (31, 31)) * 1.4 + 4).astype(np.float32)


def menu_text(norm):
    """How strongly the Esc menu's lines ("게임으로 돌아가기", "저장 후 나가기") show (0..1)."""
    if norm is None:
        return 0.0
    best = 0.0
    for k in ("menu_exit", "menu_back"):
        if k not in _marks:
            continue
        best = max(best, _seen(k, norm, pad=30))
        img, x, y = _marks[k]
        h, w = img.shape[:2]
        y0, x0 = max(0, y - 40), max(0, x - 40)
        roi = norm[y0:y + h + 40, x0:x + w + 40]
        if roi.shape[0] >= h and roi.shape[1] >= w:
            v = cv2.minMaxLoc(cv2.matchTemplate(_textmap(roi), _textmap(img), cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(v):
                best = max(best, float(v) + 0.1)     # (the text map scores lower than a picture match)
    return best


def menu_open(norm):
    """The Esc menu is up: its lines, or a pair of red pentagrams beside one of its lines."""
    if norm is None:
        return False
    v = menu_text(norm)
    if v >= 0.55:
        return True
    return v >= 0.4 and esc_menu_open(norm)      # weaker lines + pentagrams beside a menu line


def leave_via_menu(io, log, open_menu, keys=None, attempt=1):
    """Esc menu -> "저장 후 나가기" -> (new game) "게임을 저장하고 나갈까요?" -> "저장 후 나가기".
    open_menu() presses Esc. keys: {"down", "up", "enter"} for io.key. Each attempt tries another way
    (keyboard / mouse), because a mouse click may land on the lit line when the game is behind other
    windows. Success only when the run information at the top right is gone (town / loading).
    -> True when the stage was left. On failure nothing is left open."""
    keys = keys or {}
    has_keys = bool(keys.get("down") and keys.get("enter"))

    def norm_now():
        f_ = io.grab()
        return (None, None) if f_ is None else (f_, Geo(f_).normalize(f_))

    def wait_left(sec):
        """-> "left" | "confirm" | "menu" | "game" after up to sec seconds."""
        t_end = time.time() + sec
        st_, gone = "game", 0
        while True:
            f_, n_ = norm_now()
            if n_ is not None:
                if confirm_shown(n_):
                    return "confirm"
                if menu_open(n_):
                    st_, gone = "menu", 0
                elif out_of_stage(f_):
                    gone += 1
                    if gone >= 2:                 # twice in a row: not just a fade
                        return "left"
                    st_ = "game"
                else:
                    st_, gone = "game", 0
            if time.time() >= t_end:
                return st_
            time.sleep(0.4)

    def press(seq):
        for k in seq:
            io.key(keys[k])
            time.sleep(0.2)

    def close_all():
        """Leave nothing open: Esc while a box or the menu is still showing."""
        for _ in range(3):
            f_, n_ = norm_now()
            if n_ is None or not (menu_open(n_) or confirm_shown(n_)):
                return
            open_menu()
            time.sleep(0.6)

    def menu_up(n_):
        return confirm_shown(n_) or menu_open(n_)

    def esc_and_look():
        open_menu()
        time.sleep(0.9)
        f_, n_ = norm_now()
        if n_ is not None and not menu_up(n_):
            time.sleep(0.8)                       # (it may open a little late)
            f_, n_ = norm_now()
        return f_, n_

    def blind_try(f_, round_):
        """The menu may be up without being recognised: click where its "저장 후 나가기" is anyway
        (with no menu that click only walks a step). -> wait_left() result"""
        if round_ == 2 and _snap_base[0]:
            try:
                save_snap(_snap_base[0], f_, "Esc 메뉴 인식 실패")
            except Exception:
                pass
        bx, by = Geo(f_).raw(*MENU_EXIT)
        io.move(bx, by)
        time.sleep(0.35)
        io.click(bx, by)
        time.sleep(0.3)
        io.click(bx, by)
        r_ = wait_left(3.5)
        log("Esc를 눌렀지만 메뉴가 보이지 않음: '저장 후 나가기' 자리를 그냥 클릭 (%d번째) -> %s%s" % (
            round_, {"left": "마을로 나감", "confirm": "확인 창", "menu": "메뉴가 보임", "game": "게임 그대로"}[r_],
            " (화면 저장함)" if round_ == 2 and _snap_base[0] else ""))
        return r_

    # a try after the first: the one before may have worked after all (town / loading), and there the
    # same menu line quits to the game's main menu. (Not on the first try: right after the boss the run
    # information may be missing from the top right too.)
    town = 0
    for _ in range(3 if attempt >= 2 else 0):
        f_, n_ = norm_now()
        if f_ is None:
            break
        covered = seen(n_, "panel") >= 0.8 or seen(n_, "inv") >= 0.7 or menu_open(n_) or confirm_shown(n_)
        if covered or not out_of_stage(f_):
            town = 0
            break
        town += 1
        time.sleep(0.35)
    if town >= 3:
        log("이미 사냥터 밖(마을)이라 Esc -> 저장 후 나가기를 하지 않음 (게임 첫 화면으로 빠지는 것 방지)")
        return True

    blind = False
    f, n = esc_and_look()
    if n is None:
        return False
    for round_ in (1, 2):
        if menu_up(n):
            if round_ == 2:
                log("Esc를 한 번 더 눌러 메뉴가 열림 (열린 창이 있었음)")
            break
        r = blind_try(f, round_)
        if r == "left":
            return True
        if r == "confirm":
            blind = True
            break
        if round_ == 1:
            # an open window (inventory, waypoint list, smith) took that Esc, or the menu was already up
            # and that Esc closed it: one more Esc
            f, n = esc_and_look()
            if n is None:
                return False
    else:
        close_all()
        return False
    geo = Geo(f)
    x, y = geo.raw(*MENU_EXIT)
    use_keys = has_keys and attempt % 2 == 1      # 1, 3, 5: keyboard / 2, 4, 6: mouse
    if not blind and not confirm_shown(n):
        if use_keys:
            row = lit_row(n)
            if row is not None:
                press(["down"] * (4 - row) + ["enter"])          # from the lit line down to the last one
            else:
                press(["down"] * 4 + ["enter"] if attempt < 3 else ["up", "enter"])   # down 4 / up 1 (wrap)
            how = "키보드"
        else:
            io.move(x, y)
            time.sleep(0.35)
            io.click(x, y)
            time.sleep(0.3)
            io.click(x, y)                        # a first click may only move the highlight
            how = "클릭"
        r = wait_left(3.5)
        log("Esc 메뉴 '저장 후 나가기' (%s, %d번째) -> %s" % (how, attempt, {"left": "마을로 나감", "confirm": "확인 창",
                                                                     "menu": "메뉴 그대로", "game": "게임으로 돌아감"}[r]))
        if r == "left":
            return True
        if r != "confirm":
            close_all()
            return False
    # "게임을 저장하고 나갈까요?": its lower button
    cx, cy = geo.raw(*CONFIRM_EXIT)
    tries = [("click", None)]
    if has_keys:
        tries += [("keys", ["down", "enter"]), ("keys", ["tab", "enter"] if keys.get("tab") else ["enter"])]
    for how, seq in tries:
        if how == "click":
            io.move(cx, cy)
            time.sleep(0.3)
            io.click(cx, cy)
        else:
            press(seq)
        r = wait_left(3.5)
        log("확인 창 '저장 후 나가기' (%s) -> %s" % ("클릭" if how == "click" else "+".join(seq),
                                             {"left": "마을로 나감", "confirm": "창 그대로", "menu": "메뉴로 돌아감",
                                              "game": "게임으로 돌아감"}[r]))
        if r == "left":
            return True
        if r != "confirm":
            close_all()
            return False
    close_all()
    return False


CONFIRM_EXIT = (400, 375)                 # "저장 후 나가기" button of the "게임을 저장하고 나갈까요?" box


def confirm_shown(norm):
    return norm is not None and "confirm" in _marks and _seen("confirm", norm, pad=30) >= 0.7


def exit_menu_shown(norm):
    """True when the Esc menu is open with "저장 후 나가기" highlighted (red pentagram on each side)."""
    def red(x, y, r=14):
        roi = norm[y - r:y + r, x - r:x + r].astype(np.int32)
        b, g, rr = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        return float(((rr > 150) & (g < 70) & (b < 70)).mean())
    return all(red(x, y) >= 0.22 for x, y in MENU_PENTAS)
