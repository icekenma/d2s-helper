# -*- coding: utf-8 -*-
"""Reads an item info box: line colours, and numbers via glyph templates of the game font."""
import json
import os

import cv2
import numpy as np

SCALE = 2.0          # work resolution, in pixels per 800x600 layout unit
GW, GH = 12, 18      # glyph template size
_glyphs = []         # (label, float32 GHxGW)


def load_glyphs(base):
    p = os.path.join(base, "assets", "glyphs.json")
    _glyphs[:] = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for lab, rows in json.load(f):
                _glyphs.append((lab, np.array([[int(c) for c in r] for r in rows], np.float32)))


def to_work(crop, s):
    f = SCALE / float(s)
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR if f >= 1 else cv2.INTER_AREA)


def binarize(img):
    return (img.max(axis=2) > 105).astype(np.uint8)


def line_bands(mask):
    rows = mask.sum(axis=1) >= 4
    bands, start = [], None
    for y, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= 8:
                bands.append((start, y))
            start = None
    merged = []
    for b in bands:                       # join pieces of one text line split by a blank row
        if merged and b[0] - merged[-1][1] <= 2 and b[1] - merged[-1][0] <= 30:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged


def segments(mask, y0, y1):
    """Glyph column runs of one line, grouped into words: [[(x0, x1), ...], ...]."""
    cols = mask[y0:y1].sum(axis=0) > 0
    runs, start = [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 3 or mask[y0:y1, start:x].sum() >= 8:
                runs.append((start, x))
            start = None
    words = []
    for r in runs:
        if words and r[0] - words[-1][-1][1] < 9:
            words[-1].append(r)
        else:
            words.append([r])
    return words


def glyph_img(mask, y0, y1, x0, x1):
    g = mask[y0:y1, x0:x1]
    ys = np.where(g.sum(axis=1) > 0)[0]
    if len(ys) == 0:
        return None, 0, 0
    g = g[ys[0]:ys[-1] + 1]
    return cv2.resize(g.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA), g.shape[1], g.shape[0]


def classify(gi):
    best, lab = 0.0, ""
    a = gi - gi.mean()
    na = np.sqrt((a * a).sum())
    if na == 0:
        return "", 0.0
    for l, t in _glyphs:
        b = t - t.mean()
        v = float((a * b).sum() / (na * np.sqrt((b * b).sum()) + 1e-9))
        if v > best:
            best, lab = v, l
    return lab, best


def line_color(img, mask, y0, y1):
    px = img[y0:y1][mask[y0:y1] > 0].astype(float)
    if len(px) == 0:
        return "none"
    b, g, r = np.median(px, axis=0)
    mx, mn = max(b, g, r), min(b, g, r)
    if mx - mn < 25:
        return "white"
    if b > 70 and r > 0.5 * b and r > 1.3 * g and b > 1.3 * g:
        return "purple"                     # '수집품에 없음': a unique / set not in the collection yet
    if b == mx and b > 1.6 * r:
        return "blue"
    if r == mx and r > 2 * g:
        return "red"
    if g == mx and g > 1.5 * r:
        return "green"
    if r >= g and g / r > 0.78 and b / r < 0.75:
        return "yellow" if b / r < 0.5 else "gold"
    if r == mx and r > 150 and 0.45 <= g / r <= 0.75 and b / r < 0.35:
        return "orange"                     # runes (and crafted items)
    return "other"


def read(crop, s):
    """-> list of lines: {"color", "y0", "y1", "numbers": [int, ...], "box": (x0, x1)} in work pixels,
    plus the work image itself."""
    img = to_work(crop, s)
    mask = binarize(img)
    m = 10                                # the crop's rim still shows the screen behind the box
    mask[:m, :] = 0
    mask[-m:, :] = 0
    mask[:, :m] = 0
    mask[:, -m:] = 0
    out = []
    for y0, y1 in line_bands(mask):
        nums = []
        label = False
        unsure = False
        words = segments(mask, y0, y1)
        for w in words:
            txt = ""
            for x0, x1 in w:
                gi, gw, gh = glyph_img(mask, y0, y1, x0, x1)
                if gi is None:
                    txt += "?"
                    continue
                if gh <= 13:                # dots, dashes, plus signs
                    txt += "."
                    continue
                if gw <= 9:                 # a colon is two dots: nothing in its middle rows
                    col = mask[y0:y1, x0:x1]
                    rows = np.where(col.sum(axis=1) > 0)[0]
                    mid = col[rows[0] + (rows[-1] - rows[0]) // 2 - 1: rows[0] + (rows[-1] - rows[0]) // 2 + 2]
                    if mid.sum() == 0:
                        label = True
                        txt += ":"
                        continue
                if gw <= 8 and gh >= 14:      # a plain upright bar: the font's "1" without foot (수량: 1)
                    col = mask[y0:y1, x0:x1]
                    rows = np.where(col.sum(axis=1) > 0)[0]
                    if len(rows) and (col[rows[0]:rows[-1] + 1].sum(axis=1) > 0).mean() > 0.9:
                        txt += "1"
                        continue
                lab, sc = classify(gi)
                txt += lab if sc >= 0.70 else "?"
            if ":" in txt:                  # "수량:1" read as one word (a narrow space after the colon)
                txt = txt.split(":")[-1]
            core = txt.strip(".").rstrip("%")
            if "?" in txt and len(txt) >= 2 and core.replace("?", "").isdigit() and core.replace("?", ""):
                unsure = True
            if core and core.isdigit():
                nums.append(int(core))
        xs = [x for w in words for r in w for x in r]
        out.append({"color": line_color(img, mask, y0, y1), "y0": y0, "y1": y1, "numbers": nums,
                    "box": (min(xs), max(xs)) if xs else (0, 0), "words": len(words),
                    "label": label, "unsure": unsure})
    # very dim grey notes (e.g. "already stored") sit under the normal brightness cut
    dim = ((img.max(axis=2) > 50) & (img.max(axis=2) <= 105)).astype(np.uint8)
    dim[:m, :] = 0
    dim[-m:, :] = 0
    dim[:, :m] = 0
    dim[:, -m:] = 0
    for y0, y1 in line_bands(dim):
        if any(not (y1 <= o["y0"] or y0 >= o["y1"]) for o in out) or y1 - y0 < 14:
            continue
        xs = np.where(dim[y0:y1].sum(axis=0) > 0)[0]
        if len(xs) < 40:
            continue
        out.append({"color": "dim", "y0": y0, "y1": y1, "numbers": [], "box": (int(xs[0]), int(xs[-1])),
                    "words": 3, "label": False, "unsure": False})
    out.sort(key=lambda l: l["y0"])
    return out, img
