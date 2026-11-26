# -*- coding: utf-8 -*-
"""Recognises a handful of fixed words by their picture (the game font is a bitmap font, so a word
always looks the same). Needs no OCR program."""
import json
import os

import cv2
import numpy as np

from . import tipread

_words = {}          # name -> list of float32 masks at work scale


def load(base):
    _words.clear()
    p = os.path.join(base, "assets", "words.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for name, rows in json.load(f):
                _words.setdefault(name, []).append(np.array([[int(c) for c in r] for r in rows], np.float32))


def label_mask(crop, s):
    """Text mask of a grey world label, at work scale. Labels pick up a little of the scene's light
    (reddish near a portal), so "grey" is taken loosely."""
    v = crop.astype(np.int32)
    keep = ((v.max(axis=2) - v.min(axis=2) <= 34) & (v.max(axis=2) >= 112) & (v.max(axis=2) <= 200)).astype(np.uint8) * 255
    f = tipread.SCALE / float(s)
    big = cv2.resize(keep, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return (big > 127).astype(np.float32)


def tip_mask(crop, s):
    """Text mask of an info box (any text colour), at work scale."""
    return tipread.binarize(tipread.to_work(crop, s)).astype(np.float32)


def score(mask, name):
    """Best match of the word anywhere in the mask, 0..1."""
    best = 0.0
    for t in _words.get(name, ()):
        th, tw = t.shape
        m = mask
        if m.shape[0] < th or m.shape[1] < tw:
            pad_y, pad_x = max(0, th - m.shape[0]), max(0, tw - m.shape[1])
            m = cv2.copyMakeBorder(m, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
        if m.std() == 0:
            continue
        v = cv2.minMaxLoc(cv2.matchTemplate(m, t, cv2.TM_CCOEFF_NORMED))[1]
        if np.isfinite(v) and v > best:
            best = float(v)
    return best


def label_score(crop, s, name, spread=(0.9, 0.95, 1.0, 1.05, 1.1)):
    """score() of a grey world label tried at a few scales round s: the screen's scale is worked out from
    the window, and a few percent off is enough to lose a small word."""
    return max(score(label_mask(crop, s * k), name) for k in spread)


def has(mask, name, need=0.72):
    return score(mask, name) >= need


def width(name):
    ws = [t.shape[1] for t in _words.get(name, ())]
    return max(ws) if ws else 0


def search(frame, s, names, need=0.7, any_colour=False, alone=True, every=False, max_hits=8):
    """Look for whole name labels anywhere in the game view. -> [(name, score, centre x, bottom y)] in
    frame pixels, best first. Does not depend on the label having been boxed first. Works at half
    resolution to stay quick; a hit is then checked at full resolution."""
    h, w = frame.shape[:2]
    v = frame.astype(np.int16)
    mx = v.max(axis=2)
    grey = (mx - v.min(axis=2) <= 34) & (mx >= 112) & (mx <= 200)
    if any_colour:
        # item labels also come in gold and orange; their shade follows the scene's light
        b, g, r = v[:, :, 0].astype(np.float32), v[:, :, 1].astype(np.float32), np.maximum(v[:, :, 2], 1).astype(np.float32)
        gold = (r >= 90) & (g / r > 0.72) & (g / r < 1.0) & (b / r > 0.30) & (b / r < 0.78)
        orange = (r >= 140) & (g / r > 0.42) & (g / r <= 0.72) & (b / r < 0.4)
        grey = grey | gold | orange | ((mx - v.min(axis=2) <= 34) & (mx > 200))
    keep = grey.astype(np.uint8) * 255
    keep[int(h - 115 * s):, :] = 0                       # the bar along the bottom
    keep[: int(0.26 * h), int(0.80 * w):] = 0            # run info in the top right corner
    f = tipread.SCALE / float(s)
    half = cv2.resize(keep, None, fx=f / 2, fy=f / 2, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out = []
    for n in names:
        best = None
        for t in _words.get(n, ()):
            th, tw = t.shape
            small = cv2.resize(t, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
            if half.shape[0] < small.shape[0] or half.shape[1] < small.shape[1] or small.std() == 0:
                continue
            res = cv2.matchTemplate(half, small, cv2.TM_CCOEFF_NORMED)
            # an empty window has no pattern to compare (the score there is meaningless): rule those out
            ink = cv2.boxFilter(half, -1, (small.shape[1], small.shape[0]), normalize=False,
                                anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)[:res.shape[0], :res.shape[1]]
            res[~np.isfinite(res)] = 0
            res[ink < 0.4 * float(small.sum())] = 0
            for _round in range(max_hits if every else 1):
                _, v0, _, loc = cv2.minMaxLoc(res)
                if v0 < need - 0.2:
                    break
                # look no further near this spot next round
                sx0, sy0 = max(0, loc[0] - small.shape[1]), max(0, loc[1] - small.shape[0] // 2)
                res[sy0:loc[1] + small.shape[0] // 2 + 1, sx0:loc[0] + small.shape[1] + 1] = 0
                hit = _confirm(keep, t, loc, f, tw, th, need, alone, n)
                if hit is None:
                    continue
                if every:
                    out.append(hit)
                elif best is None or hit[1] > best[1]:
                    best = hit
        if best:
            out.append(best)
    out.sort(key=lambda b: -b[1])
    return out


def _confirm(keep, t, loc, f, tw, th, need, alone, n):
    """Check a coarse hit at full resolution. -> (name, score, centre x, bottom y) or None."""
    # confirm at full resolution around the coarse hit, with room on both sides
    gap = 16
    x0 = int(loc[0] * 2 / f) - int((gap + 6) / f)
    y0 = int(loc[1] * 2 / f) - int(6 / f)
    x1 = x0 + int((tw + 2 * gap + 12) / f)
    y1 = y0 + int((th + 12) / f)
    x0, y0 = max(0, x0), max(0, y0)
    patch = cv2.resize(keep[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    patch = (patch > 127).astype(np.float32)
    if patch.shape[0] < th or patch.shape[1] < tw or patch.std() == 0:
        return None
    _, v1, _, l1 = cv2.minMaxLoc(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED))
    if not np.isfinite(v1) or v1 < need:
        return None
    # a name standing alone: nothing written right before or after it ("라주크" vs "라주크의 정수")
    rows = patch[l1[1] + 3:l1[1] + th - 3]
    before = rows[:, max(0, l1[0] - 12):max(0, l1[0] - 3)].sum()
    after = rows[:, l1[0] + tw + 3:l1[0] + tw + 12].sum()
    if alone and (before > 6 or after > 6):
        return None
    return (n, float(v1), x0 + (l1[0] + tw / 2.0) / f, y0 + (l1[1] + th) / f)


def area_mask(crop, s, gold=False):
    """Text mask of light grey / white lettering (gold too with gold=True), at work scale."""
    v = crop.astype(np.int32)
    keep = (v.max(axis=2) - v.min(axis=2) <= 40) & (v.max(axis=2) >= 110)
    if gold:
        b, g, r = v[:, :, 0].astype(np.float32), v[:, :, 1].astype(np.float32), np.maximum(v[:, :, 2], 1).astype(np.float32)
        keep = keep | ((r >= 110) & (g / r > 0.62) & (g / r < 1.0) & (b / r < 0.75))
    keep = keep.astype(np.uint8) * 255
    f = tipread.SCALE / float(s)
    big = cv2.resize(keep, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return (big > 127).astype(np.float32)


def terror_zone(frame, s):
    """How strongly the top right corner reads '공포 영역' (the run info shown only in a terror zone)."""
    h, w = frame.shape[:2]
    crop = frame[: int(0.13 * h), int(0.60 * w):]
    return score(area_mask(crop, s), "공포영역")
