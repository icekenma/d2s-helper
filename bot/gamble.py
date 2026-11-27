# -*- coding: utf-8 -*-
"""도박 (gambling at the town NPC) until a wanted item comes out: buy the arrow quiver shown in the gamble
list, read the bought one, keep it and stop when it has the wanted line ('+3 아마존 기술'), else sell it
straight back (Shift+click in the shop window) and buy the next. '리롤' when no quiver is listed.

Needs the gamble window open in town (the user opens it) and the game window in front (info boxes)."""
import os
import re
import time

import cv2
import numpy as np

from . import namer, scorer, sorter, tipread

TEMPLATES = {"quiver": "gamble_quiver.png", "reroll": "gamble_reroll.png", "title": "gamble_title.png"}
_tpl = {}
WANT_N = 3                   # '+3 아마존 기술' (or more)
SKILL_NONE = 0.45            # '기술' picture under this on every line: no skill line at all (no OCR needed)
SKILL_LINE = 0.6             # a line with '기술' on it (skill lines 0.78-0.80, others up to 0.49)


def _load(base_dir):
    if not _tpl:
        for k, f in TEMPLATES.items():
            t = cv2.imread(os.path.join(base_dir, "assets", f))
            if t is not None:
                _tpl[k] = t
    return _tpl


def _find(norm, name, area, need):
    """Best match of a template inside area (x0, y0, x1, y1 in the 800x600 layout). -> (cx, cy, score) or None."""
    t = _tpl.get(name)
    if t is None:
        return None
    x0, y0, x1, y1 = area
    roi = norm[y0:y1, x0:x1]
    if roi.shape[0] < t.shape[0] or roi.shape[1] < t.shape[1]:
        return None
    r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
    _, v, _, p = cv2.minMaxLoc(r)
    if v < need:
        return None
    return x0 + p[0] + t.shape[1] / 2.0, y0 + p[1] + t.shape[0] / 2.0, v


def gamble_open(norm):
    return _find(norm, "title", (0, 0, 420, 160), 0.8) is not None or _find(norm, "reroll", (0, 380, 420, 560), 0.8) is not None


def judge(tip, s):
    """-> ("hit" | "no" | "unsure", why). The words come from the Windows OCR, the level from the game's own
    digits (the OCR reads the game's '+3' as '하' or '*3'): the n-th OCR line with '기술' is the n-th line
    whose '기술' picture scores high."""
    lines, img = tipread.read(tip, s)
    mask = tipread.binarize(img).astype(np.float32)
    sk_lines = []                            # (score, first number) of lines that may hold '기술', top down
    best = 0.0
    for l in lines:
        if l.get("color") in ("blue", "yellow", "orange", "other"):
            sc = scorer.skill_score(mask[max(0, l["y0"] - 2):l["y1"] + 2])
            best = max(best, sc)
            if sc >= SKILL_LINE:
                sk_lines.append((sc, (l.get("numbers") or [None])[0]))
    if best < SKILL_NONE:
        return "no", "기술 줄 없음 (그림 %.2f)" % best
    got = namer.ocr_now([tip])
    if got is None:
        return "unsure", "윈도우 한국어 글자 인식(OCR)을 쓸 수 없음: %s" % (namer.LAST_ERR[0] or "원인 모름")
    texts = got[0] if got else []
    if not texts:
        return "unsure", "정보창 글자를 못 읽음"
    ocr_sk = [t for t in texts if "기술" in t.replace(" ", "")]
    for k, t in enumerate(ocr_sk):
        u = t.replace(" ", "")
        if "아마존기술" not in u or "전용" in u:
            continue                          # a skill tab ('활과 석궁 기술 (아마존 전용)') or another class
        n = None
        if len(ocr_sk) == len(sk_lines):
            n = sk_lines[k][1]
        elif len(sk_lines) == 1:
            n = sk_lines[0][1]
        if n is None:
            m = re.search(r"(\d+)아마존기술", u)
            n = int(m.group(1)[-1]) if m else None
        if n is None:
            return "unsure", "'아마존 기술' 줄의 숫자를 못 읽음: %s (기술 줄 %d개 / 글자 %d줄)" % (t, len(sk_lines), len(ocr_sk))
        if n >= WANT_N:
            return "hit", "+%d 아마존 기술 (%s)" % (n, t)
        return "no", "+%d 아마존 기술" % n
    return "no", "아마존 기술 없음 (%s)" % " / ".join(ocr_sk[:3] or texts[:3])


def run(io, base_dir, log, stop, status, notify):
    """Gamble until the wanted quiver comes, the window closes, or stop is set."""
    _load(base_dir)
    if "quiver" not in _tpl:
        log("도박: 화살통 그림이 없음 (assets/gamble_quiver.png)")
        return
    n = rerolls = fails = 0
    t0 = time.time()
    status.update(on=True, n=0, msg="도박 시작")
    try:
        f = io.grab()
        if f is not None and not io.front():
            from . import win
            if win.focus(io.hwnd):
                time.sleep(0.5)
        while not stop.is_set():
            f = io.grab()
            if f is None:
                log("도박: 게임 창을 찾지 못함 - 멈춤")
                return
            geo = sorter.Geo(f)
            norm = geo.normalize(f)
            if not gamble_open(norm):
                log("도박: 도박 창이 열려 있지 않음 - 멈춤 (마을 도박 NPC의 '도박' 탭을 연 뒤 시작)")
                return
            q = _find(norm, "quiver", (0, 60, 420, 470), 0.8)
            if q is None:
                r = _find(norm, "reroll", (0, 380, 420, 560), 0.7)
                if r is None:
                    log("도박: 목록에 화살통이 없고 '리롤' 버튼도 못 찾음 - 멈춤")
                    return
                io.click(*geo.raw(r[0], r[1]))
                rerolls += 1
                time.sleep(0.6)
                continue
            before = sorter.read_inventory(norm)
            io.click(*geo.raw(q[0], q[1]))
            time.sleep(0.5)
            sorter._park(io, geo)
            base = io.grab()
            if base is None:
                return
            nb = geo.normalize(base)
            after = sorter.read_inventory(nb)
            new = [(c, r_) for r_ in range(sorter.INV_ROWS) for c in range(sorter.INV_COLS)
                   if after[r_][c] and not before[r_][c]]
            if not new:
                fails += 1
                if fails >= 3:
                    msg = "도박: 3번 연속 아이템이 안 들어옴 (골드 부족이나 인벤토리에 세로 3칸 자리 없음) - 멈춤"
                    log(msg)
                    notify("도박 멈춤", msg)
                    return
                continue
            fails = 0
            n += 1
            c0 = min(c for c, _ in new)
            r0 = min(r_ for c, r_ in new if c == c0)
            io.move(*geo.raw(*sorter.cell_center(c0, r0)))
            time.sleep(0.25)
            hov = io.grab()
            tip = sorter.tooltip_crop(base, hov, geo) if hov is not None else None
            if tip is None:
                verdict, why = "unsure", "정보창이 안 뜸 (게임 창이 뒤에 있으면 앞으로 가져와 주세요)"
            else:
                verdict, why = judge(tip, geo.s)
            status.update(n=n, msg=why, rerolls=rerolls)
            if verdict == "hit":
                msg = "도박 %d번째에 원하는 화살통이 나옴: %s (인벤토리 %d열 %d줄, %.0f분)" % (n, why, c0 + 1, r0 + 1,
                                                                                (time.time() - t0) / 60)
                log(msg)
                notify("도박 성공", msg)
                status.update(msg="성공: " + why)
                return
            if verdict == "unsure":
                msg = "도박 %d번째: 판단이 불확실해 팔지 않고 멈춤 - %s (인벤토리 %d열 %d줄 확인)" % (n, why, c0 + 1, r0 + 1)
                log(msg)
                notify("도박 확인 필요", msg)
                return
            sorter._park(io, geo)
            io.click(*geo.raw(*sorter.cell_center(c0, r0)), mod="shift")       # sold back to the NPC
            time.sleep(0.4)
            f = io.grab()
            if f is None or sorter.read_inventory(geo.normalize(f))[r0][c0]:
                msg = "도박 %d번째: Shift+클릭으로 팔리지 않음 - 멈춤 (인벤토리 %d열 %d줄)" % (n, c0 + 1, r0 + 1)
                log(msg)
                notify("도박 멈춤", msg)
                return
            if n % 10 == 0 or n <= 3:
                log("도박 %d번째 (리롤 %d번): %s -> 판매" % (n, rerolls, why))
        log("도박: 멈춤 (%d번 시도, 리롤 %d번)" % (n, rerolls))
    finally:
        status.update(on=False)
