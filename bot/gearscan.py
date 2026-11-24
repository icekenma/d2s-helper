# -*- coding: utf-8 -*-
"""장비 스캔: with the smith's window open, the cursor rests on every worn item, every item in the bags
and in the stash tabs, and the info box of each is saved as a picture (nothing is clicked on an item,
nothing is moved). The pictures go into one ZIP to send for a best-gear recommendation."""
import os
import time
import zipfile

import cv2
import numpy as np

from . import paths, sorter

# worn items on the right-hand panel (800x600 layout)
EQUIP = [("투구", (561, 93)), ("목걸이", (620, 106)), ("무기", (447, 159)), ("갑옷", (561, 178)),
         ("방패_화살통", (678, 159)), ("장갑", (447, 265)), ("반지1", (506, 249)), ("허리띠", (561, 249)),
         ("반지2", (620, 249)), ("신발", (678, 265))]
STASH_X, STASH_Y, STASH_COLS, STASH_ROWS = 95, 65, 10, 12
# the 고유 (collection) tab: kind icons (13 + 12) and the card list (3 columns, scrolls)
UNIQ_CATS = [(108 + 22.3 * i, 75) for i in range(13)] + [(108 + 22.3 * i, 101) for i in range(12)]
UNIQ_LIST = (94, 136, 373, 412)            # x0, y0, x1, y1
UNIQ_COLS = (140, 235, 328)
UNIQ_LIST_CENTER = (235, 270)
SCAN_TABS = sorter.STASH_ORDER


# the skills picked in a run: icons at the top left (5 x 5), shown only while a run is on
SKILL_CELLS = [(174 + 25 * c, 17 + 25 * r) for r in range(5) for c in range(5)]
PENDING = {"dir": None}          # a scan waiting for the next run's skills (set after a scan in town)
_PFILE = paths.p("pending_skills.txt")
try:
    with open(_PFILE, "r", encoding="utf-8") as _f:
        PENDING["dir"] = _f.read().strip() or None      # (kept across a restart)
except OSError:
    pass


def _set_pending(d):
    PENDING["dir"] = d
    try:
        if d:
            with open(_PFILE, "w", encoding="utf-8") as f:
                f.write(d)
        elif os.path.exists(_PFILE):
            os.remove(_PFILE)
    except OSError:
        pass


def scan_skills(io, log):
    """During a run (called once after the boss dies): the info box of every skill icon, added to the last
    scan's folder and ZIP."""
    out = PENDING.get("dir")
    _set_pending(None)
    if not out or not os.path.isdir(out):
        return
    f = io.grab()
    if f is None:
        return
    geo = sorter.Geo(f)
    prev_fg = None
    if not io.front():
        # info boxes only show with the game in front: for these few seconds only
        try:
            import ctypes
            from . import win
            prev_fg = ctypes.windll.user32.GetForegroundWindow()
            if not win.focus(io.hwnd):
                prev_fg = None
            time.sleep(0.4)
        except Exception:
            prev_fg = None
    io.move(*geo.raw(400, 300))
    time.sleep(0.3)
    base = io.grab()
    if base is None:
        return
    _save(os.path.join(out, "000_화면_스킬.jpg"), base)
    tips, got = [], 0
    for i, (x, y) in enumerate(SKILL_CELLS):
        tip = None
        for k in range(3):
            io.move(*geo.raw(x + (k % 2), y))
            time.sleep(0.12 + 0.08 * k)
            hov = io.grab()
            if hov is not None:
                tip = sorter.tooltip_crop(base, hov, geo)
                if tip is not None:
                    break
        if tip is None or any(sorter.same_tip(tip, t) for t in tips):
            continue
        tips.append(tip)
        got += 1
        _save(os.path.join(out, "스킬_%02d.jpg" % (i + 1)), tip)
    io.move(*geo.raw(400, 300))
    if prev_fg:
        try:
            from . import win
            win.focus(prev_fg)
        except Exception:
            pass
    with zipfile.ZipFile(out + ".zip", "a", zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for fn in sorted(os.listdir(out)):
            if fn not in have:
                z.write(os.path.join(out, fn), fn)
    log("장비 스캔: 이번 판 스킬 %d개 정보창을 스캔 결과에 추가함" % got)


def _list_gray(norm):
    x0, y0, x1, y1 = UNIQ_LIST
    return cv2.cvtColor(norm[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)


def _thumb(norm):
    """(top, bottom) of the card list's scroll bar thumb, or None."""
    col = cv2.cvtColor(norm[143:402, 378:384], cv2.COLOR_BGR2GRAY).astype(np.float32).mean(1)
    on = col > 60
    best, start = None, None
    for i, v in enumerate(list(on) + [False]):        # the longest bright run (not the arrows)
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None or best[1] - best[0] < 4:
        return None
    return 143 + best[0], 143 + best[1]


def _shift(prev, cur):
    """How far the list went up between two pictures of it: where the bottom strip of the earlier one
    is in the later one. None when it is gone (moved a whole page or more)."""
    h = prev.shape[0]
    strip = prev[h - 50:h]
    if strip.std() < 3:
        strip = prev[h - 100:h - 50]
        base_y = h - 100
    else:
        base_y = h - 50
    r = cv2.matchTemplate(cur, strip, cv2.TM_CCOEFF_NORMED)
    _, v, _, loc = cv2.minMaxLoc(r)
    if not np.isfinite(v) or v < 0.8:
        return None
    d = int(base_y - loc[1])
    if d < 0 or d >= h - 10:
        return None
    # check the whole overlap, not just the strip (cards look alike: a strip can match the wrong row)
    if d > 0 and float(np.abs(prev[d:] - cur[:h - d]).mean()) > 6.0:
        return None
    return float(d)


def _blank(norm, x, y):
    """An empty spot of the card list (below the last card): dark and flat."""
    p = norm[int(y) - 12:int(y) + 12, int(x) - 30:int(x) + 30].astype(np.float32)
    return p.size == 0 or (p.mean() < 22 and p.std() < 9)


def _color_at(norm, x, y, cell):
    """Same test as the bags' cells, for a cell whose top-left corner is x, y."""
    ox, oy = sorter.INV_X, sorter.INV_Y
    try:
        sorter.INV_X, sorter.INV_Y = x, y
        return sorter.cell_color(norm, 0, 0)
    finally:
        sorter.INV_X, sorter.INV_Y = ox, oy


def _save(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if ok:
        buf.tofile(path)


def scan(io, base_dir, log, stop=None):
    """-> path of the ZIP, or None."""
    f = io.grab()
    if f is None:
        log("장비 스캔: 게임 화면을 읽지 못함")
        return None
    geo = sorter.Geo(f)
    n0 = geo.normalize(f)
    if sorter.seen(n0, "panel") < 0.6 and not sorter.stash_open(n0):
        log("장비 스캔: 대장장이 창(또는 보관함)이 열려 있지 않음. 마을에서 연 뒤 다시 누르세요")
        return None
    out = os.path.join(base_dir, "gear_scan", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    tips = []
    count = [0]

    def halted():
        return stop is not None and stop.is_set()

    def park():
        io.move(*geo.raw(*sorter.PARK))
        time.sleep(0.25)
        return io.grab()

    def hover(pt, base, tries=6):
        """Rest the cursor on pt until the info box shows (it can take a moment; with the game behind other
        windows the box may flicker, so the cursor is nudged and the screen read again)."""
        if base is None:
            return None
        x, y = geo.raw(*pt)
        for i in range(tries):
            io.move(x + (i % 2), y)
            time.sleep(0.06 + 0.05 * i)
            hov = io.grab()
            if hov is None:
                continue
            tip = sorter.tooltip_crop(base, hov, geo)
            if tip is not None:
                return tip
        return None

    def keep(tip, name):
        for t in tips:
            if sorter.same_tip(tip, t):
                return False                   # another cell of an item already saved
        tips.append(tip)
        count[0] += 1
        _save(os.path.join(out, "%03d_%s.jpg" % (count[0], name)), tip)
        return True

    # 1) worn items
    def worn_pass():
        b = park()
        _save(os.path.join(out, "000_화면_착용.jpg"), b)
        k = 0
        for name, pt in EQUIP:
            if halted():
                break
            tip = hover(pt, b)
            if tip is not None and keep(tip, "착용_" + name):
                k += 1
        return k

    prev_fg = None
    worn = worn_pass()
    if worn == 0 and not halted() and not io.front():
        # with the game behind other windows its info boxes do not show: bring it forward for the scan only
        try:
            import ctypes
            from . import win
            prev_fg = ctypes.windll.user32.GetForegroundWindow()
            if win.focus(io.hwnd):
                log("장비 스캔: 게임 창이 뒤에 있으면 정보창이 안 떠서, 스캔하는 동안만 게임 창을 앞으로 가져옴")
                time.sleep(0.5)
                worn = worn_pass()
            else:
                prev_fg = None
        except Exception:
            prev_fg = None
    log("장비 스캔: 착용 %d개" % worn)

    # 2) the bags
    base = park()
    nb = geo.normalize(base)
    bag = 0
    for r in range(sorter.INV_ROWS):
        for c in range(sorter.INV_COLS):
            if halted() or not sorter.cell_color(nb, c, r):
                continue
            tip = hover(sorter.cell_center(c, r), base)
            if tip is not None and keep(tip, "인벤_%d_%d" % (c + 1, r + 1)):
                bag += 1
    log("장비 스캔: 인벤토리 %d개" % bag)

    def scan_unique():
        n_ = geo.normalize(io.grab())
        if not sorter.stash_open(n_):
            io.click(*geo.raw(*sorter.ui_tabs(n_)[1]))
            time.sleep(0.5)
            park()
            n_ = geo.normalize(io.grab())
        tabs = sorter.stash_tabs(n_)
        io.click(*geo.raw(*tabs["고유"]))
        time.sleep(0.7)
        b = park()
        _save(os.path.join(out, "000_화면_보관함_고유.jpg"), b)
        total = 0
        lx, ly = UNIQ_LIST_CENTER
        for ci, (cx, cy) in enumerate(UNIQ_CATS):
            if halted():
                return
            io.click(*geo.raw(cx, cy))
            time.sleep(0.6)
            for _ in range(4):                       # to the top of the list
                io.scroll(*geo.raw(lx, ly), 15)
                time.sleep(0.12)
            time.sleep(0.3)
            got, page, new_from = 0, 0, UNIQ_LIST[1]
            H = UNIQ_LIST[3] - UNIQ_LIST[1]
            b = park()
            while page < 120 and not halted():
                nb = geo.normalize(b)
                # the list itself too (names and which are collected), page by page
                _save(os.path.join(out, "고유_%02d_목록_%02d.jpg" % (ci + 1, page + 1)), nb[40:420, 88:392])
                # rest the cursor on the cards that came into view (with a little overlap)
                y_start = max(UNIQ_LIST[1] + 12, new_from - 30)
                for yy in range(int(y_start), UNIQ_LIST[3] - 8, 45):
                    for xx in UNIQ_COLS:
                        if halted():
                            return
                        if _blank(nb, xx, yy):
                            continue                     # nothing there (below the last card)
                        tip = hover((xx, yy), b, tries=3)
                        if tip is not None and keep(tip, "고유_%02d" % (ci + 1)):
                            got += 1
                # next: one wheel notch at a time, measured each time, until about half a page has come in
                moved, end = 0.0, False
                prev = _list_gray(nb)
                t_prev = _thumb(nb)
                while moved < 0.5 * H:
                    io.scroll(*geo.raw(lx, ly), -1)
                    time.sleep(0.25)
                    io.move(*geo.raw(*sorter.PARK))
                    time.sleep(0.12)
                    b = io.grab()
                    if b is None:
                        end = True
                        break
                    cur = _list_gray(geo.normalize(b))
                    if float(np.abs(cur - prev).mean()) < 1.5:
                        end = True                       # the wheel moved nothing: the end of the list
                        break
                    ncur = geo.normalize(b)
                    img = _shift(prev, cur)
                    est = None
                    if t_prev is not None:
                        t_cur = _thumb(ncur)
                        if t_cur is not None:
                            # the scroll bar's thumb: rough but never fooled by cards that look alike
                            est = (t_cur[0] - t_prev[0]) * H / max(4.0, float(t_prev[1] - t_prev[0]))
                        t_prev = t_cur
                    if img is not None and (est is None or abs(img - est) < max(25.0, 0.25 * H)):
                        step = img
                    elif est is not None and est > 0:
                        step = est
                    else:
                        step = H                         # not measurable: count a whole page
                    moved += step
                    prev = cur
                if moved > 0:
                    new_from = UNIQ_LIST[3] - min(moved, H)
                    page += 1
                if end:
                    if moved > 0:                        # the last bit that came in before the end
                        nb = geo.normalize(b) if b is not None else nb
                        y_start = max(UNIQ_LIST[1] + 12, new_from - 30)
                        _save(os.path.join(out, "고유_%02d_목록_%02d.jpg" % (ci + 1, page + 1)), nb[40:420, 88:392])
                        for yy in range(int(y_start), UNIQ_LIST[3] - 8, 45):
                            for xx in UNIQ_COLS:
                                if halted():
                                    return
                                if _blank(nb, xx, yy):
                                    continue
                                tip = hover((xx, yy), b, tries=3)
                                if tip is not None and keep(tip, "고유_%02d" % (ci + 1)):
                                    got += 1
                    break
            total += got
            log("장비 스캔: 고유 %d/%d번째 종류 %d개" % (ci + 1, len(UNIQ_CATS), got))
        log("장비 스캔: 고유 탭 %d개" % total)

    # 3) the stash tabs
    for tab in SCAN_TABS:
        if halted():
            break
        n_ = geo.normalize(io.grab())
        if not sorter.stash_open(n_):
            io.click(*geo.raw(*sorter.ui_tabs(n_)[1]))
            time.sleep(0.5)
            park()
        n_ = geo.normalize(io.grab())
        tabs = sorter.stash_tabs(n_)
        if tab not in tabs:
            continue                           # (the old stash has 3 shared tabs)
        (dx, dy), _ = sorter.stash_shift(n_)
        io.click(*geo.raw(*tabs[tab]))
        time.sleep(0.5)
        base = park()
        nb = geo.normalize(base)
        _save(os.path.join(out, "000_화면_보관함_%s.jpg" % tab), base)
        if not sorter.stash_open(nb):
            # not recognised (the stash may look new): only the cursor goes over the cells, so look anyway
            log("장비 스캔: 보관함 %s 화면을 확실히 알아보지 못함 - 그래도 칸마다 확인" % tab)
        got = 0
        for r in range(STASH_ROWS):
            for c in range(STASH_COLS):
                if halted():
                    break
                x, y = STASH_X + dx + sorter.CELL * c, STASH_Y + dy + sorter.CELL * r
                if not _color_at(nb, x, y, sorter.CELL):
                    continue
                tip = hover((x + sorter.CELL // 2, y + sorter.CELL // 2), base)
                if tip is not None and keep(tip, "보관_%s_%d_%d" % (tab.replace(" ", ""), c + 1, r + 1)):
                    got += 1
        log("장비 스캔: 보관함 %s %d개" % (tab, got))
    # back to the shop tab, cursor off the items
    try:
        n_ = geo.normalize(io.grab())
        if sorter.seen(n_, "panel") >= 0.6:
            io.click(*geo.raw(*sorter.ui_tabs(n_)[0]))
    except Exception:
        pass
    park()

    # 4) the 고유 (collection) tab: every category, the whole list scrolled through
    if not halted():
        scan_unique()

    if prev_fg:
        try:
            from . import win
            win.focus(prev_fg)                 # give the front back to the window that had it
        except Exception:
            pass

    zpath = out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), fn)
    _set_pending(out)
    log("장비 스캔 끝: 아이템 %d개 -> %s" % (count[0], zpath))
    log("장비 스캔: 다음 판 보스 처치 직후 스킬 정보창도 찍어서 이 ZIP에 더함 (그 뒤 보내 주세요)")
    return zpath
