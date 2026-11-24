# -*- coding: utf-8 -*-
"""장비 스캔: with the smith's window open, the cursor rests on every worn item, every item in the bags
and in the stash tabs, and the info box of each is saved as a picture (nothing is clicked on an item,
nothing is moved). The pictures go into one ZIP to send for a best-gear recommendation."""
import os
import time
import zipfile

import cv2
import numpy as np

from . import sorter

# worn items on the right-hand panel (800x600 layout)
EQUIP = [("투구", (561, 93)), ("목걸이", (620, 106)), ("무기", (447, 159)), ("갑옷", (561, 178)),
         ("방패_화살통", (678, 159)), ("장갑", (447, 265)), ("반지1", (506, 249)), ("허리띠", (561, 249)),
         ("반지2", (620, 249)), ("신발", (678, 265))]
STASH_X, STASH_Y, STASH_COLS, STASH_ROWS = 95, 65, 10, 12
SCAN_TABS = sorter.STASH_ORDER


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

    def hover(pt, base):
        io.move(*geo.raw(*pt))
        time.sleep(0.25)
        hov = io.grab()
        if hov is None or base is None:
            return None
        return sorter.tooltip_crop(base, hov, geo)

    def keep(tip, name):
        for t in tips:
            if sorter.same_tip(tip, t):
                return False                   # another cell of an item already saved
        tips.append(tip)
        count[0] += 1
        _save(os.path.join(out, "%03d_%s.jpg" % (count[0], name)), tip)
        return True

    # 1) worn items
    base = park()
    _save(os.path.join(out, "000_화면_착용.jpg"), base)
    worn = 0
    for name, pt in EQUIP:
        if halted():
            break
        tip = hover(pt, base)
        if tip is not None and keep(tip, "착용_" + name):
            worn += 1
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

    zpath = out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), fn)
    log("장비 스캔 끝: 아이템 %d개 -> %s (이 ZIP을 보내 주세요)" % (count[0], zpath))
    return zpath
