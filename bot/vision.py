# -*- coding: utf-8 -*-
"""Template matching helpers (no Windows dependency, testable anywhere)."""
import cv2
import numpy as np


def load_template(path):
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image(path, bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise IOError("PNG encode failed")
    buf.tofile(path)


def find(frame_bgr, tmpl_bgr):
    """Return (score, (cx, cy)) of the best match, score in 0..1."""
    if tmpl_bgr is None or frame_bgr is None:
        return 0.0, (0, 0)
    th, tw = tmpl_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    if th > fh or tw > fw:
        return 0.0, (0, 0)
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    if not np.isfinite(maxv):
        maxv = 0.0
    return float(maxv), (maxloc[0] + tw // 2, maxloc[1] + th // 2)
