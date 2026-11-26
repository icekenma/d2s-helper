# -*- coding: utf-8 -*-
"""Phone dashboard: tiny built-in web server (standard library only)."""
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from pynput.keyboard import Key

from . import win
from .gameio import GameIO
from . import paths

SNAP_RE = re.compile(r"^[0-9_]+\.jpg$")
ITEM_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{3}\.jpg$")
INV_RE = re.compile(r"^(grid_[0-9]{6}|tip_[0-9]{6}_[0-9]{2,3})\.jpg$")


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        s.close()


def read_snaps(base, limit=60):
    p = os.path.join(base, "snaps", "index.jsonl")
    out = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    out.reverse()
    return out


KEYS = {"esc": Key.esc, "enter": Key.enter, "up": Key.up, "down": Key.down, "left": Key.left,
        "right": Key.right, "tab": Key.tab, "c": "c", "t": "t", "space": Key.space}


def essence_stats(days=3):
    """Essences gained per tidy-up (one per run), from the counts of the stacks in the bags:
    gained = now - before for every stack (a stack that went to the stash starts again from 0).
    -> {"total", "runs", "per_run", "per_hour", "today", "stages": [...], "recent": [...]}"""
    from . import sorter
    p = paths.p(sorter.ESS_LOG)
    rows = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    since = time.time() - days * 86400
    base, prev_t, runs = None, None, []
    for r in rows:
        stacks = r.get("stacks") or {}
        first = base is None
        gained = 0 if first else sum(max(0, q - base.get(k, 0)) for k, q in stacks.items())
        base = {k: (0 if k in (r.get("stashed") or []) else q) for k, q in stacks.items()}
        dt = (r["t"] - prev_t) if prev_t else None
        prev_t = r["t"]
        if first or r["t"] < since:
            continue
        runs.append({"t": r["t"], "stage": r.get("stage", 0), "gained": gained,
                     "dt": dt if dt is not None and dt < 15 * 60 else None})
    def summ(rs):
        n = len(rs)
        g = sum(x["gained"] for x in rs)
        secs = sum(x["dt"] for x in rs if x["dt"])
        timed = [x for x in rs if x["dt"]]
        gt = sum(x["gained"] for x in timed)
        return {"runs": n, "gained": g, "per_run": round(g / n, 2) if n else 0,
                "per_hour": round(gt * 3600.0 / secs, 1) if secs > 60 else None,
                "min_per_run": round(secs / 60.0 / len(timed), 1) if timed else None}
    stages = {}
    for x in runs:
        stages.setdefault(x["stage"], []).append(x)
    day0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    out = summ(runs)
    out["today"] = summ([x for x in runs if x["t"] >= day0])
    out["stages"] = sorted([dict(summ(v), stage=k) for k, v in stages.items()],
                           key=lambda d: -(d["per_hour"] or 0))
    out["recent"] = [{"t": x["t"], "stage": x["stage"], "gained": x["gained"]} for x in runs[-12:]][::-1]
    return out


def read_items(base, limit=2000):
    """Registered / stashed items, newest first (sold ones are never logged)."""
    p = paths.p(os.path.join("collected", "index.jsonl"))
    out = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    out.reverse()
    return out


def read_sold(limit=3000):
    """Every item the helper sold, newest first (collected/actions.jsonl, the lines with action 판매)."""
    p = paths.p(os.path.join("collected", "actions.jsonl"))
    out = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if '"판매"' not in line:
                    continue
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if x.get("action") == "판매":
                    out.append(x)
    out = out[-limit:]
    out.reverse()
    return out


def clear_items():
    """Empty the collected-items list. Nothing is deleted: the list and its pictures move into
    collected/old_<date_time> (the list on the page starts again from zero). -> how many were moved."""
    d = paths.p("collected")
    idx = os.path.join(d, "index.jsonl")
    if not os.path.isfile(idx):
        return 0
    with open(idx, "r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    dst = os.path.join(d, "old_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    keep = set()                       # pictures the run log (진행 상황) still shows stay where they are
    try:
        with open(os.path.join(d, "actions.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    keep.add(json.loads(line).get("img") or "")
                except ValueError:
                    pass
    except OSError:
        pass
    for name in os.listdir(d):
        if name == "index.jsonl" or (ITEM_RE.match(name) and name not in keep):
            os.replace(os.path.join(d, name), os.path.join(dst, name))
    return n


class Remote:
    def __init__(self, base, port, pin, get_state, toggle, sort, io_lock=None, title="D2 Survivors"):
        self.base, self.port, self.pin = base, port, str(pin)
        self.get_state, self.toggle, self.sort = get_state, toggle, sort
        self.shot = (0.0, None)
        self.lock = threading.Lock()
        self.io_lock = io_lock or threading.Lock()
        self.io = GameIO(title)
        self.held = {}                  # key name -> release deadline (held while the phone keeps saying so)
        self.drag_until = 0.0
        self.alive = True
        threading.Thread(target=self._expire, daemon=True).start()
        with open(os.path.join(base, "bot", "page.html"), "rb") as f:
            self.page = f.read()
        self.httpd = None

    def url(self):
        return "http://%s:%d/?k=%s" % (local_ip(), self.port, self.pin)

    def live_jpeg(self):
        with self.lock:
            ts, data = self.shot
            if time.time() - ts < 0.07 and data:
                return data
            frame = self.io.grab()                 # works with the game behind other windows too
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960.0 / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 58])
            data = buf.tobytes() if ok else None
            self.shot = (time.time(), data)
            return data

    # ---- control from the phone ------------------------------------------------------------------
    def _pt(self, d):
        """Phone coordinates (0..1 of the picture) -> game client pixels."""
        if not self.io.locate(fresh=False):
            return None
        w, h = self.io.rect[2], self.io.rect[3]
        x, y = float(d.get("x", -1)), float(d.get("y", -1))
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return int(x * (w - 1)), int(y * (h - 1))

    def control(self, path, d):
        # wait a moment for the helper to finish what it is doing; the phone wins if it takes longer
        got = self.io_lock.acquire(timeout=0.3)
        try:
            self._control(path, d)
        finally:
            if got:
                self.io_lock.release()

    def _control(self, path, d):
        if True:
            if path == "/api/tap":
                pt = self._pt(d)
                if pt:
                    mod = d.get("mod") if d.get("mod") in ("ctrl", "shift") else None
                    self.io.click(pt[0], pt[1], mod)
                    self.shot = (0.0, None)
                return
            if path == "/api/drag":
                act = d.get("act")
                if act == "end":
                    self.io.drag_end()
                    self.drag_until = 0.0
                    return
                pt = self._pt(d)
                if not pt:
                    return
                if act == "start":
                    self.io.drag_start(*pt)
                else:
                    self.io.move(*pt)
                self.drag_until = time.time() + 1.5
                return
            if path == "/api/key":
                name = d.get("k")
                k = KEYS.get(name)
                if k is None:
                    return
                act = d.get("act", "tap")
                if act == "tap":
                    self.io.key(k)
                elif act == "down":
                    if name not in self.held:
                        self.io.key_down(k)
                    self.held[name] = time.time() + 1.2
                elif act == "up" and name in self.held:
                    del self.held[name]
                    self.io.key_up(k)

    def _expire(self):
        """A key or a drag the phone stopped confirming (connection lost, app switched) is let go."""
        while self.alive:
            time.sleep(0.3)
            now = time.time()
            for name, until in list(self.held.items()):
                if now > until:
                    self.held.pop(name, None)
                    try:
                        self.io.key_up(KEYS[name])
                    except Exception:
                        pass
            if self.drag_until and now > self.drag_until:
                self.drag_until = 0.0
                try:
                    self.io.drag_end()
                except Exception:
                    pass

    def start(self):
        remote = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if "Cache-Control" not in (extra or {}):
                    self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _authed(self, q):
                if q.get("k", [""])[0] == remote.pin:
                    return True
                return ("k=" + remote.pin) in (self.headers.get("Cookie") or "").replace(" ", "").split(";")

            def _route(self, method):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if not self._authed(q):
                    return self._send(403, "주소 끝의 접속 코드(?k=...)가 맞지 않습니다. PC 화면에 표시된 주소로 다시 접속하세요.".encode("utf-8"),
                                      "text/plain; charset=utf-8")
                ck = {"Set-Cookie": "k=%s; Max-Age=31536000; Path=/; SameSite=Lax" % remote.pin}
                if u.path == "/" and method == "GET":
                    return self._send(200, remote.page, extra=ck)
                if u.path == "/api/status" and method == "GET":
                    st = dict(remote.get_state())
                    st["snaps"] = read_snaps(remote.base)
                    st["now"] = time.time()
                    try:
                        with open(os.path.join(remote.base, "VERSION"), "r", encoding="utf-8") as vf:
                            st["version"] = vf.read().strip()     # the phone page reloads itself when it changes
                    except OSError:
                        pass
                    return self._send(200, json.dumps(st, ensure_ascii=False).encode("utf-8"),
                                      "application/json; charset=utf-8")
                if u.path == "/api/toggle" and method == "POST":
                    remote.toggle()
                    return self._send(200, b"{}", "application/json")
                if u.path == "/api/stage" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        stage = max(0, min(999, int(d.get("stage", 0))))
                    except Exception:
                        return self._send(400, b'{"error":"stage"}', "application/json")
                    cb = getattr(remote, "set_stage", None)
                    if cb:
                        cb(stage)
                    return self._send(200, json.dumps({"stage": stage}).encode("utf-8"), "application/json")
                if u.path == "/api/fast" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        on = bool(json.loads(self.rfile.read(min(n, 1024)) or b"{}").get("on"))
                    except Exception:
                        return self._send(400, b'{"error":"fast"}', "application/json")
                    cb = getattr(remote, "set_fast", None)
                    if cb:
                        cb(on)
                    return self._send(200, json.dumps({"on": on}).encode("utf-8"), "application/json")
                if u.path == "/api/climb" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        on = bool(json.loads(self.rfile.read(min(n, 1024)) or b"{}").get("on"))
                    except Exception:
                        return self._send(400, b'{"error":"climb"}', "application/json")
                    cb = getattr(remote, "set_climb", None)
                    if cb:
                        cb(on)
                    return self._send(200, json.dumps({"on": on}).encode("utf-8"), "application/json")
                if u.path == "/api/ess_check" and method == "POST":
                    cb = getattr(remote, "ess_check", None)
                    if cb:
                        cb()
                    return self._send(200, b'{"ok":true}', "application/json")
                if u.path == "/api/tidy_now" and method == "POST":
                    cb = getattr(remote, "tidy_now", None)
                    if cb:
                        cb()
                    return self._send(200, b"{}", "application/json")
                if u.path == "/api/limits" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        d = {k: int(d[k]) for k in ("normal", "fast") if k in d}
                    except Exception:
                        return self._send(400, b'{"error":"limits"}', "application/json")
                    cb = getattr(remote, "set_limits", None)
                    if cb and d:
                        cb(d)
                    return self._send(200, json.dumps(d).encode("utf-8"), "application/json")
                if u.path == "/api/alert" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        d = {k: d[k] for k in ("on", "minutes") if k in d}
                    except Exception:
                        return self._send(400, b'{"error":"alert"}', "application/json")
                    cb = getattr(remote, "set_alert", None)
                    if cb and d:
                        cb(d)
                    return self._send(200, json.dumps(d).encode("utf-8"), "application/json")
                if u.path == "/api/alert/test" and method == "POST":
                    cb = getattr(remote, "test_alert", None)
                    if cb:
                        cb()
                    return self._send(200, b"{}", "application/json")
                if u.path in ("/api/stash", "/api/inv") and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        v = int(json.loads(self.rfile.read(min(n, 1024)) or b"{}").get("n"))
                    except Exception:
                        return self._send(400, b'{"error":"count"}', "application/json")
                    from . import runlog
                    if u.path == "/api/stash":
                        runlog.set_stash(v)
                    else:
                        runlog.set_inv(v)
                    return self._send(200, json.dumps({"n": v}).encode("utf-8"), "application/json")
                if u.path == "/api/play" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        d = {str(k): bool(v) for k, v in d.items()}
                    except Exception:
                        return self._send(400, b'{"error":"play"}', "application/json")
                    cb = getattr(remote, "set_play", None)
                    if cb and d:
                        cb(d)
                    return self._send(200, json.dumps(d).encode("utf-8"), "application/json")
                if u.path == "/api/fastext" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        ext = {k: bool(d[k]) for k in ("count", "jewel", "unique") if k in d}
                    except Exception:
                        return self._send(400, b'{"error":"fastext"}', "application/json")
                    cb = getattr(remote, "set_fast_ext", None)
                    if cb and ext:
                        cb(ext)
                    return self._send(200, json.dumps(ext).encode("utf-8"), "application/json")
                if u.path in ("/api/sort", "/api/scan") and method == "POST":
                    remote.sort()                # (this slot now starts the gear scan)
                    return self._send(200, b"{}", "application/json")
                if u.path == "/scan/latest.zip" and method == "GET":
                    d = os.path.join(remote.base, "gear_scan")
                    zips = sorted(z for z in (os.listdir(d) if os.path.isdir(d) else []) if z.endswith(".zip"))
                    if not zips:
                        return self._send(404, "아직 장비 스캔 결과가 없음".encode("utf-8"), "text/plain; charset=utf-8")
                    with open(os.path.join(d, zips[-1]), "rb") as f:
                        data = f.read()
                    return self._send(200, data, "application/zip",
                                      {"Content-Disposition": 'attachment; filename="gear_scan_%s"' % zips[-1]})
                if u.path in ("/api/tap", "/api/drag", "/api/key") and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                        remote.control(u.path, d)
                    except Exception as e:
                        return self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")
                    return self._send(200, b"{}", "application/json")
                if u.path == "/api/runs/clear" and method == "POST":
                    from . import runlog
                    n = runlog.clear()
                    return self._send(200, json.dumps({"cleared": n}).encode("utf-8"), "application/json")
                if u.path == "/api/items/clear" and method == "POST":
                    n = clear_items()
                    return self._send(200, json.dumps({"cleared": n}).encode("utf-8"), "application/json")
                if u.path == "/api/keeplist" and method == "GET":
                    from . import scorer
                    ok, how = scorer.status()
                    body = json.dumps({"order": scorer.KEEP_LIST.get("order", []),
                                       "classes": scorer.KEEP_LIST.get("classes", {}),
                                       "pics": sorted(scorer.keep_pictures()), "ocr": ok, "ocr_text": how,
                                       "min_star": scorer.KEEP_MIN_STAR[0], "picked": sorted(scorer.KEEP_PICK | scorer.ALWAYS_KEEP),
                                       "always": sorted(scorer.ALWAYS_KEEP)},
                                      ensure_ascii=False)
                    return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
                if u.path == "/api/keeppick" and method == "POST":
                    from . import scorer
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                        name, on = str(d.get("n", "")), bool(d.get("on"))
                    except Exception:
                        return self._send(400, b'{"error":"pick"}', "application/json")
                    if name not in scorer.KEEP_SET:
                        return self._send(400, b'{"error":"name"}', "application/json")
                    scorer.set_pick(name, on)
                    cb = getattr(remote, "note", None)
                    if cb:
                        cb("휴대폰 보관 목록: '%s' %s" % (name, "무조건 보관으로 선택" if on else "선택 해제"))
                    return self._send(200, json.dumps({"n": name, "on": on}, ensure_ascii=False).encode("utf-8"),
                                      "application/json")
                if u.path == "/api/keepstar" and method == "POST":
                    from . import scorer
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        star = max(1, min(5, int(d.get("star", 1))))
                    except Exception:
                        return self._send(400, b'{"error":"star"}', "application/json")
                    scorer.set_min_star(star)            # at once; the PC side saves it
                    cb = getattr(remote, "set_keep_star", None)
                    if cb:
                        cb(star)
                    return self._send(200, json.dumps({"star": star}).encode("utf-8"), "application/json")
                if u.path == "/api/runs" and method == "GET":
                    from . import runlog
                    from . import namer
                    rs = runlog.build(300)
                    for r_ in rs[:150]:
                        namer.fill(r_.get("items") or [])
                    body = json.dumps({"runs": rs[:150], "hold": runlog.holdings(rs), "now": time.time()}, ensure_ascii=False)
                    return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
                if u.path == "/api/essences" and method == "GET":
                    body = json.dumps(essence_stats(), ensure_ascii=False)
                    return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
                if u.path == "/api/items" and method == "GET":
                    from . import namer
                    body = json.dumps({"items": namer.fill(read_items(remote.base)), "now": time.time()}, ensure_ascii=False)
                    return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
                if u.path == "/api/sold" and method == "GET":
                    from . import namer
                    sold = read_sold()
                    namer.fill(sold[:600])
                    body = json.dumps({"items": sold, "now": time.time()}, ensure_ascii=False)
                    return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
                if u.path == "/api/invnow" and method == "GET":
                    try:
                        with open(paths.p("inv_now.json"), "r", encoding="utf-8") as f:
                            d = json.load(f)
                    except (OSError, ValueError):
                        d = {"items": [], "t": 0}
                    from . import namer
                    namer.fill(d.get("items") or [], folder="inv_now")
                    d["now"] = time.time()
                    return self._send(200, json.dumps(d, ensure_ascii=False).encode("utf-8"),
                                      "application/json; charset=utf-8")
                if u.path == "/api/sellnow" and method == "POST":
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                        d = json.loads(self.rfile.read(min(n, 1024)) or b"{}")
                        d = {"c": int(d["c"]), "r": int(d["r"]), "name": str(d.get("name") or "")[:40]}
                    except Exception:
                        return self._send(400, b'{"error":"sell"}', "application/json")
                    cb = getattr(remote, "sell_now", None)
                    if cb:
                        cb(d)
                    return self._send(200, json.dumps(d, ensure_ascii=False).encode("utf-8"), "application/json")
                if u.path == "/api/invview" and method == "POST":
                    cb = getattr(remote, "inv_view", None)
                    if cb:
                        cb()
                    return self._send(200, b"{}", "application/json")
                if u.path.startswith("/invnow/") and method == "GET":
                    name = u.path[8:]
                    p = paths.p(os.path.join("inv_now", name))
                    if INV_RE.match(name) and os.path.isfile(p):
                        with open(p, "rb") as f:
                            return self._send(200, f.read(), "image/jpeg", {"Cache-Control": "max-age=3600"})
                    return self._send(404, b"", "text/plain")
                if u.path.startswith("/items/") and method == "GET":
                    name = u.path[7:]
                    p = paths.p(os.path.join("collected", name))
                    if ITEM_RE.match(name) and os.path.isfile(p):
                        with open(p, "rb") as f:
                            return self._send(200, f.read(), "image/jpeg",
                                              {"Cache-Control": "max-age=86400"})
                    return self._send(404, b"", "text/plain")
                if u.path == "/stream.mjpg" and method == "GET":
                    # a live picture: one JPEG after another on the same connection (about 12 a second)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    last = None
                    started = time.time()
                    try:
                        while remote.alive and time.time() - started < 600:     # the page reconnects
                            t0 = time.time()
                            try:
                                data = remote.live_jpeg()
                            except Exception:
                                data = None
                            if data and data is not last:
                                last = data
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                                self.wfile.flush()
                            time.sleep(max(0.01, 0.08 - (time.time() - t0)))
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        pass
                    return
                if u.path == "/live.jpg" and method == "GET":
                    try:
                        data = remote.live_jpeg()
                    except Exception:
                        data = None
                    if not data:
                        return self._send(404, b"", "text/plain")
                    return self._send(200, data, "image/jpeg")
                if u.path.startswith("/snaps/") and method == "GET":
                    name = u.path[7:]
                    p = os.path.join(remote.base, "snaps", name)
                    if SNAP_RE.match(name) and os.path.isfile(p):
                        with open(p, "rb") as f:
                            return self._send(200, f.read(), "image/jpeg")
                return self._send(404, b"not found", "text/plain")

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

        for i in range(20):
            try:
                self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), H)
                break
            except OSError:
                if i == 19:
                    raise
                time.sleep(0.5)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.alive = False
        try:
            self.io.close()
        except Exception:
            pass
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()

