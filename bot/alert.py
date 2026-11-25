# -*- coding: utf-8 -*-
"""Warns the phone when the burning essences stop coming in: every check looks at the last N minutes
of runs; no essence in that time while the helper runs -> a push through ntfy.sh (the free ntfy app on
the phone, subscribed to this PC's topic). Repeats every N minutes while it lasts, and says once when
essences come in again."""
import json
import random
import string
import threading
import time
import urllib.request

from . import runlog

SERVER = "https://ntfy.sh/"
DEFAULT_MIN = 30


def new_topic():
    return "d2s-" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))


def send(topic, title, body, prio="high", tags="warning"):
    """JSON publish (Korean title and text go through as UTF-8). -> "" on success, else the error text."""
    try:
        data = json.dumps({"topic": topic, "title": title, "message": body,
                           "priority": {"min": 1, "low": 2, "default": 3, "high": 4, "max": 5}.get(prio, 4),
                           "tags": [t for t in tags.split(",") if t]}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(SERVER, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        return ""
    except Exception as e:
        return "%s: %s" % (e.__class__.__name__, e)


def gained(runs, t0, t1):
    """Essences that came in between t0 and t1: read counts, else act / stage, else 1 for a boss kill."""
    g, n = 0, 0
    for r in runs:
        t = r.get("boss") or (r.get("end") if r.get("ess") else None)
        if not t or not (t0 <= t < t1) or r.get("skip"):
            continue
        n += 1
        if r.get("ess") is not None:
            g += r["ess"]
        elif r.get("est") is not None:
            g += r["est"]
        else:
            g += 1
    return g, n


class Watch:
    """check() once a minute from the UI thread; the sending happens on a thread of its own."""

    def __init__(self, cfg, save, log):
        self.cfg, self.save, self.log = cfg, save, log
        a = cfg.setdefault("alert", {})
        a.setdefault("on", True)
        a.setdefault("minutes", DEFAULT_MIN)
        if not a.get("topic"):
            a["topic"] = new_topic()
            save(cfg)
        self.since = None            # when the helper started running (the first window starts there)
        self.warn = False
        self.sent_at = 0.0
        self.checked_at = 0.0
        self.last = {"gain": None, "runs": None, "at": None}
        self.err = ""

    @property
    def conf(self):
        return self.cfg["alert"]

    def state(self):
        c = self.conf
        return {"on": bool(c.get("on")), "topic": c.get("topic"), "minutes": int(c.get("minutes") or DEFAULT_MIN),
                "warn": self.warn, "sent_at": self.sent_at or None, "err": self.err,
                "gain": self.last["gain"], "runs": self.last["runs"], "at": self.last["at"],
                "url": SERVER + (c.get("topic") or "")}

    def _post(self, title, body, prio="high", tags="warning"):
        def go():
            e = send(self.conf["topic"], title, body, prio, tags)
            self.err = e
            self.log("휴대폰 알림 %s: %s" % ("보냄" if not e else "실패 (" + e + ")", title))
        threading.Thread(target=go, daemon=True).start()

    def test(self):
        self._post("D2S 도우미 시험 알림", "이 알림이 보이면 경고 알림이 잘 옵니다.", "default", "bell")

    def check(self, running, now=None):
        now = now or time.time()
        if not running:
            self.since, self.warn = None, False
            return
        if self.since is None:
            self.since = now
        mins = int(self.conf.get("minutes") or DEFAULT_MIN)
        win = mins * 60
        if now - self.since < win or now - self.checked_at < 60:
            return                                   # not N minutes of running yet / once a minute
        self.checked_at = now
        try:
            runs = runlog.build(200)
        except Exception:
            return
        g, n = gained(runs, now - win, now)
        self.last = {"gain": g, "runs": n, "at": now}
        if g > 0:
            if self.warn:
                self.warn = False
                if self.conf.get("on"):
                    self._post("D2S: 정수가 다시 들어옴", "최근 %d분 동안 불타는 정수 %d개 (%d판)." % (mins, g, n),
                               "default", "white_check_mark")
            return
        if not self.warn or now - self.sent_at >= win:
            first = not self.warn
            self.warn, self.sent_at = True, now
            last = next((r for r in runs if r.get("boss")), None)
            ago = " 마지막 보스 처치: %s" % time.strftime("%H:%M", time.localtime(last["boss"])) if last else ""
            self.log("경고: %d분 동안 불타는 정수가 늘지 않음%s" % (mins, "" if first else " (계속)"))
            if self.conf.get("on"):
                self._post("D2S 경고: %d분 동안 정수 0개" % mins,
                           "도우미는 켜져 있는데 최근 %d분 동안 보스를 잡지 못했습니다 (끝난 판 %d개).%s 게임 화면을 확인하세요."
                           % (mins, sum(1 for r in runs if r.get("end") and r["end"] >= now - win), ago))
