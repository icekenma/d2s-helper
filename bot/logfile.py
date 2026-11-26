# -*- coding: utf-8 -*-
"""Every line of the helper's log, kept in day files (logs/YYYY-MM-DD.txt in the data folder, 7 days),
and a text export of a time window for the phone's '로그 파일 받기': the settings, the state, the
runs of that window and every log line, in one file that is easy to read back."""
import datetime
import json
import os
import threading
import time

from . import paths

DIR = "logs"
KEEP_DAYS = 7
_lock = threading.Lock()
_f = {"day": None, "fh": None}
HIDE = ("remote_pin", "pin", "ntfy_topic", "topic", "token", "password")


def _dir():
    d = paths.p(DIR)
    os.makedirs(d, exist_ok=True)
    return d


def _prune():
    try:
        cut = time.time() - KEEP_DAYS * 86400
        for n in os.listdir(_dir()):
            p = os.path.join(_dir(), n)
            if n.endswith(".txt") and os.path.getmtime(p) < cut:
                os.remove(p)
    except OSError:
        pass


def write(msg, t=None):
    """One log line, with its date and time, into today's file."""
    t = t or time.time()
    lt = time.localtime(t)
    day = time.strftime("%Y-%m-%d", lt)
    line = "%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", lt), str(msg).replace("\n", " / "))
    with _lock:
        try:
            if _f["day"] != day or _f["fh"] is None:
                if _f["fh"] is not None:
                    _f["fh"].close()
                _f["fh"] = open(os.path.join(_dir(), day + ".txt"), "a", encoding="utf-8", buffering=1)
                _f["day"] = day
                _prune()
            _f["fh"].write(line)
        except OSError:
            _f["fh"] = None


def _lines_since(t0):
    """Log lines from t0 on (day files of every day touched)."""
    out = []
    d0 = datetime.date.fromtimestamp(t0)
    d1 = datetime.date.today()
    cut = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    day = d0
    while day <= d1:
        p = os.path.join(_dir(), day.isoformat() + ".txt")
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if ln[:19] >= cut:
                        out.append(ln.rstrip("\n"))
        except OSError:
            pass
        day += datetime.timedelta(days=1)
    return out


def _clean(d):
    if isinstance(d, dict):
        return {k: ("***" if any(h in str(k).lower() for h in HIDE) else _clean(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean(x) for x in d]
    return d


def _jsonl_since(name, t0, limit=400):
    out = []
    try:
        with open(paths.p(name), "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                t = r.get("t") or r.get("time") or 0
                try:
                    t = float(t)
                except (TypeError, ValueError):
                    t = 0
                if t >= t0:
                    out.append(r)
    except OSError:
        pass
    return out[-limit:]


def export(minutes, meta=None):
    """-> (file name, utf-8 bytes): the last `minutes` (0 = today from midnight) of everything."""
    now = time.time()
    if minutes and minutes > 0:
        t0 = now - minutes * 60
        span = "최근 %d분" % minutes if minutes < 120 else "최근 %g시간" % (minutes / 60.0)
    else:
        lt = time.localtime(now)
        t0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        span = "오늘 전체"
    meta = _clean(meta or {})
    lines = _lines_since(t0)
    out = ["# D2S Helper 로그 (%s)" % span,
           "# 받은 시각: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
           "# 범위: %s ~ 지금, 로그 %d줄" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), len(lines)),
           ""]
    for k in ("version", "state", "config"):
        if k in meta:
            out.append("## %s" % {"version": "버전", "state": "지금 상태", "config": "설정"}[k])
            v = meta[k]
            out.append(json.dumps(v, ensure_ascii=False, indent=1, sort_keys=True) if not isinstance(v, str) else v)
            out.append("")
    runs = _jsonl_since("runs.jsonl", t0)
    if runs:
        out.append("## 판 기록 (runs.jsonl, %d줄)" % len(runs))
        out += [json.dumps(r, ensure_ascii=False) for r in runs]
        out.append("")
    acts = _jsonl_since(os.path.join("collected", "actions.jsonl"), t0)
    if acts:
        out.append("## 아이템 처리 기록 (actions.jsonl, %d줄)" % len(acts))
        out += [json.dumps({k: v for k, v in r.items() if k not in ("img", "tip")}, ensure_ascii=False) for r in acts]
        out.append("")
    out.append("## 로그")
    out += lines or ["(이 시간에 남은 로그가 없음: 이 기능은 v3.50부터 로그를 파일에 남김)"]
    tag = "today" if not (minutes and minutes > 0) else "%dm" % minutes
    name = "d2s_log_%s_%s.txt" % (time.strftime("%Y%m%d_%H%M"), tag)
    return name, ("﻿" + "\n".join(out) + "\n").encode("utf-8")
