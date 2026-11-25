"""One line per run: when the stage was entered and left. Joined with the essence counts and the
collected-items log to show, for every run, how long it took, how many essences came in and what was
stashed or registered."""
import json
import os
import threading
import time

from . import paths

RUNS = "runs.jsonl"
_lock = threading.Lock()
_open = {"t": None}


def _write(d):
    try:
        with _lock, open(paths.p(RUNS), "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except OSError:
        pass


def start(t, stage):
    if _open["t"] is not None:
        end(t, "다음 판 시작")
    _open["t"] = t
    _write({"ev": "start", "t": t, "stage": int(stage or 0)})


def end(t, why=""):
    if _open["t"] is None:
        return
    _open["t"] = None
    _write({"ev": "end", "t": t, "why": why})


def _lines(name):
    p = paths.p(name)
    out = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _essence_rows():
    """(t, gained) per tidy-up, from the stack counts (see sorter.record_essences)."""
    base, rows = {}, []
    for r in _lines("essence_log.jsonl"):
        stacks = r.get("stacks") or {}
        if base:
            g = sum(max(0, q - base.get(k, 0)) for k, q in stacks.items())
            rows.append((r["t"], g))
        base = {k: (0 if k in (r.get("stashed") or []) else q) for k, q in stacks.items()}
    return rows


def build(limit=200):
    """-> newest first: {"n", "stage", "start", "end", "secs", "live", "ess", "items": [...]}"""
    runs = []
    for r in _lines(RUNS):
        if r.get("ev") == "start":
            if runs and runs[-1]["end"] is None:
                runs[-1]["end"] = r["t"]
            runs.append({"stage": r.get("stage", 0), "start": r["t"], "end": None, "why": ""})
        elif r.get("ev") == "end" and runs and runs[-1]["end"] is None:
            runs[-1]["end"] = r["t"]
            runs[-1]["why"] = r.get("why", "")
    runs = runs[-limit:]
    ess = _essence_rows()
    items = _lines(os.path.join("collected", "index.jsonl"))
    now = time.time()
    for i, r in enumerate(runs):
        # the tidy-up at the smith may come a little after the stage is left: count it with this run
        nxt = runs[i + 1]["start"] if i + 1 < len(runs) else None
        hi = r["end"] + 90 if r["end"] else now + 1
        if nxt is not None:
            hi = min(hi, nxt)
        r["live"] = r["end"] is None and i == len(runs) - 1
        if r["end"] is None and not r["live"]:
            r["end"] = nxt
        r["secs"] = (r["end"] or now) - r["start"]
        e = [g for t, g in ess if r["start"] <= t < hi]
        r["ess"] = sum(e) if e else None
        r["items"] = [{"t": x.get("t"), "action": x.get("action", ""), "tab": x.get("tab", ""),
                       "name": x.get("name") or x.get("raw") or x.get("kind") or x.get("quality") or "아이템"}
                      for x in items if r["start"] <= float(x.get("t", 0) or 0) < hi]
    for k, r in enumerate(runs):
        r["n"] = k + 1
    return runs[::-1]
