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
_open = {"t": None, "stage": 0}


def _write(d):
    try:
        with _lock, open(paths.p(RUNS), "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except OSError:
        pass


def start(t, stage):
    if _open["t"] is not None:
        end(t, "다음 판 시작")
    _open["t"], _open["stage"] = t, int(stage or 0)
    _write({"ev": "start", "t": t, "stage": int(stage or 0)})


def end(t, why=""):
    if _open["t"] is None:
        return
    _open["t"] = None
    _write({"ev": "end", "t": t, "why": why})


def tag(k, v, t=None):
    """A fact about the run under way: act (1..5), skip (the act it was left for), boss (kill time)."""
    if _open["t"] is None:
        return
    _write({"ev": "tag", "t": t if t is not None else time.time(), "k": k, "v": v})


def essence_guess(act, stage):
    """Burning essences a cleared run gives: acts 3 and 4 give 2, the others 1; +1 for every 50 stages."""
    if not act:
        return None
    return (2 if act in (3, 4) else 1) + int(stage or 0) // 50


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
    base, rows = None, []
    from . import sorter
    for r in _lines(sorter.ESS_LOG):
        stacks = r.get("stacks") or {}
        if base is not None:
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
            runs.append({"stage": r.get("stage", 0), "start": r["t"], "end": None, "why": "",
                         "act": None, "skip": None, "boss": None})
        elif r.get("ev") == "tag" and runs and runs[-1]["end"] is None:
            k, v = r.get("k"), r.get("v")
            if k == "skip":
                runs[-1]["skip"] = v or "?"            # left at once for its act (패스트 모드)
                if isinstance(v, int) and v:
                    runs[-1]["act"] = v
            elif k in ("act", "boss"):
                runs[-1][k] = v
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
        r["boss_secs"] = (r["boss"] - r["start"]) if r.get("boss") else None
        r["est"] = essence_guess(r.get("act"), r.get("stage")) if r.get("boss") and not r.get("skip") else None
        r["items"] = [{"t": x.get("t"), "action": x.get("action", ""), "tab": x.get("tab", ""),
                       "kind": x.get("kind", ""), "color": x.get("color", ""), "ed": x.get("ed"),
                       "name": x.get("name") or x.get("raw") or x.get("kind") or x.get("quality") or "아이템"}
                      for x in items if r["start"] <= float(x.get("t", 0) or 0) < hi]
    for k, r in enumerate(runs):
        r["n"] = k + 1
    return runs[::-1]


STASH = "essence_stash.json"


def set_stash(n):
    """The user's count of burning essences in the stash, as of now (the helper adds what it stashes later)."""
    try:
        with _lock, open(paths.p(STASH), "w", encoding="utf-8") as f:
            json.dump({"n": max(0, int(n)), "t": time.time()}, f)
    except (OSError, ValueError):
        pass


def holdings(runs=None):
    """Burning essences the character holds: the last inventory reading, the stash (the user's count plus
    what the helper stashed after it) and, since that reading, what the runs gave (counted by act / stage).
    -> {"inv", "inv_t", "stash", "stash_set", "since", "total"} (None where not known)"""
    from . import sorter
    rows = _lines(sorter.ESS_LOG)
    try:
        with open(paths.p(STASH), "r", encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = None
    inv = inv_t = None
    moved = 0
    for r in rows:
        stacks, gone = r.get("stacks") or {}, set(r.get("stashed") or [])
        inv, inv_t = sum(q for k, q in stacks.items() if k not in gone), r.get("t")
        if st and r.get("t", 0) > st.get("t", 0):
            moved += sum(q for k, q in stacks.items() if k in gone)
    stash = (int(st.get("n", 0)) + moved) if st else None
    since = 0
    if inv_t:
        for r in runs if runs is not None else build(300):
            if r.get("boss") and r["boss"] > inv_t and not r.get("skip") and r.get("est"):
                since += r["est"]
    total = None
    if inv is not None or stash is not None:
        total = (inv or 0) + (stash or 0) + since
    return {"inv": inv, "inv_t": inv_t, "stash": stash, "stash_set": bool(st), "stash_t": st.get("t") if st else None,
            "since": since, "total": total}


def clear():
    """Start the run table again: the old lines are moved aside (runs_old_<time>.jsonl). A run under way
    stays in the new table, and the last essence count stays as the starting point of the next one.
    -> how many runs were moved aside"""
    from . import sorter
    stamp = time.strftime("%Y%m%d_%H%M%S")
    n = sum(1 for r in _lines(RUNS) if r.get("ev") == "start")
    with _lock:
        for name in (RUNS, sorter.ESS_LOG):
            p = paths.p(name)
            if not os.path.exists(p):
                continue
            last = None
            if name == sorter.ESS_LOG:
                rows = _lines(name)
                last = rows[-1] if rows else None
            try:
                os.replace(p, paths.p(name.replace(".jsonl", "_old_%s.jsonl" % stamp)))
            except OSError:
                continue
            if last is not None:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(json.dumps(last, ensure_ascii=False) + "\n")
    if _open["t"] is not None:
        _write({"ev": "start", "t": _open["t"], "stage": _open["stage"]})
    return n
