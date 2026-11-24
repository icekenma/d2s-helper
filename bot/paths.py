# -*- coding: utf-8 -*-
"""Where the user's data lives: one fixed folder outside the program folder, so a new version
(unzipped anywhere) keeps the arrow-key recording, the phone PIN, the registered list and the
collected-items history."""
import glob
import os
import shutil

if os.name == "nt":
    DATA = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "D2SHelper")
else:
    DATA = os.path.join(os.path.expanduser("~"), ".d2s_helper")

FILES = ("config.json", "route.json", "collection.json", "progress.json")
DIRS = ("collected",)


def p(name):
    return os.path.join(DATA, name)


def _old_installs(base):
    """This folder first, then every other copy of the program next to it or one level up."""
    out = [base]
    parent = os.path.dirname(base)
    for root in {parent, os.path.dirname(parent)}:
        for d in glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "*", "*")):
            if os.path.isfile(os.path.join(d, "main.py")) and os.path.isdir(os.path.join(d, "bot")):
                if os.path.abspath(d) not in [os.path.abspath(x) for x in out]:
                    out.append(d)
    return out


def migrate(base, log=None):
    """First run of this version: bring the data over from an older copy (the newest one wins)."""
    os.makedirs(DATA, exist_ok=True)
    try:
        olds = _old_installs(base)
    except Exception:
        olds = [base]
    moved = []
    for name in FILES:
        if os.path.exists(p(name)):
            continue
        cands = [os.path.join(d, name) for d in olds if os.path.isfile(os.path.join(d, name))]
        if cands:
            src = max(cands, key=os.path.getmtime)
            try:
                shutil.copy2(src, p(name))
                moved.append(name)
            except OSError:
                pass
    for name in DIRS:
        if os.path.isdir(p(name)):
            continue
        cands = [os.path.join(d, name) for d in olds if os.path.isdir(os.path.join(d, name))]
        if cands:
            src = max(cands, key=os.path.getmtime)
            try:
                shutil.copytree(src, p(name))
                moved.append(name)
            except OSError:
                pass
    return moved
