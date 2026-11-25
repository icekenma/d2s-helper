# -*- coding: utf-8 -*-
"""Live updates: the newest program is kept on GitHub (icekenma/d2s-helper). Once a minute the helper
asks for its VERSION; when it is newer, the whole program is downloaded and copied over this folder
(the user's data lives elsewhere and is not touched). main.py then restarts itself at a quiet moment."""
import io
import os
import shutil
import threading
import time
import urllib.request
import zipfile

REPO = "icekenma/d2s-helper"
BRANCH = "main"
RAW = "https://raw.githubusercontent.com/%s/%s/VERSION" % (REPO, BRANCH)
ZIP = "https://codeload.github.com/%s/zip/refs/heads/%s" % (REPO, BRANCH)
KEEP = {"config.json", "progress.json", "collection.json", "route.json"}       # never overwritten
SKIP_DIRS = {"records", "snaps", "collected", "gear_scan", "__pycache__", ".git"}


def _ver(text):
    try:
        return tuple(int(x) for x in text.strip().split("."))
    except ValueError:
        return ()


def local_version(base):
    try:
        with open(os.path.join(base, "VERSION"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "0"


def _get(url, timeout=15):
    req = urllib.request.Request(url + ("&" if "?" in url else "?") + "t=%d" % time.time(),
                                 headers={"Cache-Control": "no-cache", "User-Agent": "d2s-helper"})
    return urllib.request.urlopen(req, timeout=timeout).read()


class Updater(threading.Thread):
    def __init__(self, base, log, every=60):
        super().__init__(daemon=True)
        self.base, self.log, self.every = base, log, every
        self.ready = None            # the version copied in, waiting for a restart
        self.status = "확인 전"
        self.alive = True

    def run(self):
        time.sleep(5)
        while self.alive:
            try:
                self.check()
            except Exception as e:
                self.status = "확인 실패 (%s)" % e.__class__.__name__
            for _ in range(self.every):
                if not self.alive:
                    return
                time.sleep(1)

    def check(self):
        if self.ready:
            return
        mine = local_version(self.base)
        remote = _get(RAW).decode("utf-8", "replace").strip()
        if not _ver(remote) or _ver(remote) <= _ver(mine):
            self.status = "최신 (v%s)" % mine
            return
        self.status = "v%s 받는 중" % remote
        data = _get(ZIP, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(data))
        names = z.namelist()
        root = names[0].split("/")[0] + "/"
        if not any(n == root + "main.py" for n in names):
            self.status = "받은 파일이 이상함: 건너뜀"
            return
        tmp = os.path.join(self.base, "_update_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
        src = os.path.join(tmp, root.rstrip("/"))
        for dp, dns, fns in os.walk(src):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            rel = os.path.relpath(dp, src)
            for fn in fns:
                if fn in KEEP:
                    continue
                dst_dir = os.path.join(self.base, rel) if rel != "." else self.base
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(os.path.join(dp, fn), os.path.join(dst_dir, fn))
        shutil.rmtree(tmp, ignore_errors=True)
        self.ready = remote
        self.status = "v%s 받음: 판 사이에 다시 시작" % remote
        self.log("새 버전 v%s 받음 (v%s -> v%s): 바로 다시 시작" % (remote, mine, remote))
