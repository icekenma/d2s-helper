# -*- coding: utf-8 -*-
"""Item names for the phone, read from the saved info-box pictures with the OCR that comes with Windows
(Windows.Media.Ocr, Korean): no Tesseract needed. Runs in the background a few pictures at a time; the
names are kept in names.json next to the pictures (picture -> name, base)."""
import json
import os
import re
import subprocess
import threading
import time

import cv2
import numpy as np

from . import paths

CACHE = "names.json"
GENERIC = {"", "아이템", "유니크", "세트", "레어", "매직", "일반", "기타", "유니크 부적", "읽지 못한 아이템"}
ITEM_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{3}\.jpg$")
INV_RE = re.compile(r"^tip_[0-9]{6}_[0-9]{2,3}\.jpg$")
_lock = threading.Lock()
_cache = {"d": None, "t": 0.0}

PS = r'''
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language 'ko'))
if ($engine -eq $null) { "NOKO" | Set-Content -Encoding UTF8 -Path $args[1]; exit 2 }
$out = @{}
foreach ($p in (Get-Content -Encoding UTF8 -Path $args[0])) {
  if (-not $p) { continue }
  try {
    $f = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile])
    $s = Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $d = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $b = Await ($d.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $r = Await ($engine.RecognizeAsync($b)) ([Windows.Media.Ocr.OcrResult])
    $out[$p] = @($r.Lines | ForEach-Object { $_.Text })
    $s.Dispose()
  } catch { $out[$p] = @() }
}
$out | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -Path $args[1]
'''


def _load():
    with _lock:
        p = paths.p(CACHE)
        try:
            m = os.path.getmtime(p)
        except OSError:
            m = 0.0
        if _cache["d"] is None or m != _cache["t"]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _cache["d"] = json.load(f)
            except (OSError, ValueError):
                _cache["d"] = {}
            _cache["t"] = m
        return _cache["d"]


def _save(d):
    with _lock:
        p = paths.p(CACHE)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, p)
        _cache["d"], _cache["t"] = d, os.path.getmtime(p)


def name_of(img, folder="collected"):
    """-> (name, base) read from that picture, or ("", "")."""
    e = _load().get(folder + "/" + img) if img else None
    return (e.get("n", ""), e.get("b", "")) if isinstance(e, dict) else ("", "")


def fill(items, folder="collected", key="name"):
    """Put the name read from each item's picture in place of what the helper called it ('레어',
    '유니크', '레어 볼트' ...); that word stays in "q"."""
    for x in items:
        if x.get("q") is None:
            n, b = name_of(x.get("img", ""), folder)
            if n:
                x["q"] = x.get(key) or x.get("quality") or ""
                x[key] = n
                if b:
                    x["base"] = b
    return items


def _prepare(src, dst):
    """The info box as dark text on white, twice the size (what OCR reads best). The whole box goes in:
    the name is picked from the lines that come back (see _pick)."""
    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    text = img.max(axis=2) > 105
    text[:, : int(w * 0.03)] = False                 # the box's frame / what is beside it
    text[:, int(w * 0.97):] = False
    out = np.where(text, 0, 255).astype(np.uint8)
    out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    out = cv2.copyMakeBorder(out, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", out)
    if ok:
        buf.tofile(dst)
    return ok


HANGUL = re.compile(r"[\uac00-\ud7a3]")


def _pick(lines):
    """-> (name, base): the first line with real words is the name; the next one is the base type when it
    is words only (no numbers, no ':'), as on uniques, sets and rares."""
    good = [t for t in lines if len(HANGUL.findall(t)) >= 2]
    if not good:
        return "", ""
    name = good[0]
    base = good[1] if len(good) > 1 and not re.search(r"[0-9:%+]", good[1]) else ""
    return name, base


def _todo(limit):
    d = _load()
    out = []
    for folder, rx in (("inv_now", INV_RE), ("collected", ITEM_RE)):
        base = paths.p(folder)
        try:
            names = sorted((n for n in os.listdir(base) if rx.match(n)), reverse=True)[:400]
        except OSError:
            continue
        for n in names:
            if folder + "/" + n not in d:
                out.append((folder, n))
                if len(out) >= limit:
                    return out
    return out


def step(log, limit=20):
    """Read the names of up to `limit` pictures not read yet. -> how many were read, or -1 when Windows OCR
    is not available (no Korean OCR on this PC)."""
    if os.name != "nt":
        return 0
    todo = _todo(limit)
    if not todo:
        return 0
    work = paths.p("ocr_tmp")
    os.makedirs(work, exist_ok=True)
    plan = {}
    for i, (folder, n) in enumerate(todo):
        dst = os.path.join(work, "n%02d.png" % i)
        if _prepare(os.path.join(paths.p(folder), n), dst):
            plan[os.path.abspath(dst)] = folder + "/" + n
    d = dict(_load())
    for folder, n in todo:
        d.setdefault(folder + "/" + n, {})            # a picture that cannot be prepared is not tried again
    if plan:
        lst, res, ps1 = (os.path.join(work, x) for x in ("list.txt", "out.json", "ocr.ps1"))
        with open(lst, "w", encoding="utf-8") as f:
            f.write("\n".join(plan))
        with open(ps1, "w", encoding="utf-8-sig") as f:
            f.write(PS)
        try:
            os.remove(res)
        except OSError:
            pass
        try:
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, lst, res],
                           timeout=120, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            log("아이템 이름 읽기(윈도우 OCR) 실행 오류: %s" % e)
            return -1
        try:
            with open(res, "r", encoding="utf-8-sig") as f:
                raw = f.read().strip()
        except OSError:
            log("아이템 이름 읽기(윈도우 OCR): 결과가 없음")
            return -1
        if raw == "NOKO":
            return -1
        try:
            got = json.loads(raw) if raw else {}
        except ValueError:
            got = {}
        for p, key in plan.items():
            lines = got.get(p) or got.get(p.replace("/", "\\")) or []
            if isinstance(lines, str):
                lines = [lines]
            lines = [re.sub(r"\s+", " ", str(t)).strip() for t in lines if str(t).strip()]
            n, b = _pick(lines)
            d[key] = {"n": n, "b": b}
    _save(d)
    return len(plan)


def start(log):
    """Background thread: a few pictures every 15 s."""
    warned, told = [False], [False]

    def run():
        quiet_until = 0.0
        while True:
            time.sleep(15)
            if time.time() < quiet_until:
                continue
            try:
                n = step(log)
            except Exception as e:
                log("아이템 이름 읽기 오류: %s" % e)
                n = -1
            if n < 0:
                quiet_until = time.time() + 300           # tried again in 5 minutes (it may be installed now)
                if not warned[0]:
                    warned[0] = True
                    log("아이템 이름 읽기: 윈도우 한국어 OCR을 쓸 수 없음 (설정 > 시간 및 언어 > 언어에서 한국어 "
                        "'광학 문자 인식' 설치 확인, 설치하면 5분 안에 다시 시도)")
            elif n > 0:
                if warned[0] or not told[0]:
                    log("아이템 이름 읽기: 윈도우 한국어 OCR로 아이템 이름 %d개 읽음" % n)
                warned[0] = False
                told[0] = True
    threading.Thread(target=run, daemon=True).start()
