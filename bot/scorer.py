# -*- coding: utf-8 -*-
"""Item score out of 100 for unique and set items, against original Diablo II roll ranges."""
import json
import re
import os
import shutil
import subprocess
import tempfile
import threading
from difflib import SequenceMatcher

import cv2
import numpy as np

from . import spot, tipread

_db = []
_tess = [None]
_data = [None]       # folder holding our own kor.traineddata
_check = [""]        # result of the start-up reading test
QUALITY_KO = {"gold": "유니크", "green": "세트", "blue": "매직", "yellow": "레어", "white": "일반"}
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def jamo(text):
    out = []
    for ch in text:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out.append(CHO[o // 588] + JUNG[o % 588 // 28] + JONG[o % 28])
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def _plain_dirs():
    """Folders whose path has only plain letters (Tesseract on Windows cannot open Korean paths)."""
    out = []
    for root in (os.environ.get("PROGRAMDATA", ""), os.environ.get("PUBLIC", ""), "C:\\Temp" if os.name == "nt" else ""):
        if not root:
            continue
        try:
            d = os.path.join(root, "d2s_helper")
            os.makedirs(d, exist_ok=True)
            out.append(d)
        except OSError:
            continue
    return out


def load(base, tess_path=""):
    load_keep(base)
    with open(os.path.join(base, "assets", "items_kr.json"), "r", encoding="utf-8") as f:
        _db[:] = json.load(f)
    for it in _db:
        it["j"] = jamo(it["n"])
        it["jb"] = jamo(it.get("b", ""))
    tipread.load_glyphs(base)
    local = os.environ.get("LOCALAPPDATA", "")
    cands = [tess_path, shutil.which("tesseract") or "",
             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    if local:
        cands += [os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe"),
                  os.path.join(local, "Tesseract-OCR", "tesseract.exe")]
    _tess[0] = next((c for c in cands if c and os.path.isfile(c)), None)
    _data[0], _mode[0], _check[0], _err[0] = None, None, "", ""
    if not _tess[0]:
        return None
    ours = os.path.join(base, "assets", "tessdata")
    plain = _plain_dirs()
    datas = []                                     # (label, folder) to try, best first
    if os.path.isfile(os.path.join(ours, "kor.traineddata")):
        for d in plain[:1]:
            try:
                dst = os.path.join(d, "tessdata")
                os.makedirs(dst, exist_ok=True)
                target = os.path.join(dst, "kor.traineddata")
                if not os.path.isfile(target) or os.path.getsize(target) != os.path.getsize(
                        os.path.join(ours, "kor.traineddata")):
                    shutil.copyfile(os.path.join(ours, "kor.traineddata"), target)
                datas.append(("복사한 데이터", dst))
            except OSError:
                pass
        datas.append(("내장 데이터", ours))
    datas.append(("설치된 데이터", None))
    modes = []                                     # how the picture reaches Tesseract
    for d in plain[:1]:
        modes.append(("파일", os.path.join(d, "tmp")))
    modes.append(("임시 파일", tempfile.gettempdir()))
    modes.append(("직접 전달", None))
    try:
        img = cv2.imdecode(np.fromfile(os.path.join(base, "assets", "selftest.png"), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    except Exception as e:
        _check[0] = "시험 그림을 열지 못함: %s" % e
        return _tess[0]
    notes = []
    for dl, d in datas:
        for ml, m in modes:
            _data[0], _mode[0], _err[0] = d, m, ""
            got = ocr_line(img)
            if SequenceMatcher(None, jamo(got.replace(" ", "")), jamo("흐라틀리")).ratio() >= 0.7:
                _check[0] = "ok"
                _how[0] = "%s, %s" % (dl, ml)
                return _tess[0]
            notes.append("[%s/%s] '%s' %s" % (dl, ml, got, _err[0]))
            if "data file" in _err[0] or "traineddata" in _err[0]:
                break                              # this data folder is no good however the picture is sent
    _check[0] = "한글 판독 시험 실패. " + " / ".join(notes)
    return _tess[0]


_err = [""]
_mode = [None]       # folder for the hand-over file, or None to pipe the picture in
_how = [""]


def status():
    """(works, text) for the UI."""
    if not _tess[0]:
        return False, "글자 인식 프로그램(Tesseract)을 찾지 못했습니다."
    if _check[0] != "ok":
        return False, "Tesseract는 찾았지만 한글을 읽지 못합니다: %s" % _check[0]
    return True, "글자 인식 정상 (%s): %s" % (_how[0], _tess[0])


def ready():
    return _tess[0] is not None


def ocr_line(img):
    """Korean text of one text line image (work scale)."""
    if not ready():
        return ""
    path = None
    try:
        g = img.max(axis=2).astype(np.float32)
        g = np.clip(g * (255.0 / max(60.0, float(g.max()))), 0, 255).astype(np.uint8)
        g = 255 - cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000      # no console window
        if _mode[0]:
            ok, buf = cv2.imencode(".png", g)
            if not ok:
                _err[0] = "그림 변환 실패"
                return ""
            os.makedirs(_mode[0], exist_ok=True)
            path = os.path.join(_mode[0], "ocr_%d_%d.png" % (os.getpid(), threading.get_ident()))
            with open(path, "wb") as f:           # Python's own file writing copes with any folder name
                f.write(buf.tobytes())
            cmd, data_in = [_tess[0], path, "stdout"], None
        else:
            # plain-text picture (PGM "P2"): a Windows pipe in text mode would cut a PNG short at byte 0x1A
            hh, ww = g.shape
            body = "\n".join(" ".join(map(str, row)) for row in g.tolist())
            data_in = ("P2\n%d %d\n255\n%s\n" % (ww, hh, body)).encode("ascii")
            cmd = [_tess[0], "stdin", "stdout"]
        cmd += ["-l", "kor", "--psm", "7"]
        env = None
        if _data[0]:
            cmd += ["--tessdata-dir", _data[0]]
            env = dict(os.environ, TESSDATA_PREFIX=_data[0])
        r = subprocess.run(cmd, input=data_in, capture_output=True, timeout=20, env=env, **kw)
        out = r.stdout.decode("utf-8", "ignore").strip()
        if not out:
            err = r.stderr.decode("utf-8", "ignore").strip() or r.stderr.decode("cp949", "ignore").strip()
            lines = [l for l in err.splitlines() if l.strip() and "resolution" not in l.lower()]
            _err[0] = (lines[0][:200] if lines else "출력 없음 (종료 코드 %s)" % r.returncode)
        return out
    except Exception as e:
        _err[0] = "%s: %s" % (type(e).__name__, str(e)[:200])
        return ""
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def find_item(name, base, quality):
    jn, jb = jamo(name), jamo(base)
    best, hit = 0.0, None
    for it in _db:
        if it["q"] != quality:
            continue
        s = SequenceMatcher(None, jn, it["j"]).ratio()
        if s < 0.6:
            continue
        if jb and it["jb"]:
            s = 0.8 * s + 0.2 * SequenceMatcher(None, jb, it["jb"]).ratio()
        if s > best:
            best, hit = s, it
    return (hit, best) if best >= 0.72 else (None, best)


def roll_score(item, numbers):
    """-> (score or None, matched, total, exact). Tries every way to pair the read numbers
    with the item's variable properties and reports the average when they disagree."""
    pool = list(numbers)
    for f in item["fixed"]:
        if f in pool:
            pool.remove(f)
    var = item["var"]
    if not var:
        return 100.0, 0, 0, True
    results = []

    def go(i, used, acc, n):
        if i == len(var):
            results.append((n, acc))
            return
        _, lo, hi = var[i]
        hit = False
        seen_vals = set()
        for k, v in enumerate(pool):
            if k in used or v in seen_vals or not (lo <= v <= hi):
                continue
            seen_vals.add(v)
            hit = True
            go(i + 1, used | {k}, acc + (v - lo) / float(hi - lo), n + 1)
        if not hit:
            go(i + 1, used, acc, n)

    go(0, frozenset(), 0.0, 0)
    top = max(n for n, _ in results)
    if top == 0:
        return None, 0, len(var), False
    vals = [acc / top * 100.0 for n, acc in results if n == top]
    return sum(vals) / len(vals), top, len(var), max(vals) - min(vals) < 0.5


STORED = jamo("이미 하나를 보관했습니다")
KEEP_WORDS = ["정수", "주얼", "작은 부적"]
GRAND_CHARM, SKILL_WORD = "거대 부적", "기술"


def has_word(text, word, need=0.8):
    """True when one of the words in text reads like word (compared letter-part by letter-part)."""
    jw = jamo(word)
    return any(SequenceMatcher(None, jamo(w), jw).ratio() >= need for w in text.split())


def ends_with(text, phrase, need=0.8):
    """True when the name ends in phrase ("... 작은 부적"), allowing a misread letter or two."""
    words, want = text.split(), phrase.split()
    if len(words) < len(want):
        return False
    tail = jamo("".join(words[-len(want):]))
    return SequenceMatcher(None, tail, jamo("".join(want))).ratio() >= need


QTY_FIX = str.maketrans({"I": "1", "l": "1", "|": "1", "ㅣ": "1", "!": "1", "O": "0", "o": "0", "@": "0",
                         "S": "5", "B": "8"})

# the player's best items: always stashed (never registered by the game's '수집', never sold)
KEEP_NAMES = {"바람살": 0.72, "신념": 0.78, "명예의 굴레": 0.6, "수수께끼": 0.6}
KEEP_LIST = {"order": [], "classes": {}}     # assets/keep_list.json: what to keep, per character class
KEEP_SET = set()                             # every name in it
_keep_pics = []                              # (name, float32 picture of the name line, layout scale)
KEEP_PIC_NEED = 0.76
STAR = {}                                    # name -> importance 1..5 (highest over the classes)
KEEP_MIN_STAR = [1]                          # only names rated this many stars or more are kept


def set_min_star(n):
    KEEP_MIN_STAR[0] = max(1, min(5, int(n or 1)))


def _set_keep(out, nm):
    """A keep-list name was recognised: kept only when rated at least KEEP_MIN_STAR stars."""
    st = STAR.get(nm, 5)
    out["star"] = st
    if st >= KEEP_MIN_STAR[0]:
        out["keep_name"], out["kind"] = nm, nm


def load_keep(base):
    KEEP_SET.clear()
    _keep_pics[:] = []
    try:
        with open(os.path.join(base, "assets", "keep_list.json"), "r", encoding="utf-8") as f:
            KEEP_LIST.update(json.load(f))
    except (OSError, ValueError):
        pass
    STAR.clear()
    for items in KEEP_LIST.get("classes", {}).values():
        for it in items:
            KEEP_SET.add(it["n"])
            STAR[it["n"]] = max(STAR.get(it["n"], 0), int(it.get("star", 5)))
    KEEP_SET.update(KEEP_NAMES)
    try:
        with open(os.path.join(base, "assets", "keep_names.json"), "r", encoding="utf-8") as f:
            for name, rows in json.load(f):
                _keep_pics.append((name, np.array([[int(c) for c in r] for r in rows], np.float32) / 9.0))
    except (OSError, ValueError):
        pass


def keep_pictures():
    """Names that can be told from their picture alone (no OCR needed)."""
    return {n for n, _ in _keep_pics} | set(KEEP_NAMES)


def _line_pic(img, l):
    """A text line as a soft picture at layout scale (the same at any window size)."""
    c = img[max(0, l["y0"] - 3):l["y1"] + 3, max(0, l["box"][0] - 4):l["box"][1] + 4].astype(np.float32)
    g = np.clip((c.max(axis=2) - 60.0) / 140.0, 0, 1)
    g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (0, 0), 0.6)


def keep_by_picture(img, lines):
    """-> (name, score): the keep-list name whose picture best matches the name (or base) line."""
    best = (None, 0.0)
    for l in lines[:2]:
        if l["color"] not in ("gold", "green"):
            continue
        q = _line_pic(img, l)
        for n, t in _keep_pics:
            if abs(t.shape[1] - q.shape[1]) > 8:
                continue
            r = q
            if r.shape[0] < t.shape[0] or r.shape[1] < t.shape[1]:
                r = cv2.copyMakeBorder(r, 0, max(0, t.shape[0] - r.shape[0] + 1), 0,
                                       max(0, t.shape[1] - r.shape[1] + 1), cv2.BORDER_CONSTANT, value=0)
            v = cv2.minMaxLoc(cv2.matchTemplate(r, t, cv2.TM_CCOEFF_NORMED))[1]
            if np.isfinite(v) and v > best[1]:
                best = (n, float(v))
    return best


def keep_by_text(text):
    """An OCR'd name -> the keep-list name it reads as, or None."""
    j = jamo(text.replace(" ", ""))
    if not j:
        return None
    best, hit = 0.0, None
    for n in KEEP_SET:
        v = SequenceMatcher(None, j, jamo(n.replace(" ", ""))).ratio()
        if v > best:
            best, hit = v, n
    return hit if best >= 0.86 else None


def evaluate(crop, s, keep_words=None):
    """crop: raw BGR image of the info box, s: window scale (pixels per layout unit).
    -> dict: quality, color, kind, keep, charm (decided from word pictures, no OCR needed) and, when
       Tesseract works, name / raw / score / stored as well."""
    lines, img = tipread.read(crop, s)
    lines = [l for l in lines if l["color"] != "none" and l["words"] > 0]
    while lines and lines[0]["color"] == "dim":
        lines = lines[1:]
    if not lines:
        return None
    color = lines[0]["color"]
    out = {"quality": QUALITY_KO.get(color, "기타"), "color": color, "name": "", "raw": "", "score": None,
           "note": "", "stored": False, "top": False, "keep": False, "charm": False, "kind": ""}
    mask = tipread.binarize(img).astype(np.float32)

    def rows(l):
        return mask[max(0, l["y0"] - 2):l["y1"] + 2]

    head = [rows(l) for l in lines[:2] if l["color"] == color]        # name line and base line
    word = lambda n: any(spot.has(m, n, 0.68) for m in head)
    is_charm = word("부적")
    for nm, need in KEEP_NAMES.items():
        if nm in spot._words and any(spot.score(m, nm) >= need for m in head):
            _set_keep(out, nm)
            break
    if not out.get("keep_name") and color in ("gold", "green") and _keep_pics:
        try:
            nm, v = keep_by_picture(img, lines)
        except Exception:
            nm, v = None, 0.0
        if nm and v >= KEEP_PIC_NEED:
            _set_keep(out, nm)
    if color in ("gold", "green"):
        out["charm"] = is_charm
        if not out.get("keep_name"):
            out["kind"] = "유니크 부적" if is_charm else out["quality"]
    elif color == "orange":
        out["kind"], out["keep"] = "룬", True
    else:
        if spot.has(rows(lines[0]), "정수", 0.68):
            out["kind"], out["keep"] = "정수", True
        elif "열쇠" in spot._words and spot.has(rows(lines[0]), "열쇠", 0.68):
            out["kind"], out["keep"] = "열쇠", True
        elif is_charm and word("작은"):
            out["kind"], out["keep"] = "작은 부적", True
        elif is_charm and word("거대"):
            out["kind"], out["keep"] = "거대 부적", True
        elif spot.has(rows(lines[0]), "주얼", 0.68):
            out["kind"], out["keep"] = "주얼", True
        elif is_charm:
            out["kind"] = "큰 부적"
        elif color == "yellow" and "화살" in spot._words and word("화살"):
            out["kind"], out["keep"] = "레어 화살", True
    if len(lines) > 1:                         # '수량: N' right under the name (essences stack in one cell);
        l = lines[1]                           # the line after it is the sale price, never the count
        if l.get("label") and l["numbers"] and l["color"] == "white":
            out["qty"] = l["numbers"][0]
    if status()[0]:
        def text(l):
            return ocr_line(img[max(0, l["y0"] - 3):l["y1"] + 3, max(0, l["box"][0] - 4):l["box"][1] + 4])

        name = text(lines[0])
        out["raw"] = name
        if color in ("gold", "green") and not out.get("keep_name"):
            nm = keep_by_text(name)
            if nm:
                _set_keep(out, nm)
        if color not in ("gold", "green"):
            if not out["keep"] and (ends_with(name, "룬") or ends_with(name, "정수") or has_word(name, "열쇠")
                                     or has_word(name, "주얼")):
                out["keep"] = True
            if not out["kind"] and ends_with(name, "정수"):
                out["kind"] = "정수"
            if out["kind"] == "정수" and out.get("qty") is None and len(lines) > 1:
                m = re.search(r"(\d+)", text(lines[1]).split(":")[-1].translate(QTY_FIX))
                if m:
                    out["qty"] = int(m.group(1))
            if not out["keep"] and color == "yellow" and len(lines) > 1:
                base = text(lines[1])
                if "화살" in base or "화살" in name:
                    out["kind"], out["keep"] = "레어 화살", True
            return out
        base = text(lines[1]) if len(lines) > 1 and lines[1]["color"] == color else ""
        for l in lines[1:4]:
            if l["color"] == "dim" and SequenceMatcher(None, jamo(text(l)), STORED).ratio() >= 0.6:
                out["stored"] = True
                break
        item, sim = find_item(name, base, "unique" if color == "gold" else "set")
        if item is None:
            out["note"] = "이름을 목록에서 찾지 못함"
            return out
        out["name"] = item["n"]
        if item["n"] in KEEP_SET and not out.get("keep_name"):
            _set_keep(out, item["n"])
        out["top"] = bool(item.get("top"))
        out["charm"] = out["charm"] or bool(item.get("charm"))
        nums = [n for l in lines[1:] if l["color"] in ("blue", "red") and not l.get("label") for n in l["numbers"]]
        score, hit, total, exact = roll_score(item, nums)
        if score is None:
            out["note"] = "옵션 수치를 읽지 못함"
            return out
        out["score"] = int(round(score))
        if total == 0:
            out["note"] = "옵션이 고정인 아이템"
        elif hit < total:
            out["note"] = "변동 옵션 %d개 중 %d개만 반영" % (total, hit)
        elif not exact:
            out["note"] = "수치 대응이 여러 가지라 평균값"
    return out
