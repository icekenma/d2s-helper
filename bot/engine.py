# -*- coding: utf-8 -*-
"""Watch loop: grab the game window, test rules in order, act on the first hit."""
import json
import os
import threading
import time

import cv2
import mss
import numpy as np
from pynput.keyboard import Controller as KB, Key
from pynput.mouse import Button, Controller as Mouse

from . import runlog, sorter, vision, win
from .gameio import GameIO

SPECIAL = {
    "enter": Key.enter, "esc": Key.esc, "space": Key.space, "tab": Key.tab,
    "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
}
for _i in range(1, 13):
    SPECIAL["f%d" % _i] = getattr(Key, "f%d" % _i)


STAGE_LIMIT = 240         # seconds in one stage, then always on to the next run (Esc -> 저장 후 나가기)
FAST_LIMIT = 160          # the same in 패스트 모드 (2:40)


def limit_text(sec):
    sec = int(sec)
    return "%d분" % (sec // 60) if sec % 60 == 0 else "%d분 %d초" % (sec // 60, sec % 60)


def stage_limit(state):
    return FAST_LIMIT if (state or {}).get("fast") else STAGE_LIMIT


def grab_client(sct, rect):
    l, t, w, h = rect
    shot = sct.grab({"left": l, "top": t, "width": w, "height": h})
    return np.ascontiguousarray(np.array(shot)[:, :, :3])  # BGRA -> BGR


class Engine(threading.Thread):
    def __init__(self, cfg, base_dir, emit, lock, state=None):
        super().__init__(daemon=True)
        self.lock = lock
        self.state = state if state is not None else {}
        self.cleaned_open = False
        self.cfg = cfg
        self.base = base_dir
        self.emit = emit          # emit(kind, payload)
        self.stop_flag = threading.Event()
        self.io = GameIO(cfg.get("window_title", "D2 Survivors"))
        self.last_fire = {}
        self.templates = {}
        self.end_snapped = False
        self.end_sorted = False
        self.last_enter = 0.0
        self.last_free = 0.0
        self.to_town = False
        self.panel_esc = 0.0
        self.exit_at = 0.0
        self.focus_at, self.focus_note = 0.0, 0.0
        self.clock_at, self.clock_prev = 0.0, None
        self.out_since = None                     # since when the run info at the top right is gone
        self.stage_at = 0.0
        self.need_out = False                     # left on purpose: the clock waits for town / loading first
        self.state["stage_in"] = None             # when this run's stage was entered (3-minute limit)

    def stop(self):
        self.stop_flag.set()

    def log(self, msg):
        if not msg.startswith(("게임 창이 맨 앞이 아님", "(", "게임 창을 자동으로", "최소화된 게임 창")):
            self.state["progress_at"] = time.time()
        self.emit("log", msg)

    def _load(self):
        for r in self.cfg["rules"]:
            p = os.path.join(self.base, "templates", r["template"])
            img = vision.load_template(p) if os.path.exists(p) else None
            if img is None:
                self.log("그림 파일이 없어 건너뜀: %s" % r["name"])
            self.templates[r["name"]] = img

    def _tap(self, key):
        self.io.key(key)

    def _run_end(self, frame, fg, dry):
        """Built-in: on the run-end screen save the result, optionally tidy up at Charsi, then press Enter."""
        try:
            norm = sorter.Geo(frame).normalize(frame)
            score = sorter.run_ended(norm, self.base)
        except Exception:
            return False
        now = time.time()
        if score < 0.5:
            want_full = self.cfg.get("free_on_full") or (self.cfg.get("stage_end") and self.cfg.get("auto_clean"))
            if want_full and fg and not dry and now > self.last_free \
                    and sorter.inventory_full(norm, self.base) >= 0.55:
                if self.state.get("over"):
                    # the hunt is over: the collector walks to the smith instead of selling here
                    self.state["inv_full"] = now
                    self.last_free = now + 5
                    self.log("인벤토리가 가득 참: 대장장이에게 가서 정리")
                elif self.cfg.get("free_on_full") and self.lock.acquire(False):
                    io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
                    try:
                        self.log("인벤토리가 가득 참: C로 인벤토리를 열고 공간 확보")
                        self._tap("c")
                        time.sleep(0.8)
                        text, sold = sorter.free_space(io, self.cfg.get("sort", {}), self.base, self.log,
                                                       self.stop_flag)
                        self.log(text)
                        if sorter.inventory_open(io):
                            self._tap(Key.esc)
                            time.sleep(0.4)
                        self.last_free = time.time() + (5 if sold else 90)
                        if sold:
                            self.emit("freed", None)
                    except Exception as e:
                        self.log("공간 확보 중 오류: %s" % e)
                        self.last_free = time.time() + 30
                    finally:
                        io.close()
                        self.lock.release()
            if self.cfg.get("auto_clean") and fg and not dry:
                if sorter.seen(norm, "panel") < 0.6:
                    self.cleaned_open = False
                elif (self.state.get("over") and now - self.state.get("want_smith", 0) > 4
                      and not self.state.get("route_done") and not self.cleaned_open):
                    # ran into the smith while moving about: not a visit, just close it and go on
                    if now - self.panel_esc > 1.0 and self.lock.acquire(False):
                        try:
                            self.panel_esc = now
                            if not self._restore_worn():
                                self._tap(Key.esc)
                                self.log("이동 중 대장장이 창이 열려 닫고 계속 이동")
                        finally:
                            self.lock.release()
                elif self.cleaned_open and fg and now - self.panel_esc > 2.5 and self.lock.acquire(False):
                    # tidied already but the window is still open (an Esc went missing): close it again
                    try:
                        self.panel_esc = now
                        if not self._restore_worn():
                            self._tap(Key.esc)
                            self.log("대장장이 창이 아직 열려 있어 다시 닫음")
                    finally:
                        self.lock.release()
                elif not self.cleaned_open and self.lock.acquire(False):
                    self.cleaned_open = True
                    io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
                    try:
                        # every time the window opens: an empty inventory makes this a quick no-op
                        self.log("대장장이 창 확인: 자동 정리 시작")
                        halted = []
                        sorter.restore_worn(io, self.log, lambda: (halted.append(1), self.emit("halt", None)))
                        if halted:
                            return True
                        text, done = sorter.cleanup(io, self.cfg.get("sort", {}), self.base, self.log,
                                                    self.stop_flag)
                        self.log(text)
                        if done:
                            self.state["cleaned"] = time.time()
                        if self.cfg.get("stage_end") and not self.stop_flag.is_set():
                            self._tap(Key.esc)
                            time.sleep(0.5)
                            # step off the smith so the window does not pop straight back up
                            back = {"up": Key.down, "down": Key.up, "left": Key.right, "right": Key.left}
                            keys = [back[k] for k in self.state.get("npc_dir", ()) if k in back] or [Key.down]
                            for k in keys:
                                self.io.key_down(k)
                            time.sleep(0.9)
                            for k in keys:
                                self.io.key_up(k)
                    except Exception as e:
                        self.log("자동 정리 중 오류: %s" % e)
                    finally:
                        io.close()
                        self.lock.release()
                    return True
            if self.end_snapped and now - self.last_enter > 2:
                self.end_snapped = self.end_sorted = False
            return False
        if not self.cfg.get("auto_restart", True):
            return False
        if not self.end_snapped:
            self.end_snapped = True
            self.state["run_reset"] = now
            dead = sorter.died(norm)
            self._snap({"name": "사망" if dead else "클리어"}, frame)
            self.to_town = False
            limit = int(self.cfg.get("drop_after", 0) or 0)
            target = int(self.state.get("target", 0) or 0)
            if dead:
                self.state["deaths"] = self.state.get("deaths", 0) + 1
                self.log("사망 (연속 %d번)" % self.state["deaths"])
                if limit > 0 and target > 1 and self.state["deaths"] >= limit:
                    self.state["target"] = target - 1
                    self.state["deaths"] = 0
                    self.to_town = True
                    self.emit("target", target - 1)
                    self.log("연속 %d번 사망: 공포 영역 %d → %d로 낮춤" % (limit, target, target - 1))
            else:
                self.state["deaths"] = 0
                self.log("클리어")
            if self.cfg.get("always_town"):
                # a fixed stage: the end screen's Enter could start another one (a death lowers it),
                # so always go back to town and pick the stage at the waypoint
                self.to_town = True
            elif self.cfg.get("town_start") and self.state.get("terror_ever") \
                    and now - self.state.get("terror_at", 0) > 20:
                # that run was not a terror zone (a wrong waypoint row): pick the stage again in town
                self.to_town = True
                self.log("방금 판은 공포 영역이 아니었음: 마을로 돌아가 웨이포인트에서 다시 고름")
        if dry or not fg or now - self.last_enter < 3:
            return True
        if not self.lock.acquire(False):
            return True
        try:
            if self.cfg.get("sort_on_end") and not self.end_sorted:
                self.end_sorted = True
                self._tap("t")
                time.sleep(1.5)
                io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
                try:
                    self.log(sorter.run(io, self.cfg.get("sort", {}), self.base, self.log, self.stop_flag))
                finally:
                    io.close()
                self._tap(Key.esc)
                time.sleep(1.0)
                return True
            if self.to_town:
                self._tap(Key.esc)          # back to town; the waypoint step picks the lower stage
                self.state["town_check"] = now + 40
                self.last_enter = now
                self.emit("restart", None)
                self.log("Esc로 마을 이동")
            else:
                self._tap(Key.enter)
                self.last_enter = now
                self.emit("restart", None)
                self.log("Enter로 다시 시작")
        except Exception as e:
            self.log("다시 시작 중 오류: %s" % e)
        finally:
            self.lock.release()
        return True

    def _clock(self, frame, now):
        """The run timer at the top ("2:09 / 10:00") ticking means a hunt is under way: that is progress
        even though nothing is logged. A frozen timer, a town, a menu or any unknown screen is not."""
        if frame is None or now - self.clock_at < 3.0:
            return
        self.clock_at = now
        try:
            geo = sorter.Geo(frame)
            n = geo.normalize(frame)
            g = cv2.cvtColor(n[8:38, 290:385], cv2.COLOR_BGR2GRAY)
            digits = (g > 150).astype(np.uint8)
        except Exception:
            return
        if digits.sum() < 40:                         # no timer on screen
            self.clock_prev = None
            return
        if self.clock_prev is not None and int(np.abs(digits.astype(int) - self.clock_prev).sum()) > 25 \
                and not self.state.get("over"):
            self.state["progress_at"] = now
        self.clock_prev = digits.astype(int)

    def _restore_worn(self):
        io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
        try:
            return sorter.restore_worn(io, self.log, lambda: self.emit("halt", None))
        except Exception:
            return False
        finally:
            io.close()

    def _stage_clock(self, frame, now):
        """Start the 3-minute clock when a stage is entered (the run info appears at the top right),
        clear it in town / on the loading screen / when the run restarts."""
        st = self.state
        si = st.get("stage_in")
        if si and st.get("run_reset", 0) > si:
            st["stage_in"] = si = None                 # the run ended and starts over
            runlog.end(now, "다시 시작")
        if frame is None or now - self.stage_at < 0.9:
            return
        self.stage_at = now
        try:
            out = sorter.out_of_stage(frame)
        except Exception:
            return
        if out:
            if self.out_since is None:
                self.out_since = now
            elif now - self.out_since >= 1.5:
                if st.get("stage_in"):
                    runlog.end(self.out_since, "사냥터를 나감")
                st["stage_in"] = None                  # town or loading: the next stage starts the clock again
                self.need_out = False
            return
        self.out_since = None
        if not si and not self.need_out:
            st["stage_in"] = now
            runlog.start(now, st.get("target", 0))
            self.log("사냥터 입장: %s 뒤에는 무조건 다음 판으로%s" % (limit_text(stage_limit(st)),
                                                                 " (패스트 모드)" if st.get("fast") else ""))

    def _stuck(self, fg, dry, frame=None):
        """Last resort: nothing has happened for 2 minutes after the boss died (or 3 minutes have gone
        by in all). Esc -> "저장 후 나가기": back to town, and the town step starts a new run."""
        st = self.state
        if dry:
            return False
        now = time.time()
        st.setdefault("progress_at", now)
        self._clock(frame, now)
        quiet_all = now - st["progress_at"]
        self._stage_clock(frame, now)
        si = st.get("stage_in")
        limit = stage_limit(st)
        if si and now - si >= limit and st.get("run_text_at") and now - st["run_text_at"] > 10:
            # the run info ('N막 - 공포 영역 ...') has been gone for a while: already in town, where
            # '저장 후 나가기' would drop to the game's main menu. Nothing to leave.
            st["stage_in"] = si = None
            runlog.end(float(st.get("run_text_at") or now), "사냥터를 나감")
            self.log("%s 제한: 이미 사냥터 밖(판 정보 없음)이라 나가지 않음" % limit_text(limit))
        if si and now - si >= limit:
            due, why = True, "사냥터에 들어간 지 %s 지남 (무조건 다음 판)" % limit_text(limit)
        elif st.get("over"):
            over_at = st.get("over_at", now)
            quiet = now - max(st["progress_at"], over_at)
            due = quiet >= 120 or now - over_at >= 240 + float(st.get("route_len", 0) or 0)
            why = "보스 처치 후 %d초 동안 진행이 없음" % int(max(quiet, now - over_at))
        else:
            # anywhere else: no log line and no ticking run timer for 2 minutes means something is wrong
            due, why = False, ""           # the general 2-minute check is off at the user's request
            tc = st.get("town_check")
            if tc and frame is not None:
                if sorter.out_of_stage(frame):
                    st["town_check"] = None              # town (or loading) reached
                elif now > tc:
                    st["town_check"] = None
                    due, why = True, "마을로 가려 했는데 40초 뒤에도 사냥터에 있음"
        if not due or now - self.exit_at < 30:
            return False
        if not self.lock.acquire(timeout=3):
            return False
        self.exit_at = now
        io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
        try:
            self.log(why + ": Esc -> 저장 후 나가기로 마을로")
            st["progress_at"] = now - 115          # if it fails, try again soon (30 s later)
            for attempt in range(3):
                if sorter.leave_via_menu(io, self.log, lambda: self._tap(Key.esc),
                                         {"down": Key.down, "up": Key.up, "enter": Key.enter, "tab": Key.tab},
                                         attempt=attempt + 1):
                    st["over"] = False
                    st["run_reset"] = time.time()
                    st["progress_at"] = time.time()
                    st["stage_in"] = None
                    runlog.end(time.time(), why)
                    self.need_out = True
                    st["town_check"] = time.time() + 40      # still in the stage 40 s later: try again
                    self.emit("restart", None)
                    self.log("저장 후 나가기 클릭")
                    return True
                # not the menu (a window was open, or the menu closed again): Esc once more
            self.log("Esc 메뉴의 '저장 후 나가기'를 찾지 못함: 30초 뒤 다시 시도")
            st["progress_at"] = time.time() - 115
        except Exception as e:
            self.log("저장 후 나가기 중 오류: %s" % e)
        finally:
            io.close()
            self.lock.release()
        return True

    def _snap(self, rule, frame):
        try:
            d = os.path.join(self.base, "snaps")
            os.makedirs(d, exist_ok=True)
            fn = time.strftime("%Y%m%d_%H%M%S") + "_%03d.jpg" % (int(time.time() * 1000) % 1000)
            h, w = frame.shape[:2]
            img = frame if w <= 1600 else cv2.resize(frame, (1600, int(h * 1600.0 / w)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            buf.tofile(os.path.join(d, fn))
            with open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": fn, "rule": rule["name"],
                                    "time": time.strftime("%m-%d %H:%M")}, ensure_ascii=False) + "\n")
            files = sorted(x for x in os.listdir(d) if x.endswith(".jpg"))
            for old in files[:-300]:
                os.remove(os.path.join(d, old))
        except Exception as e:
            self.log("화면 저장 실패: %s" % e)

    def _act(self, rule, rect, pos):
        act = rule.get("action", "click")
        if act == "click":
            self.io.locate()
            self.io.click(pos[0] + int(rule.get("dx", 0)), pos[1] + int(rule.get("dy", 0)))
        elif act == "key":
            name = str(rule.get("key", "")).strip().lower()
            k = SPECIAL.get(name, name[:1])
            if k:
                self.io.key(k)
        elif act == "sort":
            io = GameIO(self.cfg.get("window_title", "D2 Survivors"))
            try:
                self.log(sorter.run(io, self.cfg.get("sort", {}), self.base, self.log, self.stop_flag))
            except Exception as e:
                self.log("정리 중 오류: %s" % e)
            finally:
                io.close()
        # act == "none": count only

    def run(self):
        self._load()
        title = self.cfg.get("window_title", "D2 Survivors")
        interval = float(self.cfg.get("interval", 0.5))
        dry = bool(self.cfg.get("dry_run", False))
        want = self.cfg.get("client_size")
        warned = ""
        self.log("감시 시작" + (" (감지만, 클릭 안 함)" if dry else ""))
        with mss.mss() as sct:
            while not self.stop_flag.is_set():
                t0 = time.time()
                if not self.io.locate():
                    hwnd = win.find_window(title)
                    state = ("게임 창을 찾지 못함: '%s'" % title) if not hwnd else "게임 창이 최소화되어 있음"
                    if state != warned:
                        self.log(state)
                        warned = state
                    self.stop_flag.wait(1.0)
                    continue
                hwnd, rect = self.io.hwnd, self.io.rect
                if want and (rect[2], rect[3]) != tuple(want):
                    win.resize_client(hwnd, want[0], want[1])
                    self.log("창 크기를 등록 당시 크기(%dx%d)로 맞춤" % tuple(want))
                    self.stop_flag.wait(0.7)
                    continue
                try:
                    frame = self.io.grab()        # the real screen, or PrintWindow when the game is behind
                except Exception as e:
                    frame = None
                    if warned != "cap":
                        self.log("화면 캡처 실패: %s" % e)
                if frame is None:
                    if warned != "cap":
                        self.log("게임 화면을 읽지 못함 (창이 가려져 그림을 멈췄을 수 있음) - 계속 다시 시도")
                        warned = "cap"
                    self.stop_flag.wait(1.0)
                    continue
                if warned:
                    warned = ""
                fg = True                         # the game can be worked whether it is in front or not
                scores = {}
                fired = False
                now = time.time()
                if self._stuck(fg, dry, frame):
                    self.stop_flag.wait(1.0)
                    continue
                if ((self.cfg.get("auto_restart", True) or self.cfg.get("free_on_full") or self.cfg.get("auto_clean")
                     or self.cfg.get("stage_end"))
                        and self._run_end(frame, fg, dry)):
                    self.emit("scores", scores)
                    self.stop_flag.wait(max(0.05, interval - (time.time() - t0)))
                    continue
                for r in self.cfg["rules"]:
                    tm = self.templates.get(r["name"])
                    if tm is None or not r.get("enabled", True):
                        continue
                    sc, pos = vision.find(frame, tm)
                    scores[r["name"]] = sc
                    if fired or sc < float(r.get("threshold", 0.85)):
                        continue
                    if now - self.last_fire.get(r["name"], 0) < float(r.get("cooldown", 2)):
                        continue
                    if dry or not fg:
                        continue
                    if not self.lock.acquire(False):
                        continue          # the inventory sorter is using the mouse
                    try:
                        if r.get("snap"):
                            self._snap(r, frame)
                        self._act(r, rect, pos)
                    finally:
                        self.lock.release()
                    self.last_fire[r["name"]] = now
                    fired = True
                    self.emit("fired", r["name"])
                    self.log("실행: %s (일치도 %.2f)" % (r["name"], sc))
                self.emit("scores", scores)
                self.stop_flag.wait(max(0.05, interval - (time.time() - t0)))
        self.log("감시 중지")
