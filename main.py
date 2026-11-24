# -*- coding: utf-8 -*-
"""D2S Helper - plays D2 Survivors runs back to back by a fixed set of rules (personal, single-player use).
No settings: start, stop, and a record of what happened."""
import json
import os
import queue
import random
import re
import threading
import time
import tkinter as tk
import subprocess
from tkinter import messagebox, ttk

from pynput import keyboard

from bot import scorer, sorter, spot, win
from bot.engine import Engine
from bot.keyrec import KeyRecorder
from bot.picker import PickerThread
from bot.record import Recorder
from bot.remote import Remote
from bot import paths
from bot.updater import Updater, local_version

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
MOVED = paths.migrate(BASE)                            # data kept across versions
CFG_PATH = paths.p("config.json")                      # phone PIN / port / stage / radius
DEFAULT_PIN = "882762"                                 # the address the phone already knows

TITLE = "D2 Survivors"
# the rules, fixed
ENGINE_CFG = {"window_title": TITLE, "interval": 0.5, "dry_run": False, "client_size": None, "rules": [],
              "auto_restart": True, "sort_on_end": False, "free_on_full": False, "auto_clean": True,
              "stage_end": True, "town_start": True, "drop_after": 0,
              "sort": {"actions": {c: "keep" for c in sorter.COLORS}, "stash_tab": "개인", "save_tips": False,
                       "min_score": 60, "keep_min": 0, "keep_words": "정수", "swap_better": False}}
PICK_OPTS = {"set": True, "unique": True, "jewel": True, "stage_end": True, "auto_clean": True,
             "town_start": True, "town_stage": 0, "collect_after": True, "mouse_move": True, "over_limit": 100}
RULES = [
    ("마을", "웨이포인트를 눌러 위에서 정한 공포 영역 단계로 들어갑니다(비워 두면 목록 맨 위의 새로 열린 단계)."),
    ("사냥", "보스가 죽어 '차원문이 열렸습니다'가 뜰 때까지 움직이지 않습니다."),
    ("줍기", "보스 처치 직후 화면의 정수부터 줍고, 보스 처치 때 보이던 화면 안을 촘촘한 줄로 오가며 달려 줍습니다"
             "(정수·열쇠·부적 이름이 보이면 바로 집음). 다 돌면 미니맵의 유니크·세트·주얼 표시로 가서 더 줍습니다"
             "(미니맵이 꺼져 있으면 Tab으로 켬, 범위는 위 칸). 포탈은 밟거나 누르지 않게 피해 다닙니다. 전체 100초 안. "
             "방향키 녹음이 있으면 위 대신: 보스 처치 3초 뒤 녹음대로만 이동하고 바로 정리·다음 판."),
    ("정리", "인벤토리에 무언가 있을 때만 대장장이에게 갑니다. 새 유니크·세트는 등록, 이미 등록된 것은 판매"
             "(유니크 부적은 보관). 정수·주얼·룬·작은 부적·거대 부적·레어 화살은 보관함(개인 → 공유 1~3). 나머지는 판매."),
    ("다음 판", "창고 정리를 했으면 Esc -> '저장 후 나가기'로 마을에 가서 웨이포인트로 다음 판, 정리할 게 없었으면 포탈을 탑니다. 죽거나 판이 끝나면 Esc로 마을에 가서 같은 단계를 다시 고릅니다. 어디서든 막히면 "
                "보스 처치 85초 뒤 창을 닫고 무조건 포탈, 마을에서 45초 막히면 창을 닫고 다시 시도합니다. 보스 처치 후 2분 동안 진행이 없으면 Esc 메뉴의 '저장 후 나가기'로 마을에 나갑니다."),
]

# palette: charcoal stone, parchment text, belt gold, life-orb red
BG, PANEL, LINE = "#1b1714", "#26201b", "#3a3027"
TEXT, DIM, GOLD, RED, GREEN = "#e6dcc6", "#9a8d78", "#c9a24a", "#b3362c", "#7fae5a"
FONT = "맑은 고딕"


def load_cfg():
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cfg(cfg):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = load_cfg()
        self.q = queue.Queue()
        self.engine = self.picker = None
        self.runs = 0
        self.started_at = None
        self.last_line = 0.0
        self.logbuf = []
        self.shared = {}
        self.io_lock = threading.Lock()
        self.rec = Recorder(BASE, TITLE)
        self.keys = KeyRecorder(paths.p("route.json"))
        self.title("D2S Helper")
        self.geometry("820x700")
        self.minsize(680, 560)
        self.configure(bg=BG)
        self._style()
        sorter.load_marks(BASE)
        spot.load(BASE)
        scorer.load(BASE, "")
        self._build()
        self._share()
        self._remote()
        self.hot = keyboard.GlobalHotKeys({"<f7>": lambda: self.q.put(("rec", None)),
                                           "<f8>": lambda: self.q.put(("toggle", None)),
                                           "<f9>": lambda: self.q.put(("halt", None))})
        self.hot.daemon = True
        self.hot.start()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.restarting, self.upd_seen = False, None
        self.upd = Updater(BASE, lambda m: self.q.put(("log", m)))
        self.upd.start()
        self.title("D2S Helper v%s" % local_version(BASE))
        self.after(100, self._pump)
        if self.cfg.pop("resume", False):            # restarted by an update while it was running
            save_cfg(self.cfg)
            self.after(2500, self.toggle)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=(FONT, 10), bordercolor=LINE,
                    lightcolor=LINE, darkcolor=LINE, troughcolor=PANEL)
        s.configure("TButton", background=PANEL, foreground=TEXT, padding=(12, 6), relief="flat")
        s.map("TButton", background=[("active", LINE)])
        s.configure("Gold.TButton", background=GOLD, foreground="#1b1714", font=(FONT, 11, "bold"), padding=(18, 8))
        s.map("Gold.TButton", background=[("active", "#dbb65c")])
        s.configure("Stop.TButton", background=RED, foreground=TEXT, font=(FONT, 11, "bold"), padding=(18, 8))
        s.map("Stop.TButton", background=[("active", "#c7463b")])
        s.configure("TEntry", fieldbackground=PANEL, foreground=TEXT)

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(18, 12))
        self.l_runs = tk.Label(top, text="0", bg=BG, fg=GOLD, font=(FONT, 34, "bold"))
        self.l_runs.pack(side="left")
        info = tk.Frame(top, bg=BG)
        info.pack(side="left", padx=12)
        tk.Label(info, text="판 진행", bg=BG, fg=TEXT, font=(FONT, 12)).pack(anchor="w")
        self.l_state = tk.Label(info, text="멈춰 있음", bg=BG, fg=DIM, font=(FONT, 10))
        self.l_state.pack(anchor="w")
        self.b_run = ttk.Button(top, text="시작 (F8)", style="Gold.TButton", command=self.toggle)
        self.b_run.pack(side="right")
        tk.Label(top, text="F9 즉시 중지 · 게임 창은 뒤에 있어도 됨", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="right", padx=12)

        sf = tk.Frame(self, bg=BG)
        sf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(sf, text="공포 영역 단계", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        st = self.cfg.get("stage")
        self.v_stage = tk.StringVar(value=str(st) if st else "")
        ttk.Entry(sf, textvariable=self.v_stage, width=5, font=(FONT, 11)).pack(side="left", padx=8)
        tk.Label(sf, text="이 단계만 계속 돕니다.", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        tk.Label(sf, text="미니맵 표시 줍기 범위", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left", padx=(24, 0))
        self.v_radius = tk.StringVar(value=str(self.cfg.get("radius_m", 10)))
        ttk.Entry(sf, textvariable=self.v_radius, width=4, font=(FONT, 11)).pack(side="left", padx=8)
        tk.Label(sf, text="m (캐릭터 키 2m 기준, 상하좌우)", bg=BG, fg=DIM,
                 font=(FONT, 9)).pack(side="left")

        kf = tk.Frame(self, bg=BG)
        kf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(kf, text="방향키 녹음", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        self.b_rec = ttk.Button(kf, text="녹음 시작 (F7)", command=self.rec_toggle)
        self.b_rec.pack(side="left", padx=8)
        ttk.Button(kf, text="녹음 지우기", command=self.rec_clear).pack(side="left")
        self.l_rec = tk.Label(kf, text="", bg=BG, fg=DIM, font=(FONT, 9))
        self.l_rec.pack(side="left", padx=10)
        self._rec_label()

        box = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        box.pack(fill="x", padx=20)
        for i, (head, body) in enumerate(RULES, 1):
            row = tk.Frame(box, bg=PANEL)
            row.pack(fill="x", padx=14, pady=(10 if i == 1 else 4, 10 if i == len(RULES) else 0))
            tk.Label(row, text="%d  %s" % (i, head), bg=PANEL, fg=GOLD, font=(FONT, 10, "bold"),
                     width=8, anchor="nw").pack(side="left", anchor="n")
            tk.Label(row, text=body, bg=PANEL, fg=TEXT, font=(FONT, 9), justify="left", anchor="w",
                     wraplength=580).pack(side="left", fill="x")

        rb = tk.Frame(self, bg=BG)
        rb.pack(fill="x", padx=20, pady=(12, 0))
        self.l_upd = tk.Label(rb, text="자동 업데이트: 확인 전", bg=BG, fg=DIM, font=(FONT, 9))
        self.l_upd.pack(side="left")
        ttk.Button(rb, text="기록 폴더 열기", command=self._open_records).pack(side="right")
        ttk.Button(rb, text="모은 아이템 보기", command=self._open_items).pack(side="right", padx=(0, 8))
        ttk.Button(rb, text="장비 스캔", command=self._gear_scan).pack(side="right", padx=(0, 8))
        ttk.Button(rb, text="게임을 '가려져도 동작' 모드로 다시 실행", command=self._relaunch).pack(side="right", padx=8)

        ph = tk.Frame(self, bg=BG)
        ph.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(ph, text="휴대폰에서 보기", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")
        self.v_url = tk.StringVar(value="")
        ttk.Entry(ph, textvariable=self.v_url, state="readonly").pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(ph, text="주소 복사", command=self._copy_url).pack(side="left")

        lf = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        lf.pack(fill="both", expand=True, padx=20, pady=(12, 18))
        self.log = tk.Text(lf, bg=PANEL, fg=TEXT, font=(FONT, 9), relief="flat", height=10, state="disabled",
                           wrap="word", padx=10, pady=8)
        self.log.pack(fill="both", expand=True)

    # ---------- records / phone ----------
    # Chromium (the engine of this game) stops drawing and slows its clock when its window is fully
    # covered or minimised. These switches keep it running at full speed behind other windows.
    BG_FLAGS = ["--disable-renderer-backgrounding", "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=CalculateNativeWinOcclusion"]

    def _game_exe(self):
        try:
            from bot import bg
            hwnd = win.find_window(TITLE)
            if hwnd:
                path = bg.exe_path(hwnd)
                if path and path != self.cfg.get("game_exe"):
                    self.cfg["game_exe"] = path
                    save_cfg(self.cfg)
                return hwnd, path
        except Exception:
            pass
        return None, self.cfg.get("game_exe")

    def _relaunch(self):
        hwnd, path = self._game_exe()
        if not path or not os.path.exists(path):
            messagebox.showinfo("D2S Helper", "게임 실행 파일 위치를 모릅니다.\n게임을 한 번 켠 상태에서 이 버튼을 눌러 주세요.")
            return
        if hwnd:
            if not messagebox.askyesno("D2S Helper", "게임을 껐다가 '가려져도 동작' 모드로 다시 켭니다.\n"
                                       "진행 중인 판은 끝납니다. 계속할까요?"):
                return
            if self.engine:
                self.halt()
            try:
                import ctypes
                ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)       # WM_CLOSE
            except Exception:
                pass
            for _ in range(40):
                time.sleep(0.25)
                if not win.find_window(TITLE):
                    break
        try:
            subprocess.Popen([path] + self.BG_FLAGS, cwd=os.path.dirname(path))
            self._say("게임을 '가려져도 동작' 모드로 실행함: 게임이 뜨면 시작(F8)")
        except Exception as e:
            self._say("게임 실행 실패: %s" % e)

    # ---------- arrow-key route ----------
    def _rec_label(self):
        if self.keys.recording:
            self.l_rec.configure(text="녹음 중... 게임에서 방향키로 움직이고, 끝나면 F7", fg=RED)
            return
        n, length = self.keys.summary()
        if n:
            self.l_rec.configure(text="녹음 있음: %d번 누름, %.1f초 - 보스 처치 3초 뒤 이 녹음으로만 이동, 정리 후 다음 판" % (n, length),
                                 fg=GREEN)
        else:
            self.l_rec.configure(text="녹음 없음 - 화면 안을 촘촘히 돌며 줍기", fg=DIM)

    def rec_toggle(self):
        if self.keys.recording:
            self.keys.stop()
            self.b_rec.configure(text="녹음 시작 (F7)")
            n, length = self.keys.summary()
            self._say("방향키 녹음 끝: %d번 누름, %.1f초 (프로그램 밖 고정 폴더에 저장, 새 버전에서도 유지)" % (n, length))
        else:
            if self.engine:
                self._say("자동 진행 중에는 녹음할 수 없음: 중지(F8)한 뒤 녹음하세요")
                return
            self.keys.start()
            self.b_rec.configure(text="녹음 종료 (F7)")
            self._say("방향키 녹음 시작: 보스가 죽은 자리에서 줍기 경로대로 방향키로 움직이세요 (첫 키부터 기록)")
        self._rec_label()

    def rec_clear(self):
        if self.keys.recording:
            self.keys.stop()
            self.b_rec.configure(text="녹음 시작 (F7)")
        self.keys.clear()
        self._say("방향키 녹음을 지움")
        self._rec_label()

    def _gear_scan(self):
        """Info boxes of all worn / bag / stash items into one ZIP (the smith's window must be open)."""
        if getattr(self, "scanning", False):
            return
        self.scanning = True
        self._say("장비 스캔 시작: 끝날 때까지 마우스를 움직이지 마세요 (게임 창이 앞에 있을 때)")

        def work():
            from bot.gameio import GameIO
            from bot import gearscan
            if not self.io_lock.acquire(timeout=20):
                self.q.put(("log", "장비 스캔: 자동 진행이 손을 놓지 않음. 중지(F8) 후 다시 누르세요"))
                self.scanning = False
                return
            io = GameIO(TITLE)
            try:
                z = gearscan.scan(io, BASE, lambda m: self.q.put(("log", m)))
                if z:
                    try:
                        os.startfile(os.path.dirname(z))
                    except Exception:
                        pass
            except Exception as e:
                self.q.put(("log", "장비 스캔 오류: %s" % e))
            finally:
                io.close()
                self.io_lock.release()
                self.scanning = False
        threading.Thread(target=work, daemon=True).start()

    def _open_items(self):
        """The same page as on the phone, opened on this PC at the collected-items list."""
        import webbrowser
        port = int(self.cfg.get("remote_port", 8765))
        webbrowser.open("http://127.0.0.1:%d/?k=%s#items" % (port, self.cfg.get("remote_pin", "")))

    def _open_records(self):
        d = self.rec.dir or os.path.join(BASE, "records")
        os.makedirs(d, exist_ok=True)
        try:
            os.startfile(d)
        except Exception as e:
            self._say("기록 폴더를 열지 못함: %s (%s)" % (d, e))

    def _share(self):
        self.shared = {"running": self.engine is not None, "sorting": False, "runs": self.runs,
                       "started": self.started_at or 0, "window_title": TITLE, "rules": [],
                       "log": self.logbuf[-60:]}

    def _remote(self):
        if not self.cfg.get("remote_pin"):
            self.cfg["remote_pin"] = DEFAULT_PIN
            save_cfg(self.cfg)
        try:
            self.remote = Remote(BASE, int(self.cfg.get("remote_port", 8765)), self.cfg["remote_pin"],
                                 lambda: self.shared, lambda: self.q.put(("toggle", None)), lambda: None,
                                 io_lock=self.io_lock, title=TITLE)
            self.remote.start()
            self._show_url()
            threading.Thread(target=self._find_public_ip, daemon=True).start()
        except Exception as e:
            self.remote = None
            self.v_url.set("휴대폰 접속을 열지 못함: %s" % e)
        if MOVED:
            self._say("이전 버전의 기록을 가져옴: %s (보관 위치: %s)" % (", ".join(MOVED), paths.DATA))

    def _show_url(self):
        """Always the same address: internet address (as before) + fixed port + fixed PIN."""
        if not self.remote:
            return
        ip = self.cfg.get("public_ip")
        if ip:
            self.v_url.set("http://%s:%d/?k=%s" % (ip, self.remote.port, self.remote.pin))
        else:
            self.v_url.set(self.remote.url())

    def _find_public_ip(self):
        import urllib.request
        for u in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://checkip.amazonaws.com"):
            try:
                ip = urllib.request.urlopen(u, timeout=4).read().decode("ascii").strip()
                if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
                    if ip != self.cfg.get("public_ip"):
                        self.cfg["public_ip"] = ip
                        save_cfg(self.cfg)
                    self.after(0, self._show_url)
                    return
            except Exception:
                continue

    def _copy_url(self):
        self.clipboard_clear()
        self.clipboard_append(self.v_url.get())
        self._say("휴대폰 주소를 복사했습니다")

    # ---------- run ----------
    def toggle(self):
        if self.engine:
            return self.halt()
        try:
            stage = max(0, min(99, int(self.v_stage.get().strip() or 0)))
        except ValueError:
            stage = 0
        self.v_stage.set(str(stage) if stage else "")
        self.cfg["stage"] = stage
        try:
            radius = max(2.0, min(30.0, float(self.v_radius.get().strip() or 10)))
        except ValueError:
            radius = 10.0
        radius = int(radius) if radius == int(radius) else radius
        self.v_radius.set(str(radius))
        self.cfg["radius_m"] = radius
        save_cfg(self.cfg)
        self._game_exe()
        self.rec.start()
        self.started_at = time.time()
        emit = lambda k, p: self.q.put((k, p))
        state = {"target": stage, "deaths": 0}
        ecfg = json.loads(json.dumps(ENGINE_CFG))
        ecfg["always_town"] = stage > 0
        if self.keys.recording:
            self.rec_toggle()
        popts = dict(PICK_OPTS, town_stage=stage, radius_m=radius, route=list(self.keys.events))   # a recording replaces the screen sweep
        self._say("공포 영역 %d단계만 반복" % stage if stage else "단계 미지정: 목록 맨 위의 새로 열린 단계로 진행")
        self.engine = Engine(ecfg, BASE, emit, self.io_lock, state)
        self.engine.start()
        self.picker = PickerThread(TITLE, popts, emit, self.io_lock, state)
        self.picker.start()
        self.b_run.configure(text="중지 (F8)", style="Stop.TButton")

    def halt(self):
        if self.picker:
            self.picker.stop()
            self.picker = None
        if self.engine:
            self.engine.stop()
            self.engine = None
            self._say("중지")
        self.b_run.configure(text="시작 (F8)", style="Gold.TButton")
        self.l_state.configure(text="멈춰 있음", fg=DIM)

    def _say(self, msg):
        self.last_line = time.time()
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        self.logbuf.append(line)
        del self.logbuf[:-200]
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.configure(state="disabled")
        if self.engine:
            try:
                self.rec.note(msg)
            except Exception:
                pass

    def _pump(self):
        try:
            while True:
                kind, p = self.q.get_nowait()
                if kind == "log":
                    self._say(p)
                    if p.startswith("포탈 클릭") or p.startswith("마을로 나감"):
                        self.runs += 1
                        self.l_runs.configure(text=str(self.runs))
                elif kind == "restart":
                    self.runs += 1
                    self.l_runs.configure(text=str(self.runs))
                elif kind == "toggle":
                    self.toggle()
                elif kind == "rec":
                    self.rec_toggle()
                elif kind == "halt":
                    self.halt()
        except queue.Empty:
            pass
        if self.engine and time.time() - self.last_line > 20:
            self._say("(20초 동안 새 로그 없음 - 현재 화면 기록)")
        if self.engine and self.started_at:
            s = int(time.time() - self.started_at)
            self.l_state.configure(text="진행 중  %02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60), fg=GREEN)
        self._share()
        self._maybe_restart()
        try:
            self.l_upd.configure(text="자동 업데이트: " + self.upd.status)
        except Exception:
            pass
        self.after(150, self._pump)

    def _maybe_restart(self):
        """A new version was copied in: restart at a quiet moment (stopped, or in town between runs)."""
        if not self.upd.ready or self.restarting:
            return
        if self.upd_seen is None:
            self.upd_seen = time.time()
        st = getattr(self.engine, "state", None) or {}
        quiet = self.engine is None or (not st.get("over") and self.picker is not None
                                        and getattr(self.picker, "in_town", lambda: False)())
        if not quiet and time.time() - self.upd_seen < 600:
            return                                   # (after 10 minutes it restarts anyway)
        self.restarting = True
        self.cfg["resume"] = self.engine is not None
        save_cfg(self.cfg)
        self._say("새 버전 v%s로 다시 시작" % self.upd.ready)
        self.after(300, self._relaunch_self)

    def _relaunch_self(self):
        import sys
        self._close(quiet=True)
        flags = 0x08000000 if os.name == "nt" else 0        # CREATE_NO_WINDOW for the console copy
        exe = sys.executable
        if os.name == "nt" and exe.lower().endswith("python.exe") and os.path.exists(exe[:-10] + "pythonw.exe"):
            exe = exe[:-10] + "pythonw.exe"
        subprocess.Popen([exe, os.path.join(BASE, "main.py")], cwd=BASE, creationflags=flags)
        os._exit(0)

    def _close(self, quiet=False):
        if self.keys.recording:
            self.keys.stop()
        self.halt()
        if getattr(self, "remote", None):
            self.remote.stop()
        try:
            self.hot.stop()
        except Exception:
            pass
        try:
            self.upd.alive = False
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    win.set_dpi_aware()
    App().mainloop()
