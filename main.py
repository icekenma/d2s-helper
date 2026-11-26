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

from bot import alert, scorer, sorter, spot, win
from bot.engine import Engine, STAGE_LIMIT, FAST_LIMIT, stage_limit, limit_text
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
PICK_OPTS = {"set": False, "unique": True, "jewel": True, "stage_end": True, "auto_clean": True,
             "town_start": True, "town_stage": 0, "collect_after": True, "mouse_move": True, "over_limit": 100}
PLAY = [("skip_act", "막 건너뛰기", "패스트 모드에서 정수를 1개 주는 1·2·5막이면 바로 나가 다시 뽑음"),
        ("open_chests", "상자 열기", "보스 처치 뒤 모인 상자도 열어 나온 아이템까지 주움"),
        ("town_tidy", "마을 창고 정리", "등록·보관할 아이템으로 인벤토리가 차면 판이 끝난 뒤 마을 보관함에서 정리"),
        ("rare_bow", "레어 활 보관", "필요 레벨 60 이상 레어 활·석궁은 보관 (끄면 레어는 모두 판매)")]
FAST_EXT = [("count", "정수 세기"), ("jewel", "주얼 줍기"), ("unique", "유니크 줍기")]
FAST_EXT_NOTE = {"count": "보스 처치 뒤 인벤토리를 잠깐 열어 불타는 정수 수량을 읽음",
                 "jewel": "보스 처치 뒤 미니맵의 주얼 표시만 주움 (증뎀은 모은 아이템에 표시)",
                 "unique": "보스 처치 뒤 미니맵의 유니크 표시만 주움 (보관 목록 아이템은 보관함)"}
RULES = [
    ("마을", "웨이포인트를 눌러 위에서 정한 공포 영역 단계로 들어갑니다(비워 두면 목록 맨 위의 새로 열린 단계)."),
    ("사냥", "보스가 죽어 '차원문이 열렸습니다'가 뜰 때까지 움직이지 않습니다."),
    ("줍기", "[1] 보스 처치 확인 -> [2] 포탈 확인 -> [3] 캐릭터 주위로 모인 아이템을 이름표 색으로 골라 가까운 것부터 눌러 주움 "
             "(유니크 황갈색·세트 초록·주얼 파란 이름·부적 주황, 정수는 저절로 들어옴). 아이템을 다 주우면 회색 '상자'를 하나씩 열고, 나온 아이템을 주운 뒤 다음 상자. 가방이 차면 대장장이 정리 후 "
             "이어서 주움 -> [4] 인벤토리에 정수만 남았는지 확인, 다른 게 있으면 대장장이 정리 후 다시 확인 -> Esc로 마을 귀환. "
             "단계마다 로그에 [1단계]~[4단계]로 자세히 남김."),
    ("정리", "인벤토리에 무언가 있을 때만 대장장이에게 갑니다. 새 유니크·세트는 등록, 이미 등록된 것은 판매"
             "(유니크 부적은 보관). 정수는 절대 팔지 않고 한 묶음이 90개 이상일 때만 보관함, 그 전엔 인벤토리에 둠. "
             "주얼·룬·작은 부적·거대 부적·레어 화살은 보관함(개인 → 공유 1~3). 나머지는 판매."),
    ("다음 판", "포탈(또는 웨이포인트)로 사냥터에 들어간 지 4분이 지나면 무엇을 하던 중이든 Esc -> '저장 후 나가기'로 마을에 가서 다음 판. "
                "창고 정리를 했으면 Esc -> '저장 후 나가기'로 마을에 가서 웨이포인트로 다음 판, 정리할 게 없었으면 포탈을 탑니다. 죽거나 판이 끝나면 Esc로 마을에 가서 같은 단계를 다시 고릅니다. 어디서든 막히면 "
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
        scorer.set_min_star(self.cfg.get("keep_star", 1))
        sorter.RUN_INFO["stage"] = int(self.cfg.get("stage", 0) or 0)
        self.watch = alert.Watch(self.cfg, save_cfg, lambda m: self.q.put(("log", m)))
        try:
            from bot import namer
            namer.start(lambda m: self.q.put(("log", m)))     # item names for the phone (Windows OCR)
        except Exception:
            pass
        self._apply_play()
        self._build()
        self._share()
        self._remote()
        self.hot = keyboard.GlobalHotKeys({"<f7>": lambda: self.q.put(("rec", None)),
                                           "<f8>": lambda: self.q.put(("toggle", None)),
                                           "<f9>": lambda: self.q.put(("halt", None)),
                                           "<f10>": lambda: self.q.put(("scan", None))})
        self.hot.daemon = True
        self.hot.start()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.restarting, self.upd_seen, self.scanning = False, None, False
        self.upd = Updater(BASE, lambda m: self.q.put(("log", m)), every=20)
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
        s.configure("Runs.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=28,
                    font=(FONT, 10), borderwidth=0)
        s.map("Runs.Treeview", background=[("selected", LINE)], foreground=[("selected", GOLD)])
        s.configure("Runs.Treeview.Heading", background=BG, foreground=DIM, font=(FONT, 9, "bold"),
                    relief="flat", padding=(6, 6))
        s.map("Runs.Treeview.Heading", background=[("active", BG)])

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
        cd = tk.Frame(top, bg=BG)                      # 3-minute countdown of the current stage
        cd.pack(side="right", padx=(0, 14))
        self.l_cd = tk.Label(cd, text="-:--", bg=BG, fg=DIM, font=(FONT, 24, "bold"))
        self.l_cd.pack(anchor="e")
        self.l_cd_note = tk.Label(cd, text="다음 판까지", bg=BG, fg=DIM, font=(FONT, 9))
        self.l_cd_note.pack(anchor="e")
        tk.Label(top, text="F9 즉시 중지 · F10 장비 스캔 · 게임 창은 뒤에 있어도 됨", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="right", padx=12)

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

        ff = tk.Frame(self, bg=BG)
        ff.pack(fill="x", padx=20, pady=(0, 10))
        self.v_fast = tk.BooleanVar(value=bool(self.cfg.get("fast_mode", False)))
        ttk.Checkbutton(ff, text="패스트 모드", variable=self.v_fast,
                        command=lambda: self._set_fast(self.v_fast.get(), "PC")).pack(side="left")
        tk.Label(ff, text="아이템을 줍지 않고 보스를 잡으면 바로 Esc로 마을 -> 다음 판 (정수는 저절로 들어옴)",
                 bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left", padx=8)
        xf = tk.Frame(self, bg=BG)
        xf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(xf, text="패스트 확장", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        ext = self._fast_ext()
        self.v_ext = {}
        for k, label in FAST_EXT:
            v = tk.BooleanVar(value=ext[k])
            self.v_ext[k] = v
            ttk.Checkbutton(xf, text=label, variable=v,
                            command=lambda k=k: self._set_fast_ext({k: self.v_ext[k].get()}, "PC")).pack(side="left", padx=(8, 0))

        pf = tk.Frame(self, bg=BG)
        pf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(pf, text="사냥 설정", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        play = self._play()
        self.v_play = {}
        for k, label, _ in PLAY:
            v = tk.BooleanVar(value=play[k])
            self.v_play[k] = v
            ttk.Checkbutton(pf, text=label, variable=v,
                            command=lambda k=k: self._set_play({k: self.v_play[k].get()}, "PC")).pack(side="left", padx=(8, 0))

        lf = tk.Frame(self, bg=BG)
        lf.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(lf, text="판 시간 제한", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        ls = self._limits_state()
        self.v_lim_n = tk.StringVar(value=limit_text(ls["limit_normal"]))
        self.v_lim_f = tk.StringVar(value=limit_text(ls["limit_fast"]))
        tk.Label(lf, text="일반", bg=BG, fg=TEXT, font=(FONT, 10)).pack(side="left", padx=(10, 4))
        ttk.Entry(lf, textvariable=self.v_lim_n, width=9, font=(FONT, 10)).pack(side="left")
        tk.Label(lf, text="패스트", bg=BG, fg=TEXT, font=(FONT, 10)).pack(side="left", padx=(10, 4))
        ttk.Entry(lf, textvariable=self.v_lim_f, width=9, font=(FONT, 10)).pack(side="left")
        ttk.Button(lf, text="적용", command=self._apply_limits_pc).pack(side="left", padx=8)
        tk.Label(lf, text="예: 4분, 2분 40초, 3:20, 200(초)", bg=BG, fg=DIM, font=(FONT, 9)).pack(side="left")

        af = tk.Frame(self, bg=BG)
        af.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(af, text="정수 경고", bg=BG, fg=GOLD, font=(FONT, 11, "bold")).pack(side="left")
        self.v_alert = tk.BooleanVar(value=bool(self.watch.conf.get("on")))
        ttk.Checkbutton(af, text="%d분 동안 정수가 안 늘면 휴대폰 알림" % self.watch.state()["minutes"],
                        variable=self.v_alert,
                        command=lambda: self._set_alert({"on": self.v_alert.get()}, "PC")).pack(side="left", padx=8)
        ttk.Button(af, text="시험 알림 (10초 뒤)", command=self.watch.test).pack(side="left")
        tk.Label(af, text="ntfy 앱에서 구독: %s" % self.watch.conf["topic"], bg=BG, fg=DIM,
                 font=(FONT, 9)).pack(side="left", padx=10)

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
        ttk.Button(rb, text="진행 상황", command=self._open_runs).pack(side="right", padx=(0, 8))
        ttk.Button(rb, text="인벤토리 즉시 정리", command=lambda: self._tidy_now("PC")).pack(side="right", padx=(0, 8))
        ttk.Button(rb, text="보관 목록", command=lambda: self._open_items("keep")).pack(side="right", padx=(0, 8))
        ttk.Button(rb, text="장비 스캔 (F10)", command=self._gear_scan).pack(side="right", padx=(0, 8))
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

    def _set_stage(self, stage):
        """From the phone: the terror zone stage to repeat. Saved, and if running, used from the next run."""
        stage = max(0, min(99, stage))
        self.v_stage.set(str(stage) if stage else "")
        self.cfg["stage"] = stage
        sorter.RUN_INFO["stage"] = stage
        save_cfg(self.cfg)
        if self.engine is not None:
            try:
                self.engine.state["target"] = stage
                self.engine.state["deaths"] = 0
                self.engine.cfg["always_town"] = stage > 0
            except Exception:
                pass
            if self.picker is not None and getattr(self.picker, "p", None) is not None:
                self.picker.p.opts["town_stage"] = stage
                self.picker.p.wp_frontier = False
        self._say("휴대폰에서 단계 변경: %s" % ("공포 영역 %d단계 반복 (다음 판부터)" % stage if stage
                                           else "단계 미지정 (목록 맨 위 새 단계)"))
        self._share()

    def _apply_limits_pc(self):
        def secs(t):
            import re as _re
            t = str(t).strip()
            m = _re.match(r"^(\d+):(\d{1,2})$", t)
            if m:
                return int(m.group(1)) * 60 + int(m.group(2))
            m = _re.match(r"^(?:(\d+)\s*분)?\s*(?:(\d+)\s*초)?$", t)
            if m and (m.group(1) or m.group(2)):
                return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
            return int(t)
        try:
            self._set_limits({"normal": secs(self.v_lim_n.get()), "fast": secs(self.v_lim_f.get())}, "PC")
        except (TypeError, ValueError):
            self._say("판 시간 제한: '4분', '2분 40초', '3:20', '200' 같은 형식으로 넣어 주세요")

    def _limits_state(self):
        return {"limit_normal": int(self.cfg.get("limit_normal") or STAGE_LIMIT),
                "limit_fast": int(self.cfg.get("limit_fast") or FAST_LIMIT)}

    def _set_limits(self, d, who="휴대폰"):
        """Seconds in one stage before the helper leaves for the next run (normal / 패스트 모드)."""
        for k, key in (("normal", "limit_normal"), ("fast", "limit_fast")):
            if k in d:
                try:
                    self.cfg[key] = max(60, min(900, int(d[k])))
                except (TypeError, ValueError):
                    pass
        save_cfg(self.cfg)
        ls = self._limits_state()
        if self.engine is not None:
            self.engine.state.update(ls)
        try:
            self.v_lim_n.set(limit_text(ls["limit_normal"]))
            self.v_lim_f.set(limit_text(ls["limit_fast"]))
        except Exception:
            pass
        self._say("%s에서 판 시간 제한 변경: 일반 %s, 패스트 %s (다음 판부터)" % (
            who, limit_text(ls["limit_normal"]), limit_text(ls["limit_fast"])))
        self._share()

    def _inv_view(self, who="휴대폰"):
        """'지금 인벤토리': read every item in the bags (nothing moved) for the phone."""
        if self.engine is not None:
            self.engine.state["inv_view"] = time.time()
            self._say("%s에서 인벤토리 보기: 인벤토리를 잠깐 열어 아이템을 읽음" % who)
            return
        if getattr(self, "viewing", False):
            return
        self.viewing = True
        self._say("%s에서 인벤토리 보기: 인벤토리를 열어 아이템을 읽음 (게임 창이 앞에 있을 때)" % who)

        def work():
            from bot.gameio import GameIO
            if not self.io_lock.acquire(timeout=10):
                self.q.put(("log", "인벤토리 보기: 다른 작업 중이라 못 읽음"))
                self.viewing = False
                return
            io = GameIO(TITLE)
            log = lambda m: self.q.put(("log", m))
            try:
                f = io.grab()
                if f is None:
                    log("인벤토리 보기: 게임 창을 찾지 못함")
                    return
                geo = sorter.Geo(f)
                opened = sorter.seen(geo.normalize(f), "inv") < 0.7
                if opened:
                    io.key("c")
                    time.sleep(0.8)
                text, _ = sorter.cleanup(io, ENGINE_CFG.get("sort", {}), BASE, log, None, mode="view")
                log("[인벤토리 보기] " + text)
                if opened:
                    io.key("c")
                    time.sleep(0.3)
            except Exception as e:
                log("인벤토리 보기 오류: %s" % e)
            finally:
                io.close()
                self.io_lock.release()
                self.viewing = False
        threading.Thread(target=work, daemon=True).start()

    def _sell_now(self, d, who="휴대폰"):
        """'판매 예약' from an item's detail sheet: sell that item from the bags right away."""
        try:
            req = (int(d["c"]), int(d["r"]), str(d.get("name") or "")[:40])
        except (KeyError, TypeError, ValueError):
            return
        self._say("%s에서 판매 예약: %s (%d열 %d줄)" % (who, req[2] or "아이템", req[0] + 1, req[1] + 1))
        if self.engine is not None:
            self.engine.state.setdefault("sell_req", []).append(req)
            self.engine.state["sell_at"] = time.time()
            return

        def work():
            from bot.gameio import GameIO
            if not self.io_lock.acquire(timeout=10):
                self.q.put(("log", "판매 예약: 다른 작업 중이라 못 함"))
                return
            io = GameIO(TITLE)
            log = lambda m: self.q.put(("log", m))
            try:
                f = io.grab()
                if f is None:
                    log("판매 예약: 게임 창을 찾지 못함")
                    return
                opened = sorter.seen(sorter.Geo(f).normalize(f), "inv") < 0.7
                if opened:
                    io.key("c")
                    time.sleep(0.8)
                sorter.sell_cells(io, [req], BASE, log)
                if opened:
                    io.key("c")
                    time.sleep(0.3)
            except Exception as e:
                log("판매 예약 오류: %s" % e)
            finally:
                io.close()
                self.io_lock.release()
        threading.Thread(target=work, daemon=True).start()

    def _tidy_now(self, who="휴대폰"):
        """'인벤토리 즉시 정리': out of the stage now, the stash in town, then on hunting."""
        if self.engine is None:
            self._say("%s에서 인벤토리 즉시 정리: 도우미가 멈춰 있음 (시작한 뒤에 누르세요)" % who)
            return
        now = time.time()
        self.engine.state["town_tidy"] = now
        self.engine.state["tidy_now"] = now
        self._say("%s에서 인벤토리 즉시 정리: 지금 마을로 가서 창고 정리 후 사냥을 이어감" % who)

    def _set_alert(self, changes, who="휴대폰"):
        c = self.watch.conf
        if "on" in changes:
            c["on"] = bool(changes["on"])
        if "minutes" in changes:
            try:
                c["minutes"] = max(10, min(180, int(changes["minutes"])))
            except (TypeError, ValueError):
                pass
        save_cfg(self.cfg)
        try:
            self.v_alert.set(bool(c["on"]))
        except Exception:
            pass
        self._say("%s에서 정수 경고 알림 %s" % (who, "켬: %d분 동안 불타는 정수가 안 늘면 휴대폰(ntfy '%s')으로 알림"
                                           % (int(c.get("minutes") or 30), c["topic"]) if c["on"] else "끔"))
        self._share()

    def _play(self):
        p = self.cfg.get("play") or {}
        if int(self.cfg.get("play_v", 0) or 0) < 2:
            # v3.24: chests are no longer opened by default (picking up was taking too long)
            p = dict(p, open_chests=False)
            self.cfg["play"], self.cfg["play_v"] = p, 2
            save_cfg(self.cfg)
        return {k: bool(p.get(k, k != "open_chests")) for k, _, _ in PLAY}

    def _apply_play(self):
        p = self._play()
        from bot import sorter
        sorter.PLAY.update(rare_bow=p["rare_bow"])
        pk = getattr(self, "picker", None)
        if pk is not None and getattr(pk, "p", None) is not None:
            pk.p.opts.update(p)

    def _set_play(self, changes, who="휴대폰"):
        """사냥 설정: act skip, chests, town stash tidy, rare bows kept, sets sold."""
        p = self._play()
        p.update({k: bool(v) for k, v in changes.items() if k in p})
        self.cfg["play"] = p
        save_cfg(self.cfg)
        for k, v in getattr(self, "v_play", {}).items():
            try:
                v.set(p[k])
            except Exception:
                pass
        self._apply_play()
        info = {k: (n, note) for k, n, note in PLAY}
        for k, v in changes.items():
            if k in info:
                self._say("%s에서 사냥 설정 '%s' %s: %s" % (who, info[k][0], "켬" if v else "끔", info[k][1]))
        self._share()

    def _fast_ext(self):
        e = self.cfg.get("fast_ext") or {}
        return {k: bool(e.get(k, False)) for k, _ in FAST_EXT}

    def _apply_fast_ext(self):
        e = self._fast_ext()
        if self.picker is not None and getattr(self.picker, "p", None) is not None:
            self.picker.p.opts.update(fast_count=e["count"], fast_jewel=e["jewel"], fast_unique=e["unique"])
        if self.engine is not None:
            self.engine.state["fast_pick"] = e["jewel"] or e["unique"]

    def _set_fast_ext(self, changes, who="휴대폰"):
        """패스트 모드 extras: count the essences, pick up jewels / uniques after the boss."""
        e = self._fast_ext()
        e.update({k: bool(v) for k, v in changes.items() if k in e})
        self.cfg["fast_ext"] = e
        save_cfg(self.cfg)
        for k, v in self.v_ext.items():
            try:
                v.set(e[k])
            except Exception:
                pass
        self._apply_fast_ext()
        names = dict(FAST_EXT)
        for k, v in changes.items():
            if k in names:
                self._say("%s에서 패스트 확장 '%s' %s%s" % (who, names[k], "켬" if v else "끔",
                                                     ": " + FAST_EXT_NOTE[k] if v else ""))
        if e["jewel"] or e["unique"]:
            self._say("패스트 모드 판 제한: 줍기·정리 시간까지 %s" % limit_text(self._limit()))
        self._share()

    def _set_fast(self, on, who="휴대폰"):
        """패스트 모드: no pick-up, no route, no smith; after the boss Esc -> town -> next run."""
        on = bool(on)
        self.cfg["fast_mode"] = on
        save_cfg(self.cfg)
        try:
            self.v_fast.set(on)
        except Exception:
            pass
        if self.picker is not None and getattr(self.picker, "p", None) is not None:
            self.picker.p.opts["fast_mode"] = on
        if self.engine is not None:
            self.engine.state["fast"] = on
        e = self._fast_ext()
        extra = [n for k, n in FAST_EXT if e[k]]
        self._say("%s에서 패스트 모드 %s" % (who, "켬: 보스를 잡으면 바로 다음 판, %s 지나면 무조건 다음 판 (%s)" % (
            limit_text(self._limit()), "확장: " + ", ".join(extra) if extra else "아이템 줍기·정리 안 함") if on
                                            else "끔: 녹음 경로·줍기·대장장이 정리를 다시 함"))
        self._share()

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

    # ---- run table: every run with its times, essences and the items kept -------------------------
    def _open_runs(self):
        w = getattr(self, "runs_win", None)
        if w is not None and w.winfo_exists():
            w.deiconify()
            w.lift()
            return
        from bot import runlog
        w = self.runs_win = tk.Toplevel(self)
        w.title("진행 상황")
        w.configure(bg=BG)
        w.geometry("880x620")
        w.minsize(640, 420)
        head = tk.Frame(w, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 10))
        tk.Label(head, text="진행 상황", bg=BG, fg=TEXT, font=(FONT, 15, "bold")).pack(side="left")
        live = tk.Label(head, text="", bg=BG, fg=GOLD, font=(FONT, 10, "bold"))
        live.pack(side="left", padx=12)

        def clear_runs():
            from tkinter import messagebox
            if messagebox.askyesno("진행 기록 비우기", "진행 기록을 비울까요?\n진행 중인 판은 남고, 지난 기록은 runs_old_ 파일로 옮겨 둡니다.",
                                   parent=w):
                runlog.clear()
                state["sel"] = None

        ttk.Button(head, text="기록 비우기", command=clear_runs).pack(side="right")
        stats = tk.Frame(w, bg=BG)
        stats.pack(fill="x", padx=18, pady=(0, 10))
        cells = {}
        for key, cap in (("runs", "오늘 판"), ("avg", "판당 평균 시간"), ("ess", "오늘 정수"), ("rate", "정수 / 시간"),
                         ("items", "오늘 보관·등록")):
            f = tk.Frame(stats, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
            f.pack(side="left", fill="x", expand=True, padx=(0, 8))
            tk.Label(f, text=cap, bg=PANEL, fg=DIM, font=(FONT, 9)).pack(anchor="w", padx=12, pady=(8, 0))
            v = tk.Label(f, text="-", bg=PANEL, fg=GOLD if key in ("ess", "rate") else TEXT, font=(FONT, 16, "bold"))
            v.pack(anchor="w", padx=12, pady=(0, 8))
            cells[key] = v
        cols = ("n", "stage", "start", "end", "secs", "ess", "items")
        heads = {"n": ("판", 50), "stage": ("단계", 60), "start": ("시작", 80), "end": ("종료", 80),
                 "secs": ("걸린 시간", 80), "ess": ("정수", 60), "items": ("보관·등록한 아이템", 380)}
        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=18)
        tv = ttk.Treeview(body, columns=cols, show="headings", style="Runs.Treeview", height=12)
        for c in cols:
            tv.heading(c, text=heads[c][0], anchor="w" if c == "items" else "center")
            tv.column(c, width=heads[c][1], anchor="w" if c == "items" else "center", stretch=(c == "items"))
        sb = ttk.Scrollbar(body, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tv.tag_configure("live", foreground=GOLD)
        tv.tag_configure("odd", background="#221c18")
        tk.Label(w, text="선택한 판의 아이템", bg=BG, fg=DIM, font=(FONT, 9, "bold")).pack(anchor="w", padx=18, pady=(12, 4))
        det = tk.Text(w, height=7, bg=PANEL, fg=TEXT, relief="flat", font=(FONT, 10), padx=12, pady=8,
                      highlightthickness=1, highlightbackground=LINE, wrap="word")
        det.pack(fill="x", padx=18, pady=(0, 16))
        det.tag_configure("dim", foreground=DIM)
        det.tag_configure("gold", foreground=GOLD)
        state = {"runs": [], "sel": None}

        def hm(t):
            return time.strftime("%H:%M:%S", time.localtime(t)) if t else ""

        def dur(sec):
            sec = int(max(0, sec))
            return "%d:%02d" % (sec // 60, sec % 60)

        def show_detail(*_):
            sel = tv.selection()
            if sel:
                state["sel"] = int(sel[0])
            r = next((x for x in state["runs"] if x["n"] == state["sel"]), None)
            det.configure(state="normal")
            det.delete("1.0", "end")
            if r is None:
                det.insert("end", "표에서 판을 고르면 그 판에 보관·등록한 아이템이 여기에 나옵니다.", "dim")
            else:
                det.insert("end", "%d판 · 공포 %s단계 · %s ~ %s · %s · 정수 %s개\n" % (
                    r["n"], r["stage"] or "-", hm(r["start"]), hm(r["end"]) or "진행 중", dur(r["secs"]),
                    "-" if r["ess"] is None else r["ess"]), "gold")
                if not r["items"]:
                    det.insert("end", "보관·등록한 아이템 없음 (판매한 아이템은 기록하지 않음)", "dim")
                for x in r["items"]:
                    det.insert("end", "%s  " % hm(x["t"]), "dim")
                    det.insert("end", "%s" % x["name"])
                    det.insert("end", "  %s%s\n" % (x["action"], (" · " + x["tab"] + " 탭") if x["tab"] else ""), "dim")
            det.configure(state="disabled")

        tv.bind("<<TreeviewSelect>>", show_detail)

        def refresh():
            if not w.winfo_exists():
                return
            try:
                runs = runlog.build(300)
            except Exception:
                runs = []
            state["runs"] = runs
            today0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
            td = [r for r in runs if r["start"] >= today0]
            done = [r for r in td if not r["live"]]
            cells["runs"].configure(text="%d판" % len(td))
            cells["avg"].configure(text=dur(sum(r["secs"] for r in done) / len(done)) if done else "-")
            e = [r["ess"] for r in td if r["ess"] is not None]
            cells["ess"].configure(text="%d개" % sum(e) if e else "-")
            secs = sum(r["secs"] for r in done if r["ess"] is not None)
            cells["rate"].configure(text="%.1f개" % (sum(r["ess"] for r in done if r["ess"] is not None) * 3600.0 / secs)
                                    if secs > 60 else "-")
            cells["items"].configure(text="%d개" % sum(len(r["items"]) for r in td))
            cur = runs[0] if runs and runs[0]["live"] else None
            live.configure(text=("● %d판 진행 중 %s" % (cur["n"], dur(cur["secs"]))) if cur else "")
            y = tv.yview()
            tv.delete(*tv.get_children())
            for i, r in enumerate(runs):
                names = ", ".join(x["name"] for x in r["items"][:4]) + (" 외 %d개" % (len(r["items"]) - 4)
                                                                         if len(r["items"]) > 4 else "")
                tags = ("live",) if r["live"] else (("odd",) if i % 2 else ())
                tv.insert("", "end", iid=str(r["n"]), tags=tags, values=(
                    r["n"], r["stage"] or "-", hm(r["start"]), "진행 중" if r["live"] else hm(r["end"]),
                    dur(r["secs"]), "-" if r["ess"] is None else r["ess"],
                    ("%d개 · %s" % (len(r["items"]), names)) if r["items"] else ""))
            if state["sel"] is not None and tv.exists(str(state["sel"])):
                tv.selection_set(str(state["sel"]))
            tv.yview_moveto(y[0])
            show_detail()
            w.after(2000, refresh)

        refresh()

    def _open_items(self, view="items"):
        """The same page as on the phone, opened on this PC at the collected-items (or keep) list."""
        import webbrowser
        port = int(self.cfg.get("remote_port", 8765))
        webbrowser.open("http://127.0.0.1:%d/?k=%s#%s" % (port, self.cfg.get("remote_pin", ""), view))

    def _open_records(self):
        d = self.rec.dir or os.path.join(BASE, "records")
        os.makedirs(d, exist_ok=True)
        try:
            os.startfile(d)
        except Exception as e:
            self._say("기록 폴더를 열지 못함: %s (%s)" % (d, e))

    def _limit(self):
        e = self._fast_ext()
        return stage_limit(dict(self._limits_state(), fast=bool(self.cfg.get("fast_mode", False)),
                                fast_pick=e["jewel"] or e["unique"]))

    def _stage_left(self):
        """Seconds left in this stage before the forced move to the next run, or None outside a stage."""
        if self.engine is None:
            return None
        si = self.engine.state.get("stage_in")
        return None if not si else max(0.0, stage_limit(self.engine.state) - (time.time() - si))

    def _countdown(self):
        left = self._stage_left()
        if left is None:
            self.l_cd.configure(text="-:--", fg=DIM)
            self.l_cd_note.configure(text="사냥터 들어가면 %s 셈" % limit_text(self._limit()) if self.engine else "다음 판까지")
        else:
            sec = int(left + 0.999)
            self.l_cd.configure(text="%d:%02d" % (sec // 60, sec % 60), fg=RED if sec <= 30 else GOLD)
            self.l_cd_note.configure(text="다음 판까지" if sec else "다음 판으로 이동 중")

    def _share(self):
        left = self._stage_left()
        self.shared = {"stage_left": left, "stage_limit": self._limit(), "running": self.engine is not None, "sorting": False, "runs": self.runs,
                       "started": self.started_at or 0, "window_title": TITLE, "rules": [],
                       "log": self.logbuf[-60:], "stage": int(self.cfg.get("stage", 0) or 0),
                       "fast": bool(self.cfg.get("fast_mode", False)), "fast_ext": self._fast_ext(), "alert": self.watch.state(),
                       "limits": self._limits_state(), "play": self._play()}

    def _remote(self):
        if not self.cfg.get("remote_pin"):
            self.cfg["remote_pin"] = DEFAULT_PIN
            save_cfg(self.cfg)
        try:
            self.remote = Remote(BASE, int(self.cfg.get("remote_port", 8765)), self.cfg["remote_pin"],
                                 lambda: self.shared, lambda: self.q.put(("toggle", None)),
                                 lambda: self.q.put(("scan", None)),
                                 io_lock=self.io_lock, title=TITLE)
            self.remote.set_stage = lambda n: self.q.put(("stage", n))
            self.remote.set_keep_star = lambda n: self.q.put(("keep_star", n))
            self.remote.set_fast = lambda on: self.q.put(("fast", on))
            self.remote.set_fast_ext = lambda d: self.q.put(("fast_ext", d))
            self.remote.set_play = lambda d: self.q.put(("play", d))
            self.remote.set_alert = lambda d: self.q.put(("alert", d))
            self.remote.set_limits = lambda d: self.q.put(("limits", d))
            self.remote.tidy_now = lambda: self.q.put(("tidy_now", None))
            self.remote.inv_view = lambda: self.q.put(("inv_view", None))
            self.remote.sell_now = lambda d: self.q.put(("sell_now", d))
            self.remote.note = lambda m: self.q.put(("log", m))
            self.remote.test_alert = lambda: self.q.put(("alert_test", None))
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
        sorter.RUN_INFO["stage"] = stage
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
        ext = self._fast_ext()
        state = {"target": stage, "deaths": 0, "fast": bool(self.cfg.get("fast_mode", False)),
                 "fast_pick": ext["jewel"] or ext["unique"]}
        state.update(self._limits_state())
        ecfg = json.loads(json.dumps(ENGINE_CFG))
        ecfg["always_town"] = stage > 0
        if self.keys.recording:
            self.rec_toggle()
        popts = dict(PICK_OPTS, town_stage=stage, radius_m=radius, fast_mode=bool(self.cfg.get("fast_mode", False)), route=list(self.keys.events), sort=ecfg.get("sort", {}),   # a recording replaces the screen sweep
                     fast_count=ext["count"], fast_jewel=ext["jewel"], fast_unique=ext["unique"], **self._play())
        self._apply_play()
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
                elif kind == "stage":
                    self._set_stage(int(p or 0))
                elif kind == "fast":
                    self._set_fast(p)
                elif kind == "fast_ext":
                    self._set_fast_ext(p)
                elif kind == "play":
                    self._set_play(p)
                elif kind == "alert":
                    self._set_alert(p)
                elif kind == "limits":
                    self._set_limits(p)
                elif kind == "tidy_now":
                    self._tidy_now()
                elif kind == "inv_view":
                    self._inv_view()
                elif kind == "sell_now":
                    self._sell_now(p)
                elif kind == "alert_test":
                    self.watch.test()
                elif kind == "keep_star":
                    n = max(1, min(5, int(p or 1)))
                    scorer.set_min_star(n)
                    self.cfg["keep_star"] = n
                    save_cfg(self.cfg)
                    self._say("휴대폰에서 보관 기준 변경: 보관 목록 중 별 %d개 이상만 보관함으로" % n)
                elif kind == "scan":
                    self._gear_scan()
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
        self._countdown()
        self._share()
        self._maybe_restart()
        try:
            self.watch.check(self.engine is not None)
        except Exception:
            pass
        if self.engine is not None and self.engine.state.pop("halt_now", False):
            self.halt()
        try:
            self.l_upd.configure(text="자동 업데이트: " + self.upd.status)
        except Exception:
            pass
        self.after(150, self._pump)

    def _maybe_restart(self):
        """A new version was copied in: restart at once (and carry on running if it was)."""
        if not self.upd.ready or self.restarting or getattr(self, "scanning", False):
            return                                   # (never in the middle of a gear scan)
        # at once: only an inventory tidy-up / item move under way (it holds the game lock) is let finish
        if not self.io_lock.acquire(False):
            return
        self.io_lock.release()
        self.restarting = True
        self.cfg["resume"] = self.engine is not None
        save_cfg(self.cfg)
        self._say("새 버전 v%s로 다시 시작" % self.upd.ready)
        self.after(300, self._relaunch_self)

    def _relaunch_self(self):
        import sys
        self._close(quiet=True)
        bat = os.path.join(BASE, "run.bat")
        if os.name == "nt" and os.path.exists(bat):
            # the same way it is normally started, in a console of its own. "cmd /c": the window closes by
            # itself when the program ends ('start' ran the batch file with cmd /k, which left an empty
            # command prompt behind after every update). This program's own window closes as it exits.
            subprocess.Popen(["cmd", "/c", bat], cwd=BASE, creationflags=0x00000010)  # CREATE_NEW_CONSOLE
        else:
            subprocess.Popen([sys.executable, os.path.join(BASE, "main.py")], cwd=BASE)
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
