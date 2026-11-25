# -*- coding: utf-8 -*-
"""Working with the game window while it is NOT in front: the picture is taken with PrintWindow (works
behind other windows), and mouse / keyboard input is posted straight to the window as messages, so the
real cursor and keyboard stay with whatever the user is doing."""
import ctypes
import time
from ctypes import wintypes

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

LRESULT = ctypes.c_ssize_t
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                       wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
user32.SendMessageTimeoutW.restype = LRESULT
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0200, 0x0201, 0x0202
WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x0100, 0x0101, 0x0102
MK_LBUTTON, MK_SHIFT, MK_CONTROL = 0x0001, 0x0004, 0x0008
VK = {"esc": 0x1B, "enter": 0x0D, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
      "shift": 0x10, "ctrl": 0x11, "tab": 0x09, "space": 0x20}
EXTENDED = {0x25, 0x26, 0x27, 0x28}


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def capture(hwnd):
    """The client area of a window as BGR, even when other windows cover it. None if it fails or comes
    back black (a minimised window, or one that stopped drawing)."""
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    w, h = rc.right - rc.left, rc.bottom - rc.top
    if w < 50 or h < 50:
        return None
    hdc_screen = user32.GetDC(0)
    memdc = gdi32.CreateCompatibleDC(hdc_screen)
    bmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    old = gdi32.SelectObject(memdc, bmp)
    try:
        # PW_CLIENTONLY | PW_RENDERFULLCONTENT: the GPU-drawn content of a Chromium/Electron window
        if not user32.PrintWindow(hwnd, memdc, 1 | 2):
            return None
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth, bi.biHeight, bi.biPlanes, bi.biBitCount = w, -h, 1, 32
        buf = (ctypes.c_ubyte * (w * h * 4))()
        if not gdi32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bi), 0):
            return None
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    finally:
        gdi32.SelectObject(memdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(0, hdc_screen)
    if int(img[::8, ::8].max()) < 12:
        return None
    return img


def _lparam(x, y):
    return ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


def vk_of(k):
    """pynput Key / a character / a name -> virtual-key code."""
    name = getattr(k, "name", None) or str(k)
    name = name.lower()
    if name in VK:
        return VK[name]
    if len(name) == 1:
        return ord(name.upper())
    return None


def _key_lparam(vk, up):
    scan = user32.MapVirtualKeyW(vk, 0) & 0xFF
    lp = 1 | (scan << 16)
    if vk in EXTENDED:
        lp |= 1 << 24
    if up:
        lp |= (1 << 30) | (1 << 31)
    return lp


_woke = {}


def wake(hwnd, every=1.0):
    """Tell the game window it is the active, focused one (without bringing it to the front). A browser-
    based game ignores keys while it thinks it is in the background; this makes it listen."""
    now = time.time()
    if now - _woke.get(hwnd, 0) < every:
        return
    _woke[hwnd] = now
    user32.PostMessageW(hwnd, 0x0086, 1, 0)        # WM_NCACTIVATE (active)
    user32.PostMessageW(hwnd, 0x0006, 1, 0)        # WM_ACTIVATE  (WA_ACTIVE)
    user32.PostMessageW(hwnd, 0x0007, 0, 0)        # WM_SETFOCUS


def key_down(hwnd, k, repeat=False):
    vk = vk_of(k)
    if vk is None:
        return
    wake(hwnd)
    lp = _key_lparam(vk, False)
    if repeat:
        lp |= 1 << 30                              # "the key was already down" (auto-repeat)
    user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lp)
    if repeat:
        return
    if 0x41 <= vk <= 0x5A:
        user32.PostMessageW(hwnd, WM_CHAR, vk + 32, _key_lparam(vk, False))


def key_up(hwnd, k):
    vk = vk_of(k)
    if vk is None:
        return
    user32.PostMessageW(hwnd, WM_KEYUP, vk, _key_lparam(vk, True))


def move(hwnd, x, y, held=False):
    wake(hwnd)
    user32.PostMessageW(hwnd, WM_MOUSEMOVE, MK_LBUTTON if held else 0, _lparam(x, y))


def wheel(hwnd, sx, sy, notches):
    """Mouse wheel at screen point (sx, sy); notches > 0 scrolls up, < 0 down."""
    delta = int(120 * notches) & 0xFFFF
    user32.PostMessageW(hwnd, 0x020A, delta << 16, ((int(sy) & 0xFFFF) << 16) | (int(sx) & 0xFFFF))


def button(hwnd, x, y, down):
    wake(hwnd)
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN if down else WM_LBUTTONUP, MK_LBUTTON if down else 0,
                        _lparam(x, y))


def click(hwnd, x, y, mod=None):
    """A left click at client (x, y); mod 'ctrl' / 'shift' held for it."""
    flag = {"ctrl": MK_CONTROL, "shift": MK_SHIFT}.get(mod, 0)
    vk = {"ctrl": 0x11, "shift": 0x10}.get(mod)
    move(hwnd, x, y)
    time.sleep(0.03)
    if not vk:
        button(hwnd, x, y, True)
        time.sleep(0.04)
        button(hwnd, x, y, False)
        return
    # the program reads the modifier from its keyboard state: share ours with its thread for the click
    tid = user32.GetWindowThreadProcessId(hwnd, None)
    cur = kernel32.GetCurrentThreadId()
    attached = bool(tid and user32.AttachThreadInput(cur, tid, True))
    state = (ctypes.c_ubyte * 256)()
    try:
        user32.GetKeyboardState(state)
        saved = state[vk]
        state[vk] = 0x80
        user32.SetKeyboardState(state)
        user32.PostMessageW(hwnd, WM_KEYDOWN, vk, _key_lparam(vk, False))
        res = ctypes.c_size_t()
        user32.SendMessageTimeoutW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON | flag, _lparam(x, y), 0x0002, 300,
                                   ctypes.byref(res))
        time.sleep(0.04)
        user32.SendMessageTimeoutW(hwnd, WM_LBUTTONUP, flag, _lparam(x, y), 0x0002, 300, ctypes.byref(res))
        user32.PostMessageW(hwnd, WM_KEYUP, vk, _key_lparam(vk, True))
        state[vk] = saved
        user32.SetKeyboardState(state)
    finally:
        if attached:
            user32.AttachThreadInput(cur, tid, False)


def restore_behind(hwnd):
    """Un-minimise the game without bringing it in front of what the user is doing."""
    SW_SHOWNOACTIVATE, HWND_BOTTOM = 4, 1
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)   # NOSIZE|NOMOVE|NOACTIVATE


def exe_path(hwnd):
    """Full path of the program that owns a window."""
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(0x1000, False, pid.value)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        n = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return buf.value
    finally:
        kernel32.CloseHandle(h)
    return None
