# -*- coding: utf-8 -*-
"""Win32 window helpers via ctypes."""
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32


def set_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def _pid(hwnd):
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def find_window(title_part):
    """The game window: a visible top-level window of another program whose title contains
    title_part. With several matches (a folder or browser tab of the same name) the largest wins,
    and an exact title beats a partial one."""
    found = []
    part = title_part.lower()
    me = ctypes.windll.kernel32.GetCurrentProcessId()
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd) and _pid(hwnd) != me:
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                t = buf.value.lower()
                if part in t:
                    rc = wintypes.RECT()
                    user32.GetClientRect(hwnd, ctypes.byref(rc))
                    found.append((t == part, rc.right * rc.bottom, hwnd))
        return True

    user32.EnumWindows(proto(cb), 0)
    if not found:
        return None
    found.sort(key=lambda f: (f[0], f[1]), reverse=True)
    return found[0][2]


def client_rect(hwnd):
    """(left, top, width, height) of the client area in screen pixels."""
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top


def is_minimized(hwnd):
    return bool(user32.IsIconic(hwnd))


def is_foreground(hwnd):
    """True when the foreground window is the game or another window of the same program."""
    fg = user32.GetForegroundWindow()
    return bool(fg) and (fg == hwnd or _pid(fg) == _pid(hwnd))


def resize_client(hwnd, w, h):
    """Resize the window so that its client area becomes w x h."""
    cr = wintypes.RECT()
    wr = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    user32.GetWindowRect(hwnd, ctypes.byref(wr))
    dw = (wr.right - wr.left) - (cr.right - cr.left)
    dh = (wr.bottom - wr.top) - (cr.bottom - cr.top)
    SWP_NOMOVE, SWP_NOZORDER = 0x0002, 0x0004
    user32.SetWindowPos(hwnd, 0, 0, 0, w + dw, h + dh, SWP_NOMOVE | SWP_NOZORDER)


def idle_seconds():
    """Seconds since the last keyboard/mouse input on this PC."""
    try:
        class LII(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
        li = LII()
        li.cbSize = ctypes.sizeof(LII)
        if not user32.GetLastInputInfo(ctypes.byref(li)):
            return 999.0
        return max(0.0, (ctypes.windll.kernel32.GetTickCount() - li.dwTime) / 1000.0)
    except Exception:
        return 999.0


def focus(hwnd):
    """Bring the game window to the front (restoring it if minimised). Uses the usual ways around
    Windows' foreground lock; returns True when it ended up in front."""
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)                      # SW_RESTORE
        fg = user32.GetForegroundWindow()
        cur = ctypes.windll.kernel32.GetCurrentThreadId()
        other = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        attached = bool(other and other != cur and user32.AttachThreadInput(cur, other, True))
        try:
            user32.keybd_event(0x12, 0, 0, 0)               # an Alt tap lifts the foreground lock
            user32.keybd_event(0x12, 0, 0x0002, 0)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(cur, other, False)
        if not is_foreground(hwnd):
            HWND_TOPMOST, HWND_NOTOPMOST, flags = -1, -2, 0x0001 | 0x0002 | 0x0040   # NOSIZE|NOMOVE|SHOW
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            user32.SetForegroundWindow(hwnd)
        return is_foreground(hwnd)
    except Exception:
        return False
