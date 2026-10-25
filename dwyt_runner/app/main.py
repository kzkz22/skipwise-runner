"""Start the desktop app: the worker in the background, the local page, a window (pywebview) or, without
it, the default browser, and the tray icon.

One app per user: the running one writes `window.json` (port + a random secret) next to the config; a
second start asks it to show its window and exits, so a shortcut click or a start at login never runs two
workers. Closing the window only hides it (plan 15b: background running, also where there is no tray);
the tray icon or a new start brings it back, "Kilépés" in the tray or in the window quits.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from .. import config
from ..i18n import gettext as _

log = logging.getLogger("dwyt-runner")
INSTANCE_HEADER = "X-Dwyt-Instance"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def instance_file() -> Path:
    return config.home() / "window.json"


APP_ID = "kzkz22.dwyt-runner"  # the same AppUserModelID as the installer's shortcuts: the taskbar shows their icon


def icon_file(kind: str = "png") -> Path:
    """The logo for the window, next to the settings: PNG for GTK / Qt, a bitmap .ico for Windows."""
    from .logo import ICO_NAME, image, write_ico
    path = config.home() / (ICO_NAME if kind == "ico" else "logo-256.png")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "ico":
            write_ico(path)
        else:
            image(256).save(path)
    return path


def window_icon(webview_module) -> dict:
    """`icon` for webview.start. Windows loads it with System.Drawing.Icon: it must be an .ico it can read (a PNG
    kept the window from appearing in 0.15.1), hence the bitmap .ico; the release self-test loads it the same way.
    macOS takes the icon from the app bundle, not from here."""
    if sys.platform == "darwin":
        return {}
    if "icon" not in inspect.signature(webview_module.start).parameters:
        return {}
    return {"icon": str(icon_file("ico" if sys.platform == "win32" else "png"))}


def set_app_id() -> None:
    """Windows: group the window under the shortcuts' AppUserModelID (their icon on the taskbar, not Python's)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        log.warning("AppUserModelID not set", exc_info=True)


def wake_running() -> bool:
    """True when another app of this user is running and was asked to show its window."""
    return ask_running("/ablak")


def ask_running(path: str) -> bool:
    """POST to the running app's instance endpoint (/ablak, /ablak/kilepes); True when it answered 204."""
    try:
        info = json.loads(instance_file().read_text(encoding="utf-8"))
        req = urllib.request.Request(f"http://127.0.0.1:{int(info['port'])}{path}", method="POST", data=b"",
                                     headers={INSTANCE_HEADER: str(info["secret"]),
                                              "Host": f"127.0.0.1:{int(info['port'])}"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a system proxy
        with opener.open(req, timeout=3) as r:
            return r.status == 204
    except Exception:  # noqa: BLE001 - no file, stale file, nobody listening
        return False


def quit_running(wait_s: float = 20) -> bool:
    """Close the running app of this user (uninstall); True when one was running and is gone."""
    try:
        pid = int(json.loads(instance_file().read_text(encoding="utf-8")).get("pid") or 0)
    except (OSError, ValueError):
        pid = 0
    if not ask_running("/ablak/kilepes"):
        return False
    wait_for(pid, wait_s)
    return True


def write_instance(port: int, secret: str) -> None:
    f = instance_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"port": port, "secret": secret, "pid": os.getpid()}), encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass


def remove_instance(secret: str) -> None:
    try:
        f = instance_file()
        if json.loads(f.read_text(encoding="utf-8")).get("secret") == secret:
            f.unlink()
    except Exception:  # noqa: BLE001
        pass


def serve(start_worker: bool = True, port: int = 0):
    """(uvicorn server, App state, Runtime) with the server running in a thread."""
    import uvicorn

    from .runtime import Runtime
    from .server import create
    port = port or free_port()
    runtime = Runtime(start_worker=start_worker)
    api, state = create(runtime, f"127.0.0.1:{port}")
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="window-server").start()
    for _n in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    return server, state, runtime


def alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for(pid: int, seconds: float = 60) -> None:
    """After an update: the old app must be gone before this one takes its place."""
    deadline = time.monotonic() + seconds
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.3)


def relaunch() -> None:
    """Start this app again (the new version after an update), hidden, once this process has ended."""
    import subprocess

    from .. import autostart, tools
    cmd = autostart.command() + ["--after", str(os.getpid())]
    subprocess.Popen(cmd, close_fds=True, start_new_session=os.name != "nt", **tools.no_window())


def run(browser: bool = False, hidden: bool = False, after: int = 0) -> int:
    if after:
        wait_for(after)
    if wake_running():
        print(_("A futtató már fut; előhoztam az ablakát."))
        return 0
    keep_login_start_current()
    from . import wizard
    if hidden and wizard.needed(config.load()):
        hidden = False  # the setup steps need the window
    server, state, runtime = serve()
    port = int(state.host.rsplit(":", 1)[1])
    write_instance(port, state.instance_secret)
    quitting = threading.Event()
    webview = None
    if not browser:
        try:
            import webview  # pywebview: WebView2 on Windows, WebKitGTK / Qt on Linux
        except ImportError:
            log.info("pywebview is not installed: opening the browser instead")
    window = None

    def show() -> None:
        if window is not None:
            window.show()
            try:
                window.restore()
            except Exception:  # noqa: BLE001 - not every backend has it
                pass
        else:
            webbrowser.open(state.new_start_url())

    def quit_app() -> None:
        quitting.set()
        if window is not None:
            window.destroy()

    from .tray import Tray
    tray = Tray(runtime.status, show, quit_app)
    state.on_show = show
    state.on_quit = quit_app

    def on_update(action: str) -> None:  # called by updater.finish (worker or window thread)
        if action == "restart":
            relaunch()
        quit_app()
    from .. import updater
    updater.ON_RESTART = on_update
    set_app_id()
    try:
        if webview is not None:
            width, height = window_size(webview)
            window = webview.create_window(_("Skipwise futtató"), state.start_url, width=width, height=height,
                                           min_size=(420, 520), hidden=hidden)

            def closing() -> bool:
                if quitting.is_set():
                    return True
                window.hide()  # keeps working; the tray icon or a new start brings the window back
                return False

            window.events.closing += closing
            # macOS: AppKit objects (the menu-bar icon) only on the main thread, before the run loop starts;
            # elsewhere the icon starts with the window (webview.start's func runs in a thread of its own)
            mac_tray = tray.start() if sys.platform == "darwin" else None

            def started() -> None:
                up = mac_tray if mac_tray is not None else tray.start()
                if not up and hidden:  # no tray: never start unseen
                    show()
            webview.start(func=started, **window_icon(webview))
        else:
            has_tray = tray.start()
            if not hidden or not has_tray:
                webbrowser.open(state.start_url)
            print(_("A futtató ablaka a böngészőben: {url}").format(url=state.start_url) + "\n"
                  + (_("Kilépés: az ablak Kilépés gombjával, a tálcaikon menüjéből vagy Ctrl+C") if has_tray
                     else _("Kilépés: az ablak Kilépés gombjával vagy Ctrl+C")))
            while not quitting.wait(1):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        tray.stop()
        runtime.shutdown()
        server.should_exit = True
        remove_instance(state.instance_secret)
    return 0


def keep_login_start_current() -> None:
    """After an update or a move the program may live elsewhere: point the login start at this one."""
    from .. import autostart
    try:
        if autostart.enabled():
            autostart.enable()
    except Exception:  # noqa: BLE001 - never a reason not to start
        log.debug("login start refresh", exc_info=True)


def entry() -> None:
    """`dwyt-runner-app`: the app without a console window (Windows: pythonw), logging to app.log."""
    import argparse
    import logging.handlers
    import sys
    ap = argparse.ArgumentParser(prog="skipwise-runner-app")
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--after", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if sys.stdout is None:  # pythonw: no console; some libraries (uvicorn) ask sys.stdout.isatty()
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    home = config.home()
    home.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(home / "app.log", maxBytes=1_000_000, backupCount=2,
                                                   encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        code = run(args.browser, args.hidden, args.after)
    except Exception as exc:  # noqa: BLE001 - without a console nobody would see it
        logging.getLogger("dwyt").exception("the app could not start")
        tell_failure(_("A Skipwise futtató nem indult el: {exc}\n\nRészletek: {log}").format(
            exc=exc, log=home / 'app.log'))
        code = 1
    sys.exit(code)


def tell_failure(text: str) -> None:
    """A start failure in a message box (Windows; the console-less app has nowhere else to say it)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, _("Skipwise futtató"), 0x10)
    except Exception:  # noqa: BLE001
        pass


def self_test() -> int:
    """`python -m dwyt_runner.app.main --self-test` (the release workflow): everything the window start needs is
    importable and callable, without opening a window or serving."""
    import webview  # noqa: F401 - the desktop extra is there

    from .runtime import Runtime
    from .server import create
    create(Runtime(start_worker=False), "127.0.0.1:1")
    width, height = window_size(webview)
    from .tray import image
    image("green", 32)
    icon = window_icon(webview).get("icon", "")
    if sys.platform == "win32" and icon:  # what pywebview's WinForms window does with it (0.15.1 broke here)
        import clr
        clr.AddReference("System.Drawing")
        from System.Drawing import Icon
        Icon(icon).Dispose()
    print(f"ok {width}x{height} icon={Path(icon).name or '-'}")
    return 0


WINDOW = (1040, 800)  # the wanted size; never more than the screen allows (a laptop at 150 % is ~1280 x 680)


def window_size(webview_module) -> tuple[int, int]:
    """The window fits the screen with room for the taskbar, so its bottom (and the Tovább button) is visible."""
    try:
        screen = webview_module.screens[0]
        sw, sh = int(screen.width), int(screen.height)
    except Exception:  # noqa: BLE001 - not every backend tells the screen before start
        return WINDOW
    if sw <= 0 or sh <= 0:
        return WINDOW
    return min(WINDOW[0], int(sw * 0.92)), min(WINDOW[1], int(sh * 0.86))


if __name__ == "__main__":
    entry()
