"""The desktop window (round 7c): its local page, modes, activity, pairing."""
import re
import sys
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from dwyt_runner import activity, config, gpuswap, modes
from dwyt_runner.app.runtime import Runtime
from dwyt_runner.app.server import COOKIE, create
from conftest import KEYRING


@pytest.fixture
def window():
    runtime = Runtime(start_worker=False)
    api, state = create(runtime, "testserver")
    client = TestClient(api)
    return client, state, runtime


def enter(client, state):
    r = client.get(f"/belepes?t={state.launch_code}", follow_redirects=False)
    assert r.status_code == 303 and COOKIE in r.cookies
    return client


def csrf(page: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)


def test_only_the_window_that_opened_it_gets_in(window):
    client, state, _ = window
    assert client.get("/allapot").status_code == 403  # no session
    code = state.launch_code
    enter(client, state)
    assert client.get("/allapot").status_code == 200
    other = TestClient(client.app)
    assert other.get(f"/belepes?t={code}").status_code == 403  # one time only
    assert client.get("/allapot", headers={"host": "evil.example:80"}).status_code == 403  # DNS rebinding
    r = client.get("/allapot")
    assert "script-src 'self'" in r.headers["content-security-policy"] and "<script>" not in r.text
    assert client.post("/mukodes", data={"mode": "felho"}).status_code == 403  # no CSRF token


def test_pairing_from_the_window(window, central):
    client, state, runtime = enter(*window[:2]) and window
    page = client.get("/allapot").text
    assert "Párosítás a Skipwise-zal" in page
    central.approved = False
    r = client.post("/parositas", data={"csrf": csrf(page), "server": central.url, "name": "laptop"},
                    follow_redirects=True)
    assert "ABC-234" in r.text and 'data-live="1"' in r.text
    central.approved = True
    for _ in range(50):
        if runtime.pairing.get("state") == "done":
            break
        time.sleep(0.2)
    cfg = config.load()
    assert (cfg.server, cfg.token, cfg.name) == (central.url, "tok-1", "laptop")
    assert runtime.running()  # the worker starts right after pairing
    runtime.shutdown()
    runtime.thread.join(timeout=60)
    assert not runtime.running()


def test_status_shows_the_current_job_and_today(window):
    client, state, _ = enter(*window[:2]) and window
    config.save(config.Config(server="https://dwyt.example", token="tok-1", name="otthoni gép"))
    job = {"job_id": 5, "kind": "transcribe", "input": {"source": {"kind": "youtube", "url": "https://youtu.be/x"},
                                                         "options": {"mode": "local"}}}
    activity.start(job, config.Config(whisper_builtin=True))
    activity.stage("diarize")
    gpuswap.STATE.update(phase="whisper", models=["qwen3:8b"])
    page = client.get("/allapot").text
    assert "beszélők szétválasztása" in page and "ezen a gépen" in page and "Dolgozom" in page
    activity.finish("done")
    gpuswap.after_whisper()
    data = client.get("/api/allapot").json()
    assert data["current"] == {} and data["today"] == {"transcribe": 1} and data["recent"][0]["kind_label"] == "Átírás"
    assert "Ma: 1 átírás" in client.get("/allapot").text


def test_modes_are_presets_read_back_from_the_settings(window):
    client, state, _ = enter(*window[:2]) and window
    cfg = config.Config(llm_url="https://openrouter.ai/api/v1", llm_public_model="osszefoglalo")
    assert modes.current(cfg) == "felho"
    assert modes.apply(replace(cfg), "vegyes") == ["Add meg az OpenRouter-kulcsot a Kulcsok lapon."]
    cfg.llm_key = "sk-or-1"
    assert modes.apply(cfg, "vegyes") == [] and modes.current(cfg) == "vegyes" and cfg.whisper_builtin
    assert modes.apply(cfg, "sajat") == ["Válaszd ki a helyi modellszervert (Ollama vagy LM Studio)."]
    assert "helyi modell írja" in modes.apply(config.Config(local_llm="ollama"), "sajat")[0]
    cfg.local_llm, cfg.llm_private_model = "ollama", "qwen3:8b"
    assert modes.apply(cfg, "sajat") == [] and modes.current(cfg) == "sajat"
    assert cfg.llm_url == "https://openrouter.ai/api/v1"  # the cloud address is kept
    boxes = modes.flow(cfg)
    assert (boxes[1].where, boxes[2].where) == ("ezen a gépen (Whisper)", "ezen a gépen (Ollama)")
    config.save(cfg)
    page = client.get("/mukodes").text
    assert 'value="sajat" checked' in page
    # every mode's picture is there, the chosen one shown: switching shows it before saving
    assert page.count("data-flow-for=") == 4 and '<div class="flow" data-flow-for="sajat">' in page
    assert '<div class="flow" data-flow-for="felho" hidden>' in page
    felho = page.split('data-flow-for="felho"')[1].split("data-flow-for")[0]
    assert "felhőben (OpenRouter)" in felho and "ezen a gépen (Whisper)" not in felho
    r = client.post("/mukodes", data={"csrf": csrf(page), "mode": "felho"}, follow_redirects=True)
    assert "AssemblyAI-kulcsot" in r.text  # what is still missing for cloud transcription
    back = config.load()
    assert not back.whisper_builtin and not back.llm_prefer_local and back.polite_mode is False


def test_detailed_settings_are_checked_and_never_touch_secrets(window):
    client, state, _ = enter(*window[:2]) and window
    config.save(config.Config(llm_key="sk-secret", local_llm="lmstudio"))
    page = client.get("/mukodes").text
    assert "sk-secret" not in page and 'data-when=' in page
    form = {"csrf": csrf(page), "name": "dolgozószoba", "polite_wait_min": "5", "diarize_threshold": "0,7",
            "local_llm": "lmstudio", "watch_mode": "local", "secret_store": "auto", "allowed_domains": "youtube.com",
            "llm_model_notes": "a=Egy\nb=Kettő", "polite_mode": "on"}
    client.post("/mukodes/reszletek", data=form)
    cfg = config.load()
    assert (cfg.name, cfg.polite_wait_min, cfg.diarize_threshold, cfg.llm_key) == ("dolgozószoba", 5, 0.7, "sk-secret")
    assert cfg.llm_model_notes == {"a": "Egy", "b": "Kettő"} and cfg.polite_mode and not cfg.whisper_builtin
    r = client.post("/mukodes/reszletek", data={**form, "polite_wait_min": "999"}, follow_redirects=True)
    assert "0 és 120 között" in r.text and config.load().polite_wait_min == 5


def test_keys_are_written_never_shown(window, assembly):
    client, state, _ = enter(*window[:2]) and window
    config.save(config.Config(assemblyai_url=assembly.url))
    page = client.get("/kulcsok").text
    assert "a rendszer kulcstárolója" in page
    r = client.post("/kulcsok", data={"csrf": csrf(page), "key": "assemblyai_key", "value": " aai-key "},
                    follow_redirects=True)
    assert "aai-key" not in r.text and "beállítva" in r.text and "próba: rendben" in r.text
    assert KEYRING.backend.data[("dwyt-runner", "assemblyai_key")] == "aai-key"
    assert client.post("/kulcsok", data={"csrf": csrf(page), "key": "token", "value": "x"}).status_code == 400


def wait_scan(runtime):
    for _ in range(100):
        if not runtime.machine["scanning"]:
            return
        time.sleep(0.05)


def test_machine_page_opens_at_once_from_the_last_scan(window, monkeypatch):
    from dwyt_runner import machine
    client, state, runtime = enter(*window[:2]) and window
    wait_scan(runtime)
    calls = []
    real = machine.scan
    monkeypatch.setattr(machine, "scan", lambda cfg=None: calls.append(1) or real(cfg))
    for _ in range(3):
        assert "Videókártya" in client.get("/gepem").text
    assert calls == []  # the page never waits for a scan
    page = client.get("/gepem").text
    client.post("/gepem/ujra", data={"csrf": csrf(page)})
    wait_scan(runtime)
    assert calls == [1] and client.get("/api/gep").json()["scanning"] is False
    runtime.machine.update(m=None)  # the very first scan still running: a waiting page that follows it
    assert "Megnézem, mi van a gépben" in client.get("/gepem").text


def test_closed_local_ports_answer_at_once():
    from dwyt_runner import localllm, machine
    started = time.monotonic()
    assert machine.port_open("http://127.0.0.1:9") is False
    assert localllm.Ollama("http://127.0.0.1:9").running() is False
    assert machine._get_json("http://127.0.0.1:9/api/version") is None
    assert time.monotonic() - started < 2


def test_machine_page_and_test_run(window, monkeypatch):
    client, state, runtime = enter(*window[:2]) and window
    wait_scan(runtime)
    page = client.get("/gepem").text
    assert "Videókártya" in page and "Próbafuttatás" in page
    from dwyt_runner import trial
    import threading
    release = threading.Event()
    monkeypatch.setattr(trial, "run", lambda cfg, model, stage, llm="": (stage("transcribe"), release.wait(10)))
    client.post("/gepem/proba", data={"csrf": csrf(page)}, follow_redirects=False)
    assert client.get("/api/proba").json()["running"] is True
    release.set()
    for _ in range(30):
        if not client.get("/api/proba").json()["running"]:
            break
        time.sleep(0.1)
    assert client.get("/api/proba").json() == {"running": False, "stage": "átírás", "error": ""}


def test_activity_titles():
    assert activity.title_of({"kind": "ask", "input": {"question": "Melyik olcsóbb?"}}) == "„Melyik olcsóbb?”"
    assert activity.title_of({"kind": "summarize", "input": {"meta": {"title": "Kerti videó"}}}) == "Kerti videó"
    assert activity.where_of({"kind": "transcribe", "input": {"options": {"mode": "cloud"}}}, config.Config()) == \
        "felhőben"


def test_tray_colour_follows_the_runner():
    from dwyt_runner.app import tray
    base = {"paired": True, "running": True, "error": "", "current": {}, "swap": {"phase": "idle"}}
    assert tray.state_of({**base, "paired": False})[0] == "grey"
    assert tray.state_of(base) == ("green", "Skipwise futtató – vár a következő munkára")
    busy = {**base, "current": {"kind_label": "Átírás", "stage_label": "beszédfelismerés"}}
    assert tray.state_of(busy) == ("blue", "Skipwise futtató – Átírás: beszédfelismerés")
    assert tray.state_of({**busy, "swap": {"phase": "waiting"}})[0] == "orange"  # swap wins over working
    colour, tip = tray.state_of({**base, "error": "x" * 500})
    assert colour == "red" and len(tip) <= 120  # Windows cuts tooltips at 128
    img = tray.image("orange")  # the logo with an orange dot in the corner (no more "d")
    assert img.size == (64, 64) and img.getpixel((52, 52))[:3] == tray.COLOURS["orange"]
    from dwyt_runner.app import logo
    assert all(abs(a - b) <= 2 for a, b in zip(img.getpixel((32, 4))[:3], logo.TILE))  # the tile (resampled)


def test_tray_icon_changes_and_says_when_the_runner_stopped():
    from dwyt_runner.app import tray

    class Icon:
        HAS_NOTIFICATION = True

        def __init__(self):
            self.icon, self.title, self.notes, self.menus = None, "", [], 0

        def notify(self, text, title):
            self.notes.append(text)

        def update_menu(self):
            self.menus += 1

    t = tray.Tray(lambda: {}, lambda: None, lambda: None)
    t.icon, t.colour = Icon(), "green"
    base = {"paired": True, "running": True, "error": "", "current": {}, "swap": {}}
    t.update({**base, "swap": {"phase": "swapping"}})
    assert t.colour == "orange" and "modellcsere" in t.icon.title and t.icon.notes == []
    t.update({**base, "error": "A dwyt nem fogadja el a párosítást."})
    assert t.colour == "red" and t.icon.notes == ["leállt: A dwyt nem fogadja el a párosítást."]
    t.update({**base, "error": "A dwyt nem fogadja el a párosítást."})
    assert len(t.icon.notes) == 1  # said once, not every poll
    assert t._tip().startswith("Leállt")


def test_a_second_start_shows_the_running_window(tmp_path):
    from dwyt_runner.app import main
    assert main.wake_running() is False  # nothing runs yet
    server, state, runtime = main.serve(start_worker=False)
    try:
        shown = []
        state.on_show = lambda: shown.append(1)
        port = int(state.host.rsplit(":", 1)[1])
        main.write_instance(port, "wrong")
        assert main.wake_running() is False and shown == []  # a stale or foreign file is not obeyed
        main.write_instance(port, state.instance_secret)
        assert main.wake_running() is True and shown == [1]
        main.remove_instance("someone else's")
        assert main.instance_file().exists()
        main.remove_instance(state.instance_secret)
        assert not main.instance_file().exists()
        first = state.start_url
        assert state.new_start_url() != first  # the browser mode's tray opens a fresh one-time entry
    finally:
        runtime.shutdown()
        server.should_exit = True


def test_quit_from_the_window(window):
    client, state, _ = enter(*window[:2]) and window
    quit_called = []
    state.on_quit = lambda: quit_called.append(1)
    page = client.get("/allapot").text
    assert 'action="/kilepes"' in page
    assert client.post("/kilepes", data={"csrf": "x"}).status_code == 403
    r = client.post("/kilepes", data={"csrf": csrf(page)})
    assert "A futtató leállt" in r.text
    for _ in range(30):
        if quit_called:
            break
        time.sleep(0.1)
    assert quit_called == [1]


def test_login_start_linux_and_mac(tmp_path, monkeypatch):
    from dwyt_runner import autostart
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(autostart.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(autostart, "command", lambda platform=None: ["/opt/my apps/dwyt-runner", "app", "--hidden"])
    assert autostart.enabled("linux") is False
    autostart.enable("linux")
    text = (tmp_path / "cfg" / "autostart" / "dwyt-runner.desktop").read_text(encoding="utf-8")
    assert 'Exec="/opt/my apps/dwyt-runner" app --hidden' in text and autostart.enabled("linux")
    autostart.disable("linux")
    assert not autostart.enabled("linux")
    autostart.disable("linux")  # twice is fine
    autostart.enable("darwin")
    plist = (tmp_path / "home" / "Library" / "LaunchAgents" / "hu.dwyt.runner.plist").read_text(encoding="utf-8")
    assert "<string>/opt/my apps/dwyt-runner</string><string>app</string>" in plist
    assert "<key>RunAtLoad</key><true/>" in plist
    assert autostart.enabled("darwin")
    autostart.set_enabled(False, "darwin")
    assert not autostart.enabled("darwin")


def test_login_start_windows_registry(monkeypatch):
    import sys
    import types
    from dwyt_runner import autostart
    values: dict = {}

    class Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = types.SimpleNamespace(HKEY_CURRENT_USER="HKCU", KEY_SET_VALUE=2, REG_SZ=1,
                                 CreateKey=lambda root, path: Key(path),
                                 OpenKey=lambda root, path, *a: Key(path),
                                 SetValueEx=lambda k, name, _r, _t, v: values.__setitem__((k.path, name), v),
                                 QueryValueEx=lambda k, name: (values[(k.path, name)], 1) if (k.path, name) in values
                                 else (_ for _ in ()).throw(FileNotFoundError()),
                                 DeleteValue=lambda k, name: values.pop((k.path, name)) if (k.path, name) in values
                                 else (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    exe = r"C:\Users\Teszt Elek\.local\bin\dwyt-runner-app.exe"
    monkeypatch.setattr(autostart, "command", lambda platform=None: [exe, "--hidden"])
    assert not autostart.enabled("win32")
    autostart.enable("win32")
    assert values == {(autostart.RUN_KEY, "dwyt-runner"):
                      f'"{exe}" --hidden'}
    assert autostart.enabled("win32")
    autostart.disable("win32")
    autostart.disable("win32")
    assert values == {} and not autostart.enabled("win32")


def test_login_start_command_prefers_the_console_less_app(tmp_path, monkeypatch):
    import sys
    from dwyt_runner import autostart
    (tmp_path / "dwyt-runner-app.exe").write_text("")
    (tmp_path / "dwyt-runner").write_text("")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "dwyt-runner")])
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))  # not the test machine's own scripts
    monkeypatch.setattr(autostart.shutil, "which", lambda name: None)
    assert autostart.command("win32") == [str((tmp_path / "dwyt-runner-app.exe").resolve()), "--hidden"]
    assert autostart.command("linux") == [str((tmp_path / "dwyt-runner").resolve()), "app", "--hidden"]
    (tmp_path / "skipwise-runner-app.exe").write_text("")  # 0.17: the new names come first
    (tmp_path / "skipwise-runner").write_text("")
    assert autostart.command("win32") == [str((tmp_path / "skipwise-runner-app.exe").resolve()), "--hidden"]
    assert autostart.command("linux") == [str((tmp_path / "skipwise-runner").resolve()), "app", "--hidden"]


def test_login_start_uses_the_installers_skipwise_exe(tmp_path, monkeypatch):
    """The Windows package: python\\Skipwise.exe (pythonw under our name) starts the app at login and after updates."""
    import sys
    from dwyt_runner import autostart
    (tmp_path / "python.exe").write_text("")
    (tmp_path / "pythonw.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "python.exe")])
    monkeypatch.setattr(autostart.shutil, "which", lambda name: None)  # not the test machine's own scripts
    assert autostart.command("win32")[0] == str(tmp_path / "pythonw.exe")
    (tmp_path / "Skipwise.exe").write_text("")
    assert autostart.command("win32") == [str(tmp_path / "Skipwise.exe"), "-m", "dwyt_runner.app.main", "--hidden"]


def test_the_command_is_skipwise_runner_and_the_old_name_stays(capsys):
    """0.17: the command and the repository carry the new name; `dwyt-runner` keeps working."""
    import tomllib
    from pathlib import Path

    from dwyt_runner import cli, release, remote
    meta = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["scripts"]["skipwise-runner"] == meta["scripts"]["dwyt-runner"] == "dwyt_runner.cli:main"
    assert meta["gui-scripts"]["skipwise-runner-app"] == meta["gui-scripts"]["dwyt-runner-app"]
    assert release.REPO == "kzkz22/skipwise-runner" and "kzkz22/skipwise-runner" in remote.SOURCE
    import sys
    old = sys.argv
    sys.argv = ["skipwise-runner", "--help"]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        sys.argv = old
    assert capsys.readouterr().out.startswith("usage: skipwise-runner")


def test_login_start_switch_in_the_window(window, tmp_path, monkeypatch):
    from dwyt_runner import autostart
    client, state, _ = enter(*window[:2]) and window
    on = []
    monkeypatch.setattr(autostart, "enabled", lambda platform=None: bool(on))
    monkeypatch.setattr(autostart, "set_enabled", lambda v, platform=None: on.append(1) if v else on.clear())
    page = client.get("/mukodes").text
    assert 'action="/mukodes/inditas"' in page and 'name="on" data-autosubmit' in page.replace(" checked", "")
    client.post("/mukodes/inditas", data={"csrf": csrf(page), "on": "on"})
    assert on and 'name="on" checked data-autosubmit' in client.get("/mukodes").text
    client.post("/mukodes/inditas", data={"csrf": csrf(page)})
    assert not on


def test_the_test_run_keeps_the_model_it_was_run_with(window):
    from dwyt_runner import trial
    client, state, runtime = enter(*window[:2]) and window
    wait_scan(runtime)
    config.save(config.Config(local_llm="lmstudio", llm_private_model="qwen/qwen3-8b"))
    runtime.machine["local_models"] = ["qwen/qwen3-8b", "google/gemma-4-e4b"]
    page = client.get("/gepem").text
    assert '<option value="qwen/qwen3-8b" selected>' in page  # nothing tried yet: the set model
    trial.save(trial.TrialResult(at=time.time(), gpu="", whisper_model="large-v3", llm_model="google/gemma-4-e4b"))
    page = client.get("/gepem").text
    assert '<option value="google/gemma-4-e4b" selected>' in page and page.count("<option") == 2


def test_the_logo_everywhere(window, tmp_path):
    """Decision 19.16: the window's page, the installer icons and the window icon use the dwyt logo."""
    from PIL import Image

    from dwyt_runner.app import logo, main
    client, state, _runtime = window
    page = enter(client, state).get("/allapot").text
    assert 'class="app-logo" src="/static/logo.svg"' in page and 'rel="icon" href="/static/logo.svg"' in page
    assert client.get("/static/logo.svg").text.strip() == logo.SVG.strip()  # Windows checkouts: CRLF
    logo.write_icons(tmp_path)
    ico = Image.open(tmp_path / logo.ICO_NAME)
    assert ico.info["sizes"] == {(s, s) for s in logo.ICO_SIZES}
    assert logo.ico_frames(tmp_path / logo.ICO_NAME) == [(s, False) for s in logo.ICO_SIZES]  # bitmaps, no PNG
    assert not (tmp_path / "dwyt-runner.ico").exists()  # the old name stays in Windows' icon cache
    png = Image.open(tmp_path / "dwyt-runner.png")
    assert png.size == (256, 256) and png.getpixel((128, 24))[:3] == logo.TILE  # above the frame: the tile
    assert png.getpixel((0, 0))[3] == 0  # rounded corner: transparent
    assert main.icon_file().is_file()


def test_window_icon_is_a_bitmap_ico_on_windows(monkeypatch):
    """0.15.1 handed pywebview a PNG icon; on Windows that broke the window (System.Drawing.Icon). Windows now gets
    a bitmap .ico, GTK / Qt a PNG, macOS nothing (the bundle's icon)."""
    from dwyt_runner.app import logo, main

    class Webview:
        @staticmethod
        def start(func=None, icon=None):
            pass
    monkeypatch.setattr(main.sys, "platform", "darwin")
    assert main.window_icon(Webview) == {}
    monkeypatch.setattr(main.sys, "platform", "win32")
    ico = main.window_icon(Webview)["icon"]
    assert ico.endswith(logo.ICO_NAME) and all(not png for _s, png in logo.ico_frames(main.Path(ico)))
    monkeypatch.setattr(main.sys, "platform", "linux")
    assert main.window_icon(Webview)["icon"].endswith(".png")

    class Old:
        @staticmethod
        def start(func=None):
            pass
    assert main.window_icon(Old) == {}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_loads_the_window_icon_like_pywebview():
    """The real check behind 0.15.2: System.Drawing.Icon (what pywebview's WinForms window calls) reads our .ico."""
    clr = pytest.importorskip("clr")
    from dwyt_runner.app import main
    clr.AddReference("System.Drawing")
    from System.Drawing import Icon
    icon = Icon(str(main.icon_file("ico")))
    assert icon.Width > 0
    icon.Dispose()


def test_tray_changes_go_to_the_main_thread_on_macos(monkeypatch):
    """AppKit crashes when the menu-bar icon is touched from another thread (a Mac report, 2026-10-25)."""
    import sys
    import types
    from dwyt_runner.app import tray
    queued = []
    helper = types.SimpleNamespace(callAfter=queued.append)
    monkeypatch.setitem(sys.modules, "PyObjCTools", types.SimpleNamespace(AppHelper=helper))
    monkeypatch.setitem(sys.modules, "PyObjCTools.AppHelper", helper)
    ran = []
    monkeypatch.setattr(sys, "platform", "darwin")
    tray.on_main(lambda: ran.append("mac"))
    assert ran == [] and len(queued) == 1
    queued[0]()
    assert ran == ["mac"]
    monkeypatch.setattr(sys, "platform", "win32")
    tray.on_main(lambda: ran.append("now"))
    assert ran == ["mac", "now"]


def test_on_macos_the_tray_starts_on_the_main_thread_before_the_window():
    import inspect
    from dwyt_runner.app import main
    src = inspect.getsource(main.run)
    assert src.index('tray.start() if sys.platform == "darwin"') < src.index("webview.start(func=started")
