"""Remote settings (round 7d): report, apply changes from dwyt, keys encrypted end to end, Frissítés most,
and the one-line installer's `adopt`."""
import base64
import io
import json
import subprocess
import sys
import threading

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from dwyt_runner import cli, config, remote, settings, worker
from dwyt_runner.protocol import Central


def encrypt(text: str) -> str:
    key = serialization.load_pem_public_key(remote.public_pem().encode())
    return base64.b64encode(key.encrypt(text.encode(), padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))).decode()


def test_every_value_goes_out_as_text_and_comes_back():
    cfg = config.Config(allowed_domains=["youtube.com", "*.example.org"], llm_model_notes={"a": "első", "b": "x"},
                        diarize_threshold=0.75, polite_mode=False)
    for s in settings.SETTINGS:
        if s.kind == "secret" or s.readonly:
            continue
        text = remote.as_text(s.kind, getattr(cfg, s.key))
        assert settings.parse(s.key, text) == getattr(cfg, s.key), s.key


def test_the_report_carries_no_secret(cfg, central):
    config.save(cfg)
    body = remote.build_report(cfg)
    assert body["remote"] is True and body["install"] == "service"
    assert body["public_key"].startswith("-----BEGIN PUBLIC KEY-----")
    assert body["values"]["allowed_domains"].startswith("youtube.com, youtu.be, telex.hu")
    assert body["values"]["polite_mode"] == "igen"
    assert body["secrets_set"] == ["assemblyai_key", "llm_key"]
    text = json.dumps(body)
    assert "aai-key" not in text and "tok-1" not in text and '"k"' not in text
    assert {m["id"] for m in body["modes"]} == {"felho", "vegyes", "sajat", "egyeni"}
    assert any(s["key"] == "remote_settings" and not s["remote"] for g in body["describe"]["groups"]
               for s in g["settings"])
    remote.APP = True  # the desktop app: off unless switched on there
    assert remote.build_report(cfg) == {"runner_version": body["runner_version"], "latest_version": None,
                                        "install": "app", "remote": False}
    cfg.remote_settings = "on"
    assert remote.build_report(cfg)["remote"] is True


def test_a_change_from_dwyt_is_checked_applied_and_answered(cfg, central):
    config.save(cfg)
    central.changes = [{"id": 5, "values": {"polite_wait_min": "15", "allowed_domains": "*",
                                            "whisper_cli": "/tmp/evil", "max_file_mb": "nagyon sok"},
                        "secrets": {"llm_key": encrypt("llm-key")}, "clear": ["assemblyai_key"], "action": None}]
    worker.loop(config.load(), threading.Event(), Central(central.url, "tok-1"), once=True)
    result = central.change_results[5]
    assert sorted(result["applied"]) == ["allowed_domains", "assemblyai_key", "llm_key", "polite_wait_min"]
    assert result["errors"]["whisper_cli"] == "ez a beállítás a Skipwise-ból nem állítható"
    assert "számot várok" in result["errors"]["max_file_mb"]
    assert result["checks"]["llm_key"][0] is True  # the key was tried at once
    saved = config.load()
    assert saved.polite_wait_min == 15 and saved.allowed_domains == ["*"] and saved.assemblyai_key == ""
    assert saved.llm_key == "llm-key" and saved.whisper_cli == ""
    assert central.reports[-1]["values"]["polite_wait_min"] == "15"  # reported again after the change
    assert central.reports[-1]["secrets_set"] == ["llm_key"]


def test_a_key_that_cannot_be_decrypted_and_a_field_the_runner_keeps(cfg):
    result = remote.apply(cfg, {"values": {"remote_settings": "off", "token": "x"},
                                "secrets": {"llm_key": base64.b64encode(b"garbage").decode(),
                                            "token": encrypt("steal")}})
    assert result["applied"] == [] and set(result["errors"]) == {"remote_settings", "token", "llm_key"}
    assert "megfejteni" in result["errors"]["llm_key"]


def test_switched_off_changes_are_not_taken(cfg, central):
    cfg.remote_settings = "off"
    config.save(cfg)
    central.changes = [{"id": 6, "values": {"polite_wait_min": "15"}}]
    worker.loop(config.load(), threading.Event(), Central(central.url, "tok-1"), once=True)
    assert central.change_results == {} and central.reports[-1] == {
        "runner_version": central.reports[-1]["runner_version"], "latest_version": None, "install": "service",
        "remote": False}
    assert config.load().polite_wait_min == 10


def test_an_older_central_without_remote_settings(cfg, central):
    central.no_settings = True
    config.save(cfg)
    worker.loop(config.load(), threading.Event(), Central(central.url, "tok-1"), once=True)
    assert len(central.claims) == 1  # the work goes on


def test_update_now(cfg, central, monkeypatch):
    from dwyt_runner import release
    monkeypatch.setattr(release, "PUBLIC_KEY", "")  # the unsigned way before the project's key (decision 18.16)
    config.save(cfg)
    ran, restarted = [], []
    monkeypatch.setattr(remote, "update_command", lambda: ["uv", "tool", "install", "--force", "--refresh", "x"])
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd, **kw: ran.append(cmd) or subprocess.CompletedProcess(
        cmd, 0, "", ""))
    monkeypatch.setattr(remote, "restart", lambda: restarted.append(True))
    central.changes = [{"id": 8, "values": {}, "secrets": {}, "clear": [], "action": "update"}]
    worker.loop(config.load(), threading.Event(), Central(central.url, "tok-1"), once=True)
    assert ran and ran[0][:5] == ["uv", "tool", "install", "--force", "--refresh"]
    assert central.change_results[8]["ok"] is True and restarted == [True]
    remote.APP = True  # the desktop app updates on its own machine
    assert remote.update()["ok"] is False


def test_restart_under_systemd_exits_for_a_restart(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "abc")
    with pytest.raises(SystemExit) as info:
        remote.restart()
    assert info.value.code == remote.RESTART_EXIT


def test_update_command_keeps_the_extras(monkeypatch):
    from dwyt_runner import ytdlp_update
    monkeypatch.setattr(ytdlp_update, "find_uv", lambda: "/home/u/.local/bin/uv")
    monkeypatch.setattr(remote.importlib.util, "find_spec", lambda name: object() if name == "sherpa_onnx" else None)
    cmd = remote.update_command()
    assert cmd[:6] == ["/home/u/.local/bin/uv", "tool", "install", "--force", "--refresh", "--python"]
    assert cmd[-1] == f"dwyt-runner[local] @ {remote.SOURCE}"


def test_adopt_keeps_the_installers_pairing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dwyt-runner", "adopt", "--server", "https://dwyt.example/", "--runner-id", "12",
                                      "--name", "lxc", "--token-stdin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("tok-xyz\n"))
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert info.value.code == 0
    cfg = config.load()
    assert (cfg.server, cfg.token, cfg.runner_id, cfg.name) == ("https://dwyt.example", "tok-xyz", 12, "lxc")
    assert "tok-xyz" not in capsys.readouterr().out


def test_the_window_switch(monkeypatch):
    import re

    from fastapi.testclient import TestClient

    from dwyt_runner.app.runtime import Runtime
    from dwyt_runner.app.server import create
    config.save(config.Config(server="https://dwyt.example", token="tok-1"))
    api, state = create(Runtime(start_worker=False), "testserver")
    client = TestClient(api)
    client.get(f"/belepes?t={state.launch_code}")
    page = client.get("/mukodes").text
    assert "A Skipwise-ból is állítható" in page and "https://dwyt.example/futtatok" in page
    assert 'name="remote_settings"' not in page  # not twice: only the switch
    token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    client.post("/mukodes/tavoli", data={"csrf": token, "on": "on"})
    assert config.load().remote_settings == "on" and remote.enabled(config.load())
    client.post("/mukodes/tavoli", data={"csrf": token})
    assert config.load().remote_settings == "off"


def test_a_new_release_is_seen_within_the_hour(monkeypatch):
    """The Update now button on the central shows only when the runner reports a newer version; a daily check
    kept it hidden for up to a day after a release (0.28.2)."""
    import time

    from dwyt_runner import updater
    calls = []
    monkeypatch.setattr(updater, "signed", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: calls.append(1) or updater.STATE.update(latest="9.9.9",
                                                                                           checked=time.time()))
    updater.STATE.update(latest="0.0.1", checked=time.time() - 2 * 3600)
    assert remote.latest_version() == "9.9.9" and calls == [1]
    assert remote.latest_version() == "9.9.9" and calls == [1]  # not again within the hour
