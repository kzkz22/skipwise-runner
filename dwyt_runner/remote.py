"""Remote settings (round 7d; plan 15b, decisions 18.11, 18.16; protocol: "Távoli beállítás").

The runner reports its settings description, its non-secret values (as text, the way `config set` takes
them) and its public key; the owner changes them on the dwyt Runners → Beállítás page. The central only
forwards: every value is checked here with settings.parse, only fields marked `remote` are accepted, and
keys arrive encrypted with this runner's public key (RSA-OAEP, SHA-256), so the central never sees them.

On by default for the command-line runner (`run`, a server), off in the desktop app (its own window is
where it is set); `remote_settings` = auto | on | off, never changeable from dwyt.
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any

import httpx

from . import __version__, config, settings, tools
from .config import Config
from .i18n import gettext as _
from .i18n import N_, labels

log = logging.getLogger("dwyt-runner")
APP = False  # set by the desktop app: "auto" means off there
SOURCE = "https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip"
VERSION_URL = "https://raw.githubusercontent.com/kzkz22/skipwise-runner/main/pyproject.toml"
LATEST_EVERY_S = 3600  # with every hourly report: the Update now button shows within the hour (0.28.2)
REPORT_EVERY_S = 3600
RESTART_EXIT = 75  # systemd (Restart=on-failure) starts the new version
MODE_TEXT = labels({"felho": N_("átírás felhőben (AssemblyAI), összefoglaló felhőben"),
             "vegyes": N_("átírás ezen a gépen, összefoglaló felhőben"),
             "sajat": N_("átírás és összefoglaló ezen a gépen"),
             "egyeni": N_("a lenti beállítások szerint (például külön GPU-gép, saját LiteLLM)")})

_latest: dict[str, Any] = {"version": "", "at": 0.0}


class RestartNow(SystemExit):
    """The new version is installed: leave the loop and start again."""


def enabled(cfg: Config) -> bool:
    if cfg.remote_settings == "on":
        return True
    if cfg.remote_settings == "off":
        return False
    return not APP


def install_kind() -> str:
    return "app" if APP else "service"


# ---- the runner's own key pair ---------------------------------------------------------------------

def key_path():
    return config.home() / "remote_key.pem"


def _private_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    path = key_path()
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    os.replace(tmp, path)
    return key


def public_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    return _private_key().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def decrypt(blob: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    data = _private_key().decrypt(base64.b64decode(blob, validate=True), padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return data.decode("utf-8")


# ---- what is reported ----------------------------------------------------------------------------

def as_text(kind: str, value: Any) -> str:
    """A value the way `config set` (settings.parse) takes it back."""
    if value is None:
        return ""
    if kind == "bool":
        return "igen" if value else "nem"
    if kind == "list":
        return ", ".join(str(v) for v in value)
    if kind == "map":
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if kind == "float":
        return f"{value:g}"
    return str(value)


def values(cfg: Config) -> dict[str, str]:
    return {s.key: as_text(s.kind, getattr(cfg, s.key)) for s in settings.SETTINGS if s.kind != "secret"}


def mode_presets(cfg: Config) -> list[dict[str, Any]]:
    from . import modes
    now = values(cfg)
    out = []
    for mode_id, title in modes.MODES.items():
        trial = replace(cfg)
        if mode_id != "egyeni":
            modes.apply(trial, mode_id)
        changed = {k: v for k, v in values(trial).items() if v != now.get(k)}
        out.append({"id": mode_id, "title": title, "text": MODE_TEXT.get(mode_id, ""), "values": changed})
    return out


def latest_version(timeout: float = 10) -> str:
    """The newest version an update would install, checked hourly (once a day kept the central's Update now
    button hidden for up to a day after a release); '' if unknown. With the project's key: the newest signed
    release (a small file on GitHub's release downloads, no API quota); before that the source's main branch.
    Installing on its own (auto_update) still follows the updater's daily check."""
    from . import updater
    if updater.signed():
        if time.time() - updater.STATE["checked"] >= LATEST_EVERY_S:
            updater.check()
        return updater.STATE["latest"]
    if time.time() - _latest["at"] < LATEST_EVERY_S:
        return _latest["version"]
    _latest["at"] = time.time()
    try:
        resp = httpx.get(VERSION_URL, timeout=timeout, headers={"User-Agent": f"dwyt-runner/{__version__}"},
                         follow_redirects=True)
        resp.raise_for_status()
        found = re.search(r'^version\s*=\s*"([^"]+)"', resp.text, re.M)
        _latest["version"] = found.group(1) if found else ""
    except (httpx.HTTPError, ValueError) as exc:
        log.info("latest version: %s", exc)
    return _latest["version"]


def build_report(cfg: Config) -> dict[str, Any]:
    body: dict[str, Any] = {"runner_version": __version__, "latest_version": latest_version() or None,
                            "install": install_kind(), "remote": enabled(cfg)}
    if body["remote"]:
        from . import modes
        body.update(public_key=public_pem(), describe=settings.describe(), modes=mode_presets(cfg),
                    mode=modes.current(cfg), values=values(cfg),
                    secrets_set=sorted(k for k in config.SECRET_KEYS if k != "token" and getattr(cfg, k)))
    return body


def report(cfg: Config, central) -> int:
    """Send the report; returns how many changes wait (0 when the central is too old to know about it)."""
    return central.report_settings(build_report(cfg))


# ---- applying a change ------------------------------------------------------------------------------

def _allowed(key: str, secret: bool) -> settings.Setting | None:
    s = settings.BY_KEY.get(key)
    if s is None or not s.remote or s.readonly or (s.kind == "secret") != secret:
        return None
    return s


def apply(cfg: Config, change: dict[str, Any]) -> dict[str, Any]:
    """Apply one change to cfg (and save it); the result for the central."""
    applied: list[str] = []
    errors: dict[str, str] = {}
    for key, text in (change.get("values") or {}).items():
        s = _allowed(key, secret=False)
        if s is None:
            errors[key] = _("ez a beállítás a Skipwise-ból nem állítható")
            continue
        try:
            setattr(cfg, key, settings.parse(key, str(text)))
            applied.append(key)
        except ValueError as exc:
            errors[key] = str(exc)
    for key, blob in (change.get("secrets") or {}).items():
        if _allowed(key, secret=True) is None:
            errors[key] = _("ez a kulcs a Skipwise-ból nem állítható")
            continue
        try:
            setattr(cfg, key, decrypt(str(blob)).strip())
            applied.append(key)
        except Exception:  # noqa: BLE001 - wrong key pair, damaged text: the owner gives it again
            errors[key] = _("a kulcsot nem sikerült megfejteni; add meg újra a Skipwise-ban")
    for key in change.get("clear") or []:
        if _allowed(key, secret=True) is None:
            errors[key] = _("ez a kulcs a Skipwise-ból nem törölhető")
            continue
        setattr(cfg, key, "")
        applied.append(key)
    if applied:
        config.save(cfg)
    checks: dict[str, list] = {}
    from . import keystore
    for key, check in (("assemblyai_key", keystore.check_assemblyai), ("llm_key", keystore.check_llm)):
        if key in applied and getattr(cfg, key):
            ok, text = check(cfg)
            checks[key] = [ok, text]
    if applied or errors:
        log.info("settings from dwyt: applied %s, refused %s", applied, sorted(errors))
    return {"ok": not errors, "applied": applied, "errors": errors, "checks": checks}


def update_command() -> list[str] | None:
    from .ytdlp_update import find_uv
    uv = find_uv()
    if uv is None:
        return None
    extras = [name for name, module in (("local", "sherpa_onnx"), ("desktop", "webview"))
              if importlib.util.find_spec(module) is not None]
    spec = f"dwyt-runner[{','.join(extras)}] @ {SOURCE}" if extras else f"dwyt-runner @ {SOURCE}"
    return [uv, "tool", "install", "--force", "--refresh", "--python", "3.13", spec]


def update() -> dict[str, Any]:
    """Frissítés most: the newest signed release (decision 18.20); before the project's key exists, the same
    command as a manual update (decision 18.16)."""
    if APP:
        return {"ok": False, "message": _("Az asztali alkalmazás a saját gépén frissül.")}
    from . import updater
    if updater.signed():
        ok, message, action = updater.update_now()
        return {"ok": ok and bool(action), "message": message, "action": action}
    cmd = update_command()
    if cmd is None:
        return {"ok": False, "message": _("Nincs uv ezen a gépen, amivel frissíthetnék.")}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200,
                              check=False, **tools.no_window())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "message": _("A frissítés nem futott le: {exc}").format(exc=exc)}
    if proc.returncode != 0:
        return {"ok": False, "message": ((proc.stderr or proc.stdout).strip()[-300:]
                                         or _("kilépési kód {returncode}").format(returncode=proc.returncode))}
    return {"ok": True, "message": _("Telepítve ({__version__} helyett a legújabb); újraindulok.").format(
        __version__=__version__)}


def restart() -> None:
    """Start the new version: under systemd exit and let it restart us, else replace this process."""
    if os.environ.get("INVOCATION_ID"):  # systemd sets this for its services
        raise RestartNow(RESTART_EXIT)
    if sys.platform == "win32":
        subprocess.Popen([sys.argv[0], *sys.argv[1:]], **tools.no_window())  # noqa: S603
        raise RestartNow(0)
    os.execv(sys.argv[0], sys.argv)  # noqa: S606


def handle(cfg: Config, central) -> Config:
    """Take the waiting changes, apply them one by one, answer each, report again. May restart (update)."""
    restart_after, action = False, ""
    for change in central.setting_changes():
        if change.get("action") == "update":
            result = update()
            restart_after = restart_after or result["ok"]
            action = result.pop("action", "restart")
        else:
            result = apply(cfg, change)
        central.setting_result(change["id"], result)
    cfg = config.load()
    report(cfg, central)
    if restart_after:
        from . import updater
        updater.finish(action or "restart")
    return cfg
