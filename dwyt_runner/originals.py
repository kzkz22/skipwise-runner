"""Kept originals of own recordings (plan 19c, decision 7 of the plan; decision 19.32).

A recording uploaded in the browser reaches this runner once (downloaded from the central service, which then deletes
its copy). With `keep_originals` the runner keeps it in the "Eredetik" folder under its data home, and tells the
central service which files it holds (`PUT /runner/v1/originals`). A new transcript of such an item (another mode, the
knowledge base's terms) is then a job of source kind `local` for this runner, like a watched-folder file: nothing is
uploaded again. Above `originals_max_gb` the least recently used files go (never the one just kept); the report says
which ones and why. The user deletes a file from the item page or the Settings page: a job of kind `drop_original`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from . import config
from .config import Config

log = logging.getLogger("dwyt.originals")
_LOCK = threading.Lock()
SHA = re.compile(r"^[0-9a-f]{64}$")


def folder() -> Path:
    return config.data_home() / "originals"


def index_path() -> Path:
    return config.home() / "originals.json"


def load() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    return {k: v for k, v in (files or {}).items() if SHA.match(k) and isinstance(v, dict)}


def _save(files: dict[str, dict[str, Any]]) -> None:
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"files": files}, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _file(sha256: str, entry: dict[str, Any]) -> Path:
    return folder() / (sha256 + str(entry.get("ext") or ""))


def _ext(name: str) -> str:
    ext = os.path.splitext(name or "")[1].lower()
    return ext if re.fullmatch(r"\.[a-z0-9]{1,8}", ext) else ""


def path_for(sha256: str) -> str | None:
    """The kept file with this content, if it is still there (and its use is remembered)."""
    if not SHA.match(sha256 or ""):
        return None
    with _LOCK:
        files = load()
        entry = files.get(sha256)
        if entry is None:
            return None
        path = _file(sha256, entry)
        if not path.is_file():
            files.pop(sha256, None)
            _save(files)
            return None
        entry["used_at"] = time.time()
        _save(files)
        return str(path)


def keep(cfg: Config, sha256: str, name: str, src: str) -> tuple[str, list[dict[str, Any]]]:
    """Move a downloaded (already checked) upload into the folder; returns its new path and what the limit removed."""
    if not cfg.keep_originals or not SHA.match(sha256):
        return src, []
    with _LOCK:
        files = load()
        entry = {"name": os.path.basename(name or "")[:200] or sha256[:12], "ext": _ext(name),
                 "size": os.path.getsize(src), "kept_at": time.time(), "used_at": time.time()}
        dest = _file(sha256, entry)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(src, dest)
        except OSError as exc:  # a full disk must not cost the transcription
            log.warning("original %s not kept: %s", sha256[:12], exc)
            return src, []
        files[sha256] = entry
        removed = _prune(cfg, files, keep=sha256)
        _save(files)
        return str(dest), removed


def _prune(cfg: Config, files: dict[str, dict[str, Any]], keep: str = "") -> list[dict[str, Any]]:
    limit = max(int(cfg.originals_max_gb or 0), 0) * 1024 ** 3
    removed = []
    total = sum(int(e.get("size") or 0) for e in files.values())
    for sha, entry in sorted(files.items(), key=lambda kv: kv[1].get("used_at") or 0):
        if total <= limit:
            break
        if sha == keep:
            continue
        try:
            _file(sha, entry).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("original %s not removed: %s", sha[:12], exc)
            continue
        total -= int(entry.get("size") or 0)
        files.pop(sha)
        removed.append({"sha256": sha, "name": entry.get("name"), "reason": "limit"})
    return removed


def drop(sha256: str) -> bool:
    """Delete one kept file (the user asked for it). False: it was not here."""
    with _LOCK:
        files = load()
        entry = files.pop(sha256, None)
        if entry is None:
            return False
        try:
            _file(sha256, entry).unlink(missing_ok=True)
        except OSError:
            files[sha256] = entry
            raise
        _save(files)
        return True


def report(cfg: Config, removed: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """What `PUT /runner/v1/originals` sends: every kept file (missing ones dropped first), the limit, what went."""
    with _LOCK:
        files = load()
        gone = [sha for sha, e in files.items() if not _file(sha, e).is_file()]
        for sha in gone:
            files.pop(sha)
        if gone:
            _save(files)
    return {"keep": bool(cfg.keep_originals), "max_bytes": max(int(cfg.originals_max_gb or 0), 0) * 1024 ** 3,
            "originals": [{"sha256": sha, "name": e.get("name"), "size": int(e.get("size") or 0),
                           "kept_at": int(e.get("kept_at") or 0), "used_at": int(e.get("used_at") or 0)}
                          for sha, e in sorted(files.items(), key=lambda kv: -(kv[1].get("kept_at") or 0))],
            "removed": (removed or []) + [{"sha256": sha, "reason": "missing"} for sha in gone]}


def send(cfg: Config, central: Any, removed: list[dict[str, Any]] | None = None) -> bool:
    """Tell the central service; an older central (404) just does not show them."""
    try:
        return bool(central.originals(report(cfg, removed)))
    except Exception as exc:  # noqa: BLE001 - a report never stops the work
        log.warning("originals not reported: %s", exc)
        return False
