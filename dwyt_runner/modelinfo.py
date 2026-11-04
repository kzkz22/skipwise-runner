"""What runs behind the runner's model names, for the central's Runners page and model pickers.

With dwyt-airouter's LiteLLM the answer comes from its /model/info: each name carries a `dwyt` block
(written by `dwyt-airouter generate`) saying what the name is for, which model runs behind it,
locally or in the cloud, with thinking or without, and what happens when the GPU host is off.
Only those fields are passed on; keys and addresses never are. Other servers give no such
details; the owner's own notes (llm_model_notes) fill in the description.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from . import i18n
from .config import Config

REFRESH_S = 3600
_cache: dict[str, Any] = {"at": 0.0, "key": None, "value": ({}, "")}


def _text(value: Any, limit: int) -> str | None:
    return " ".join(str(value).split())[:limit] if isinstance(value, str) and value.strip() else None


def _clean(block: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, limit in (("description", 200), ("target", 160), ("target_label", 160), ("provider", 40), ("gpu", 60)):
        if _text(block.get(key), limit):
            out[key] = _text(block.get(key), limit)
    if block.get("where") in ("local", "cloud"):
        out["where"] = block["where"]
    for key in ("context", "thinking_budget"):
        if isinstance(block.get(key), int) and not isinstance(block.get(key), bool) and block[key] > 0:
            out[key] = block[key]
    for key in ("vision", "thinking", "trial"):
        if isinstance(block.get(key), bool):
            out[key] = block[key]
    return out


def _sanitize(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = _clean(block)
    fallback = block.get("fallback")
    if isinstance(fallback, list):
        out["fallback"] = [{**_clean(f), **({"name": _text(f.get("name"), 100)} if _text(f.get("name"), 100) else {})}
                           for f in fallback[:3] if isinstance(f, dict)]
    return out


def names(cfg: Config) -> list[str]:
    out: list[str] = []
    for name in [cfg.llm_public_model, cfg.llm_translate_model, cfg.llm_ask_model, cfg.llm_private_model,
                 *cfg.llm_models, *cfg.llm_private_models]:
        if name and name not in out:
            out.append(name)
    return out[:40]


def _root(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def fetch(cfg: Config, timeout: float = 5.0) -> tuple[dict[str, dict[str, Any]], str]:
    """(info per name, server label). Never raises: without details the names are simply undescribed."""
    wanted = names(cfg)
    info: dict[str, dict[str, Any]] = {}
    server = ""
    from . import gateway
    if cfg.llm_url and wanted and gateway.is_gateway(cfg.llm_url):  # no model list there: the runner knows its one
        for name in wanted:
            block = {"where": "cloud", "target_label": name, "provider": "AssemblyAI"}
            if name == gateway.TRIAL_MODEL:
                block.update(context=gateway.TRIAL_CONTEXT, trial=True, thinking=False,
                             description=i18n.gettext(gateway.TRIAL_NOTE)[:200])
            info[name] = block
        server = "AssemblyAI LLM Gateway"
    elif cfg.llm_url and wanted:
        headers = {"Authorization": f"Bearer {cfg.llm_key}"} if cfg.llm_key else {}
        for path in ("/v1/model/info", "/model/info"):
            try:
                resp = httpx.get(_root(cfg.llm_url) + path, headers=headers, timeout=timeout)
            except httpx.HTTPError:
                break
            if resp.status_code != 200:
                continue
            try:
                rows = resp.json().get("data") or []
            except ValueError:
                break
            server = "LiteLLM"
            for row in rows if isinstance(rows, list) else []:
                name = row.get("model_name") if isinstance(row, dict) else None
                if name in wanted and name not in info:
                    block = _sanitize((row.get("model_info") or {}).get("dwyt"))
                    if block:
                        info[name] = block
            break
        if not server:
            server = "OpenRouter" if "openrouter.ai" in cfg.llm_url else "OpenAI-kompatibilis szerver"
            from . import localllm
            if localllm.uses_local(cfg):
                server = localllm.server(cfg).title
    if server == "OpenRouter":  # everything there runs in the cloud; its model list tells context and thinking
        catalogue = openrouter_models(cfg, timeout)
        for name in wanted:
            block = info.setdefault(name, {"where": "cloud", "target_label": name, "provider": "OpenRouter"})
            row = catalogue.get(name)
            if row is None:
                continue
            if isinstance(row.get("context_length"), int) and row["context_length"] > 0:
                block["context"] = row["context_length"]
            thinking = thinking_of(row, cfg.llm_reasoning)
            if thinking is not None:
                block["thinking"] = thinking
    if server in ("Ollama", "LM Studio"):  # a model server on the runner's own machine
        for name in wanted:
            info.setdefault(name, {"where": "local", "target_label": name, "provider": server})
    for name, note in (cfg.llm_model_notes or {}).items():
        if name in wanted and _text(note, 200):
            info.setdefault(name, {})["description"] = _text(note, 200)
    return info, server


def openrouter_models(cfg: Config, timeout: float = 5.0) -> dict[str, dict[str, Any]]:
    """OpenRouter's public model list (id -> row), for context length and reasoning; {} when unreachable."""
    try:
        resp = httpx.get(cfg.llm_url.rstrip("/") + "/models", timeout=timeout,
                         headers={"Authorization": f"Bearer {cfg.llm_key}"} if cfg.llm_key else {})
        rows = resp.json().get("data") if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}
    return {r["id"]: r for r in rows or [] if isinstance(r, dict) and isinstance(r.get("id"), str)}


def thinking_of(row: dict[str, Any], setting: str) -> bool | None:
    """Will the model think with this setting? None: unknown (the model's own default, not told)."""
    reasoning = row.get("reasoning")
    if not isinstance(reasoning, dict):
        params = row.get("supported_parameters")
        if isinstance(params, list) and "reasoning" not in params:
            return False  # not a thinking model
        return None if setting == "auto" else setting == "on"
    if reasoning.get("mandatory") is True:
        return True
    if setting in ("on", "off"):
        return setting == "on"
    return reasoning.get("default_enabled") if isinstance(reasoning.get("default_enabled"), bool) else None


def request_extra(cfg: Config, model: str) -> dict[str, Any]:
    """The request field for llm_reasoning (OpenRouter only; 'auto' sends nothing). A model that always thinks
    is never asked not to: it would refuse the request."""
    if cfg.llm_reasoning not in ("on", "off") or "openrouter.ai" not in (cfg.llm_url or ""):
        return {}
    row = openrouter_models_cached(cfg).get(model) or {}
    reasoning = row.get("reasoning") if isinstance(row.get("reasoning"), dict) else {}
    if cfg.llm_reasoning == "off" and reasoning.get("mandatory") is True:
        return {}
    return {"reasoning": {"enabled": cfg.llm_reasoning == "on"}}


_models_cache: dict[str, Any] = {"at": 0.0, "url": "", "value": {}}


def openrouter_models_cached(cfg: Config) -> dict[str, dict[str, Any]]:
    if _models_cache["url"] != cfg.llm_url or time.monotonic() - _models_cache["at"] >= REFRESH_S:
        _models_cache.update(at=time.monotonic(), url=cfg.llm_url, value=openrouter_models(cfg))
    return _models_cache["value"]


def cached(cfg: Config) -> tuple[dict[str, dict[str, Any]], str]:
    """fetch(), at most once an hour (or when the model settings change)."""
    key = (cfg.llm_url, tuple(names(cfg)), tuple(sorted((cfg.llm_model_notes or {}).items())), cfg.llm_reasoning)
    if _cache["key"] != key or time.monotonic() - _cache["at"] >= REFRESH_S:
        _cache.update(at=time.monotonic(), key=key, value=fetch(cfg))
    return _cache["value"]


def reset() -> None:
    _cache.update(at=0.0, key=None, value=({}, ""))
    _models_cache.update(at=0.0, url="", value={})
