"""One job: check it, run it, return the result shape of the protocol."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Any, Callable
from urllib.parse import urlsplit

from . import (ask, captions, langs, categorize, chapters, cloud, gateway, indavideo, knowledge, local, localllm, media,
               minutes, modelinfo, originals, refine, resolve, summarize, tracks, translate, watch)
from .config import Config
from .errors import WorkError
from .llm import Llm
from .protocol import Central, CentralError
from .i18n import gettext as _

LLM_KINDS = ("summarize", "translate", "ask", "categorize", "chapters", "minutes", "knowledge", "refine")
KINDS = ("transcribe", *LLM_KINDS, "drop_original")


def capabilities(cfg: Config) -> dict[str, Any]:
    cfg = localllm.effective(cfg)
    local_whisper = bool(cfg.whisper_url or cfg.whisper_builtin)
    modes = ["captions"] + (["cloud"] if cfg.assemblyai_key else []) + (["local"] if local_whisper else [])
    roles = [r for r, model in (("public", cfg.llm_public_model), ("private", cfg.llm_private_model))
             if cfg.llm_url and model]
    kinds = ["transcribe"] + (list(LLM_KINDS) if roles else [])
    keeps = bool(cfg.keep_originals or originals.load())
    if keeps:
        kinds.append("drop_original")  # the user deletes a kept original from the dwyt pages
    info, server = modelinfo.cached(cfg) if roles else ({}, "")
    defaults = {"summarize": model_for(cfg, "summarize", "public"), "translate": model_for(cfg, "translate", "public"),
                "ask": model_for(cfg, "ask", "public"), "private": cfg.llm_private_model}
    return {"kinds": kinds, "transcribe_modes": modes, "llm_roles": roles,
            "llm_models": {role: offered_models(cfg, role) for role in roles},
            "llm_model_info": info, "llm_server": server,
            "llm_defaults": {k: v for k, v in defaults.items() if v} if roles else {},
            "allowed_domains": cfg.allowed_domains, "local_sources": False,
            "whisper_models": whisper_models(cfg),
            "source_kinds": ["upload"] + (["local"] if cfg.watch_dirs or keeps else []),
            "keeps_originals": bool(cfg.keep_originals),
            "embedded_subtitles": True,  # captions mode on an own file: its embedded subtitles (19.12)
            # round 9: summaries, translations, chapters and answers in these languages (older runners: Hungarian)
            "content_languages": sorted(langs.TABLE) if roles else []}


def whisper_models(cfg: Config) -> list[str]:
    if cfg.whisper_url:
        return list(cfg.whisper_models)
    if cfg.whisper_builtin:
        from . import whispercpp
        return [m for m in cfg.whisper_models if m in whispercpp.MODELS]
    return []


def model_for(cfg: Config, kind: str, role: str | None) -> str:
    """The model of a role; own recordings (private) always use the private model."""
    if role == "private":
        return cfg.llm_private_model
    if role == "public":
        own = {"translate": cfg.llm_translate_model, "ask": cfg.llm_ask_model}.get(kind, "")
        return own or cfg.llm_public_model
    return ""


def offered_models(cfg: Config, role: str) -> list[str]:
    """The models a user may choose for a role: the configured ones first, no duplicates.
    Own recordings (private) only ever get llm_private_model and llm_private_models."""
    if role == "private":
        names = [cfg.llm_private_model, *cfg.llm_private_models]
    else:
        names = [cfg.llm_public_model, cfg.llm_translate_model, cfg.llm_ask_model, *cfg.llm_models]
    out: list[str] = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return out[:30]


def _allowed_media(media_url: str, page_url: str, allowed: list[str]) -> bool:
    """A recording found on a page: from an allowed site, or from the page's own site (audio-assets.telex.hu)."""
    if domain_allowed(media_url, allowed):
        return True
    page_host = (urlsplit(page_url).hostname or "").lower().removeprefix("www.")
    host = (urlsplit(media_url).hostname or "").lower()
    return bool(page_host) and (host == page_host or host.endswith("." + page_host))


def domain_allowed(url: str, allowed: list[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(d == "*" or host == d.lower() or host.endswith("." + d.lower()) for d in allowed)


def _own_file(job: dict[str, Any], data: dict[str, Any], cfg: Config, stage: Callable[[str], None],
              cancelled: Callable[[], bool], central: Central | None) -> dict[str, Any]:
    """An own recording: uploaded to the central service, or a file of this runner's watched folder."""
    source, options = data["source"], data.get("options") or {}
    reports: list[list[dict[str, Any]]] = []
    sha, name = str(source.get("sha256") or ""), str(source.get("name") or _("felvétel"))
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise WorkError(_("Hibás fájlazonosító a munkában."))
    if source["kind"] == "upload":
        media_path = str(source.get("media") or "")
        if central is None or not re.fullmatch(r"/runner/v1/jobs/\d+/media", media_path):
            raise WorkError(_("Hibás letöltési cím a munkában."))

        def obtain(workdir: str) -> str:
            kept = originals.path_for(sha)  # kept before: no download (the central may not have it any more)
            if kept is not None:
                return kept
            dest = os.path.join(workdir, "upload.bin")
            stage("download")
            try:
                central.download(media_path, dest, cfg.max_file_mb * 1024 ** 2)
            except CentralError as exc:
                raise WorkError(str(exc), "later") from exc
            if cfg.keep_originals and media.file_sha256(dest) == sha:  # plan 19c: the "Eredetik" folder
                dest, removed = originals.keep(cfg, sha, name, dest)
                reports.append(removed)
            return dest
    else:
        def obtain(workdir: str) -> str:
            path = watch.path_for(sha) or originals.path_for(sha)
            if path is None:
                raise WorkError(_("A fájl már nincs meg ezen a gépen (a figyelt mappából vagy az Eredetik közül "
                                  "áthelyezték vagy törölték)."))
            return path
    fetch = media.from_file(name, sha, obtain, options)
    mode = options.get("mode")
    if mode == "local":
        result = local.run(fetch, options, cfg, stage, cancelled)
    elif mode == "cloud":
        result = cloud.run(fetch, options, cfg, stage, cancelled)
    elif mode == "captions":  # the subtitles embedded in the file (decision 19.12)
        result = _embedded_subtitles(fetch, options, cfg, stage)
    else:
        raise WorkError(_("Saját felvételhez ez a mód nem kérhető: {mode}.").format(mode=mode))
    report = fetch.report()
    if report is not None:
        result["tracks"] = report
    if reports and central is not None:  # a newly kept original: the central shows it on the item page
        originals.send(cfg, central, reports[-1])
    return result


def _embedded_subtitles(fetch: media.from_file, options: dict[str, Any], cfg: Config,
                        stage: Callable[[str], None]) -> dict[str, Any]:
    stage("download")
    workdir = tempfile.mkdtemp(prefix="dwyt_", dir=cfg.work_dir or None)
    try:
        path = fetch.verified(workdir)
        fetch.audio_used = tracks.choose_audio(fetch.tracks, options)
        try:
            sub = tracks.choose_subtitle(fetch.tracks, options, fetch.audio_used)
        except WorkError as exc:  # say what the file has, so the user can choose
            raise WorkError(f"{exc} {tracks.describe(fetch.tracks)}", exc.retry) from exc
        fetch.subtitle_used = sub.index
        stage("captions")
        transcript = tracks.transcript(path, fetch.tracks, sub, fetch.audio_used)
        info = fetch.info()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"video": media.video_meta(info), "transcript": transcript}


def run(job: dict[str, Any], cfg: Config, stage: Callable[[str], None],
        cancelled: Callable[[], bool], central: Central | None = None,
        progress: Callable[[dict[str, Any]], None] = lambda p: None,
        partial: Callable[[str, bool], None] | None = None) -> dict[str, Any]:
    """Validate the job strictly (the central service is not trusted blindly) and run it."""
    kind, data = job.get("kind"), job.get("input") or {}
    if kind == "transcribe" and (data.get("source") or {}).get("kind") in ("upload", "local"):
        return _own_file(job, data, cfg, stage, cancelled, central)
    if kind == "drop_original":  # the user deleted a kept original (plan 19c)
        sha = str(data.get("sha256") or "")
        if not originals.SHA.match(sha):
            raise WorkError(_("Hibás fájlazonosító a munkában."))
        dropped = originals.drop(sha)
        if central is not None:
            originals.send(cfg, central)
        return {"dropped": dropped}
    if kind == "transcribe":
        source, options = data.get("source") or {}, data.get("options") or {}
        url = str(source.get("url") or "")
        if source.get("kind") not in ("youtube", "url") or not url.startswith(("https://", "http://")):
            raise WorkError(_("Ezt a forrást ez a futtató nem dolgozza fel."))
        if not domain_allowed(url, cfg.allowed_domains):
            raise WorkError(_("A(z) {urlsplit} oldal nincs az engedélyezett tartományok között "
                               "(allowed_domains).").format(
                              urlsplit=urlsplit(url).hostname))
        mode = options.get("mode")
        if mode not in ("captions", "cloud", "local"):
            raise WorkError(_("Ez a futtató nem tud ilyen módot: {mode}.").format(mode=mode))
        page = None
        if source.get("kind") == "url" and not resolve.is_file(url) and not resolve.known(url):
            # a page yt-dlp does not know (a news site's video or podcast page): find the recording on it (19.10)
            stage("page")
            page = resolve.read(url)
            chosen = resolve.choose(page, mode, str(options.get("pick") or "") or None)
            if chosen.kind != "youtube" and not _allowed_media(chosen.url, url, cfg.allowed_domains):
                raise WorkError(_("A felvétel a(z) {urlsplit} oldalon van, ami nincs az "
                                   "engedélyezett tartományok között (allowed_domains).").format(
                                  urlsplit=urlsplit(chosen.url).hostname))
            url = chosen.url
        elif mode == "captions" and source.get("kind") != "youtube":
            raise WorkError(_("YouTube-feliratot csak YouTube-videóhoz lehet kérni."))
        try:
            if mode != "captions":
                # Indavideo: yt-dlp's tokens are refused; ask the way the player does (indavideo.py)
                url = indavideo.direct(url) or url
            if mode == "captions":
                result = captions.run(url, options, cfg, stage)
            elif mode == "cloud":
                result = cloud.run(url, options, cfg, stage, cancelled)
            else:
                result = local.run(url, options, cfg, stage, cancelled)
        except WorkError as exc:
            if page is None:
                raise
            # say what was found on the page, so a failure shows that the resolver did its part
            raise WorkError(_("Az oldalon ezt találtam: {label}, {url}. {exc}").format(
                label=chosen.label, url=chosen.url, exc=exc), exc.retry) from exc
        if page is not None:
            result["page"] = page.summary(chosen)
            video = result.setdefault("video", {})
            video["title"] = video.get("title") or page.title
            video["channel"] = video.get("channel") or page.site
        return result
    if kind in LLM_KINDS:
        cfg = localllm.effective(cfg)
        role = data.get("role")
        model = model_for(cfg, kind, role)
        if not cfg.llm_url or not model:
            raise WorkError(_("Ennek a futtatónak nincs modellje ehhez ({role}).").format(role=role))
        wanted = data.get("model")
        if wanted:  # chosen on the item page: only a model this runner offers for this role
            if role not in ("public", "private") or wanted not in offered_models(cfg, role):
                raise WorkError(_("Ezt a modellt a futtató nem ajánlja fel ehhez: {str} "
                                   "(llm_models, llm_private_models).").format(
                                  str=str(wanted)[:80]))
            model = str(wanted)
        context = localllm.before_llm_job(cfg, model)  # a local model: its context decides the text sizes
        context = context or gateway.context(cfg.llm_url, model)  # AssemblyAI's trial model: 32K
        if localllm.uses_local(cfg):
            localllm.mark_used(model)
        extra = {} if localllm.uses_local(cfg) else modelinfo.request_extra(cfg, model)
        from .llm import LOCAL_STALL_S, STALL_S
        llm = Llm(cfg.llm_url, cfg.llm_key, extra=extra,
                  stall_s=LOCAL_STALL_S if localllm.uses_local(cfg) else STALL_S,
                  fixed_temperature=gateway.TRIAL_TEMPERATURE if gateway.is_trial(cfg.llm_url, model) else None)
        if kind == "summarize":
            result = summarize.run(data, llm, model, stage, progress=progress, context_tokens=context)
            categorize.after_summary(data, result, llm, model)  # decision 19.13: the user's categories
            chapters.after_summary(data, result, llm, model, context)  # decision 19.21: when the central asks
            return result
        if kind == "chapters":
            return chapters.run(data, llm, model, stage, progress=progress, context_tokens=context)
        if kind == "minutes":  # plan 19c round A
            return minutes.run(data, llm, model, stage, progress=progress, context_tokens=context)
        if kind == "knowledge":  # plan 19c round B
            return knowledge.run(data, llm, model, stage, progress=progress, context_tokens=context)
        if kind == "refine":
            return refine.run(data, llm, model, stage, progress=progress, context_tokens=context)
        if kind == "categorize":
            return categorize.run(data, llm, model, stage, progress=progress)
        if kind == "ask":
            return ask.run(data, llm, model, stage, progress=progress, context_tokens=context, partial=partial)
        return translate.run(data, llm, model, stage, cancelled, progress=progress)
    raise WorkError(_("Ismeretlen munkafajta: {kind}.").format(kind=kind))
