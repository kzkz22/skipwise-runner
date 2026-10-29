"""Local mode: the audio goes to the user's own Whisper GPU worker, through llama-swap.

llama-swap (dwyt-airouter) gives the GPU to one program at a time. The whole job is ONE
streamed HTTP request to  WHISPER_URL/transcribe : llama-swap starts the worker container
(stopping the LLM), keeps queued LLM requests waiting until this request ends, and the worker
streams its progress as JSON lines: {"stage": ...} at least every 10 s, then {"result": ...}
or {"error": ...}. Closing the connection stops the job on the worker and frees the GPU.

No fallback to another mode: when the GPU host cannot be reached the job is retried later
(it waits for the GPU host); when the worker refuses or fails, the job fails.
(Ported from the yt-transcribe API's local mode.)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any, Callable

import httpx

from . import formatting, media, whispercpp
from .config import Config
from .errors import WorkError
from .i18n import gettext as _

FIRST_BYTE_S = 900      # llama-swap answers only after the worker container is up (wake-up included)


def deadline_s(cfg: Config, audio_seconds: float) -> int:
    """The whole job may take whisper_timeout_min, or 4x the audio when that is more (a CPU run)."""
    return int(max(cfg.whisper_timeout_min * 60, 4 * (audio_seconds or 0)))


def _query(options: dict[str, Any], model: str) -> dict[str, str]:
    q = {"model": model, "diarize": "true" if options.get("diarize", True) else "false"}
    if options.get("language"):
        q["language"] = str(options["language"])
    for key in ("num_speakers", "min_speakers", "max_speakers"):
        if options.get(key):
            q[key] = str(int(options[key]))
    from . import terms
    prompt, _left_out = terms.prompt_list(terms.clean(options.get("terms")))
    if prompt:  # WhisperX applies the initial prompt to every chunk (no previous text: condition_on_previous_text off)
        q["prompt"] = prompt
    return q


def _refusal(resp: httpx.Response) -> WorkError:
    text = resp.text[:600]
    try:
        body = resp.json()
        detail = body.get("detail") or body.get("error") or text
        if isinstance(detail, dict):
            detail = detail.get("message", text)
    except ValueError:
        detail = text
    if resp.status_code in (502, 503, 504):
        return WorkError(_("A GPU-gép most nem indítható (HTTP {status_code}: {str}); később "
                            "újrapróbálom.").format(
                           status_code=resp.status_code, str=str(detail)[:300]), "later")
    if resp.status_code == 404:
        return WorkError(_("A Whisper-cím nem jó (HTTP 404): {str}. Ellenőrizd: skipwise-runner "
                            "config set whisper_url http://AI-ROUTER:8080/upstream/whisper").format(
                           str=str(detail)[:200]))
    return WorkError(_("A Whisper-worker elutasította a munkát (HTTP {status_code}): {str}").format(
        status_code=resp.status_code, str=str(detail)[:400]))


def transcribe_on_worker(flac: str, options: dict[str, Any], model: str, cfg: Config,
                         stage: Callable[[str], None], cancelled: Callable[[], bool], limit_s: int) -> dict:
    url = cfg.whisper_url.rstrip("/") + "/transcribe"
    deadline = time.monotonic() + limit_s
    timeout = httpx.Timeout(connect=15, read=FIRST_BYTE_S, write=600, pool=15)
    try:
        with open(flac, "rb") as fh, httpx.Client(timeout=timeout) as http, http.stream(
                "POST", url, params=_query(options, model), content=fh,
                headers={"content-type": "audio/flac", "content-length": str(os.path.getsize(flac))}) as resp:
            if resp.status_code != 200:
                resp.read()
                raise _refusal(resp)
            result = None
            for raw in resp.iter_lines():  # read to the end: the worker closes the stream after the result
                if cancelled():
                    raise WorkError(_("megszakítva"))
                if time.monotonic() > deadline:
                    raise WorkError(_("A Whisper-worker {limit_s60} perc alatt sem végzett; a munka "
                                       "leállt (whisper_timeout_min).").format(
                                      limit_s60=limit_s // 60))
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if "result" in msg:
                    result = msg["result"]
                    continue
                if "error" in msg:
                    error = str(msg["error"])
                    raise WorkError(_("A Whisper-worker hibát jelzett: {error}").format(error=error[:600]))
                if msg.get("stage"):
                    stage(str(msg["stage"]))
            if result is not None:
                return result
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise WorkError(_("A Whisper-worker nem érhető el ({whisper_url}): {exc}. Be van kapcsolva "
                           "az AI-router? Később újrapróbálom.").format(
                          whisper_url=cfg.whisper_url, exc=exc), "later") from exc
    except httpx.ReadTimeout as exc:
        raise WorkError(_("A Whisper-worker {FIRST_BYTE_S60} percig nem válaszolt; később "
                           "újrapróbálom.").format(
                          FIRST_BYTE_S60=FIRST_BYTE_S // 60),
                        "later") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise WorkError(_("Megszakadt a kapcsolat a Whisper-workerrel ({exc}); később újrapróbálom "
                           "(a kész átírási lépés megmarad a workeren).").format(
                          exc=exc), "later") from exc
    raise WorkError(_("A Whisper-worker eredmény nélkül zárta le a kapcsolatot (leállították vagy újraindult); "
                    "később újrapróbálom."), "later")


def run(fetch: media.Fetch | str, options: dict[str, Any], cfg: Config, stage: Callable[[str], None],
        cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """fetch: where the media comes from (media.from_url, media.from_file); a URL string also works."""
    if isinstance(fetch, str):
        fetch = media.from_url(fetch, cfg)
    if not cfg.whisper_url and not cfg.whisper_builtin:
        raise WorkError(_("Ennek a futtatónak nincs helyi Whispere (skipwise-runner config set whisper_url ..., vagy "
                        "ezen a gépen: skipwise-runner config set whisper_builtin igen)."))
    model = str(options.get("model") or (cfg.whisper_models or ["large-v3"])[0])
    if model not in cfg.whisper_models or (not cfg.whisper_url and model not in whispercpp.MODELS):
        raise WorkError(_("Ezt a Whisper-modellt a futtató nem ismeri: {model} (whisper_models: "
                           "{join}).").format(
                          model=model, join=', '.join(cfg.whisper_models)))
    started = time.monotonic()
    stage("download")
    info = fetch.info()
    workdir = tempfile.mkdtemp(prefix="dwyt_", dir=cfg.work_dir or None)
    try:
        source = fetch.get(workdir)
        info = fetch.info()
        if cancelled():
            raise WorkError(_("megszakítva"))
        limit = deadline_s(cfg, float(info.get("duration") or 0))
        if cfg.whisper_url:  # the GPU host through llama-swap
            stage("prepare-audio")
            flac = media.to_flac(source, workdir)
            stage("waiting-for-gpu")
            raw = transcribe_on_worker(flac, options, model, cfg, stage, cancelled, limit)
        else:  # whisper.cpp + sherpa-onnx on this machine
            raw = whispercpp.transcribe(source, options, model, cfg, workdir, stage, cancelled, limit)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    language = (raw.get("language") or options.get("language") or "en").split("-")[0].lower()
    segments = formatting.clean_segments(raw.get("segments") or [])
    if not segments:
        raise WorkError(_("A hangban nem volt felismerhető beszéd."))
    segments, repeats = formatting.drop_repeats(segments)  # Whisper's repetition loops (19c round A)
    diarized = bool(raw.get("diarized")) and any(seg.get("speaker") for seg in segments)
    warnings = [str(w) for w in raw.get("warnings") or []][:20]
    if repeats:
        warnings.append(_("Az átiratból {n} ismétlődő részt kihagytam (a Whisper ismétlő hurka).").format(n=repeats))
    speakers: list[dict] = []
    if diarized:
        speakers = formatting.relabel_speakers(segments, formatting.speaker_word(language),
                                               formatting.unknown_word(language))
    else:
        for seg in segments:
            seg["speaker"] = None
    if raw.get("device") == "cpu" and not any("CPU" in w for w in warnings):
        warnings.append(_("A worker CPU-n futott (nem látott GPU-t).") if cfg.whisper_url
                        else _("Az átírás a processzoron futott."))
    return {"video": media.video_meta(info),
            "transcript": {"mode": "local", "model": model, "language": language, "diarized": diarized,
                           "speakers": speakers, "warnings": warnings, "segments": segments,
                           "elapsed_s": round(time.monotonic() - started, 1)}}
