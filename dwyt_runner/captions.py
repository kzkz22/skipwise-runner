"""Captions mode: the video's own subtitles from YouTube, through yt-dlp. No GPU, no cost.

Exactly ONE track is requested, in the video's language: the uploader's own subtitles,
else YouTube's automatic captions of the original audio (`<lang>-orig`; a plain `<lang>` key
only when there is no `-orig` key). Machine translations are never used. The video's language is
the asked one, else the language of the audio track YouTube marks original (default), else yt-dlp's
`language`: that one follows the formats yt-dlp picked, and with auto-dubbed tracks it was the
language of a dub (an English video came back with Arabic captions). The track comes as
json3 and becomes sentence-sized segments without speakers. (Ported from yt-transcribe.)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from . import formatting, media, tools
from .config import Config
from .errors import WorkError, from_ytdlp
from .i18n import gettext as _

_ANNOTATION = re.compile(r"^\[[^\]]*\]$")  # [Music], [Zene], [Applause]


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def _script(code: str) -> str | None:
    parts = code.split("-")
    return parts[1].title() if len(parts) > 1 and len(parts[1]) == 4 and parts[1].isalpha() else None


ORIGINAL_PREF = 10  # yt-dlp's language_preference of the audio track YouTube marks original
DEFAULT_PREF = 5    # ... and of the default one


def audio_language(info: dict) -> str | None:
    """The language of the video's original (else default) audio track; None when no track says.

    An auto-dubbed video has one audio track per dub; yt-dlp's top-level `language` is that of the
    formats it picked, which may be a dub."""
    best: tuple[int, str] | None = None
    for f in info.get("formats") or []:
        lang = f.get("language")
        if not isinstance(lang, str) or not lang or f.get("acodec") in (None, "none"):
            continue
        pref = f.get("language_preference")
        note = (f.get("format_note") or "").lower()
        if "original" in note:
            pref = max(pref if isinstance(pref, int) else 0, ORIGINAL_PREF)
        if isinstance(pref, int) and pref >= DEFAULT_PREF and (best is None or pref > best[0]):
            best = (pref, lang)
    return best[1] if best else None


def choose_track(info: dict, language: str | None = None) -> tuple[str, str, list] | None:
    """(kind, code, formats) of the track to use, or None. kind: manual | auto."""
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    auto = info.get("automatic_captions") or {}
    lang = (language or audio_language(info) or info.get("language") or "").strip()
    if not lang:
        origs = [k for k in auto if k.endswith("-orig")]
        if len(origs) == 1:
            lang = origs[0][: -len("-orig")]
        elif len(manual) == 1:
            lang = next(iter(manual))
        else:
            return None
    want = _base(lang)
    for code in manual:
        if code.lower() == lang.lower():
            return "manual", code, manual[code]
    for code in manual:
        if _base(code) == want and _script(code) == _script(lang):
            return "manual", code, manual[code]
    origs = [code for code in auto if code.endswith("-orig")]
    for code in origs:
        if code[: -len("-orig")].lower() in (lang.lower(), want):
            return "auto", code, auto[code]
    if not origs:
        for code in auto:
            if code.lower() in (lang.lower(), want):
                return "auto", code, auto[code]
    return None


def words_from_json3(data: dict, kind: str) -> list[dict]:
    units: list[dict] = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs or event.get("aAppend"):
            continue
        start = int(event.get("tStartMs") or 0)
        end = start + int(event.get("dDurationMs") or 0)
        if kind == "auto":
            for seg in segs:
                text = (seg.get("utf8") or "").strip()
                if text:
                    units.append({"text": text, "start": (start + int(seg.get("tOffsetMs") or 0)) / 1000,
                                  "event_end": end / 1000})
        else:
            text = " ".join("".join(seg.get("utf8") or "" for seg in segs).split())
            if text and not _ANNOTATION.match(text):
                units.append({"text": text, "start": start / 1000, "end": end / 1000})
    for i, unit in enumerate(units):
        if "end" not in unit:
            nxt = units[i + 1]["start"] if i + 1 < len(units) else unit["event_end"]
            unit["end"] = max(unit["start"], min(nxt, unit["start"] + 2.0, unit["event_end"]))
        unit.pop("event_end", None)
    return [u for u in units if not _ANNOTATION.match(u["text"])]


def run(url: str, options: dict[str, Any], cfg: Config, stage: Callable[[str], None]) -> dict[str, Any]:
    import yt_dlp

    stage("captions")
    try:
        with yt_dlp.YoutubeDL(tools.ytdlp_options(cfg.cookies_file)) as ydl:
            info = ydl.extract_info(url, download=False)
            media.check(info, cfg)
            track = choose_track(info, options.get("language"))
            if track is None:
                lang = options.get("language") or audio_language(info) or info.get("language") or "ismeretlen"
                raise WorkError(_("A videónak nincs felirata a saját nyelvén ({lang}). Válassz "
                                   "helyi vagy felhős átírást.").format(
                                  lang=lang), "next_step")
            kind, code, formats = track
            fmt = next((f for f in formats if f.get("ext") == "json3" and f.get("url")), None)
            if fmt is None:
                raise WorkError(_("A(z) {code} felirat nem érhető el json3 formában.").format(code=code), "next_step")
            raw = ydl.urlopen(fmt["url"]).read()
    except WorkError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise from_ytdlp(exc, _("letölteni a feliratot"), url) from exc
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except ValueError as exc:
        raise WorkError(_("A feliratfájl nem olvasható.")) from exc
    segments = formatting.clean_segments(formatting.segments_from_words(words_from_json3(data, kind)))
    if not segments:
        raise WorkError(_("A felirat üres."), "next_step")
    for seg in segments:
        seg["speaker"] = None
    warnings = []
    if kind == "auto":
        warnings.append(_("YouTube automatikus feliratából készült: felismerési hibák lehetnek benne, és nincsenek "
                        "beszélők. Jobb átirathoz válassz helyi vagy felhős átírást."))
    return {"video": media.video_meta(info),
            "transcript": {"mode": "captions", "model": f"youtube-{kind}", "caption_kind": kind,
                           "language": _base(code), "diarized": False, "speakers": [], "warnings": warnings,
                           "segments": segments}}
