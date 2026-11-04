"""Summaries from the original-language transcript (ported from yt-transcribe-webui); in Hungarian with the
proven Hungarian instructions, in the 19 other content languages with one English instruction (round 9).

The whole transcript goes in one request when it fits; longer ones are first condensed part
by part into timestamped notes, and the summary is written from those.
"""
from __future__ import annotations

from typing import Any, Callable

from . import formatting, langs
from .errors import WorkError
from .llm import Llm, Usage, chars_for

PROMPT_VERSION = 1
MAX_CHARS = 120000
LANGUAGE_NAMES = {"hu": "magyar"}

SYSTEM = (
    "Videók és hangfelvételek átiratából készítesz összefoglalót. Mindig {language} nyelven írsz, "
    "akkor is, ha az átirat más nyelvű. Csak arra támaszkodj, ami az átiratban elhangzik; ne találj ki "
    "semmit, és ne egészítsd ki saját tudással. A neveket, mozaikszavakat és szakkifejezéseket eredeti "
    "alakjukban hagyd; ha nem {language} nyelvűek, az első előfordulásuknál írd mögéjük zárójelben a "
    "jelentésüket {language} nyelven."
)

FORMAT = """Pontosan ebben a szerkezetben válaszolj, Markdownban, más bevezetés vagy zárás nélkül:

## Összefoglaló
5–10 mondat, amely az egész felvételt lefedi, az elejétől a végéig.

## Fő pontok
- [időbélyeg] egy fő gondolat egy mondatban
(5–12 pont, időrendben. Az időbélyeget pontosan úgy másold át, ahogy az átirat sorának elején áll, például [12:34] vagy [01:02:03].)
{speakers_block}"""

SPEAKERS_BLOCK = """
## Beszélők
- Címke: egy mondat arról, mit képviselt vagy mivel járult hozzá a beszélgetéshez
(Minden beszélő-címkéhez egy sor, a címkét pontosan úgy írd, ahogy az átiratban áll, például „Szereplő 1”. Valódi nevet vagy szerepet csak zárójelben adj hozzá, és csak akkor, ha az átiratból egyértelmű, például mert megszólítják vagy bemutatkozik.)"""

NOTES_PROMPT = """Ez egy hosszú átirat {part}. része ({parts} részből). Írj róla tömör, időrendi jegyzetet {language} nyelven:
pontokba szedve a fontos állításokat, mindegyik elején az átirat időbélyegével ([12:34] alakban, ahogy az átiratban áll),
és ha vannak beszélő-címkék, azzal, hogy ki mondta. Bevezetés és összegzés nélkül.

Átirat:
{text}"""

DESCRIBE = {"captions": "YouTube-felirat", "cloud": "felhős átírás", "local": "helyi Whisper"}

# every other language (round 9): one English instruction, the target language a parameter
SYSTEM_EN = (
    "You summarise transcripts of videos and audio recordings. You always write in {language}, even when the "
    "transcript is in another language. Rely only on what is said in the transcript; do not invent anything and do "
    "not add your own knowledge. Keep names, acronyms and technical terms in their original form; if they are not "
    "{language}, give their meaning in {language} in brackets at their first occurrence."
)

FORMAT_EN = """Answer in exactly this structure, in Markdown, in {language}, with no introduction or closing:

## {h_summary}
5–10 sentences covering the whole recording, from beginning to end.

## {h_points}
- [timestamp] one main idea in one sentence
(5–12 points, in chronological order. Copy the timestamp exactly as it appears at the start of the transcript line, for example [12:34] or [01:02:03].)
{speakers_block}"""

SPEAKERS_BLOCK_EN = """
## {h_speakers}
- Label: one sentence on what they stood for or contributed to the conversation
(One line per speaker label; write the label exactly as it appears in the transcript, for example "{example}". Add a real name or role in brackets only if it is clear from the transcript, for example because they are addressed by name or introduce themselves.)"""

NOTES_PROMPT_EN = """This is part {part} of a long transcript ({parts} parts). Write concise, chronological notes on it in {language}:
the important statements as bullet points, each starting with the transcript's timestamp ([12:34] form, as in the transcript),
and, if there are speaker labels, who said it. No introduction or conclusion.

Transcript:
{text}"""

DESCRIBE_EN = {"captions": "YouTube captions", "cloud": "cloud transcription", "local": "local Whisper"}


def transcript_lines(segments: list[dict]) -> list[str]:
    out = []
    for turn in formatting.turns(segments, max_chars=400):
        who = f" {turn['speaker']}:" if turn["speaker"] else ""
        out.append(f"[{formatting.hms(turn['start'])}]{who} {turn['text']}")
    return out


def _header(meta: dict, t: dict) -> str:
    lines = [f"Cím: {meta.get('title') or meta.get('url') or 'ismeretlen'}"]
    if meta.get("channel"):
        lines.append(f"Csatorna: {meta['channel']}")
    if meta.get("duration_s"):
        lines.append(f"Hossz: {formatting.hms(meta['duration_s'])}")
    if meta.get("page_text"):  # the page the recording was found on (19.10): background, not part of the recording
        lines.append("Az oldal szövege, ahol a felvétel megjelent (csak háttér, nem a felvétel része):\n"
                     + str(meta["page_text"])[:3000])
    from .slides import background  # the slides of the video (plan 19c round C)
    if background(meta, True):
        lines.append(background(meta, True))
    lines.append(f"Az átirat nyelve: {t.get('language') or 'ismeretlen'}")
    lines.append(f"Az átirat forrása: {DESCRIBE.get(t.get('mode'), t.get('mode'))}")
    if t.get("mode") == "captions" and t.get("caption_kind") == "auto":
        lines.append("Megjegyzés: automatikus feliratból készült, felismerési hibák lehetnek benne; "
                     "a nyilvánvaló elhallásokat értelem szerint kezeld.")
    if t.get("diarized") and t.get("speakers"):
        lines.append("Beszélők: " + ", ".join(str(s) for s in t["speakers"]))
    return "\n".join(lines)


def _header_en(meta: dict, t: dict) -> str:
    lines = [f"Title: {meta.get('title') or meta.get('url') or 'unknown'}"]
    if meta.get("channel"):
        lines.append(f"Channel: {meta['channel']}")
    if meta.get("duration_s"):
        lines.append(f"Length: {formatting.hms(meta['duration_s'])}")
    if meta.get("page_text"):
        lines.append("Text of the page the recording appeared on (background only, not part of the recording):\n"
                     + str(meta["page_text"])[:3000])
    from .slides import background
    if background(meta, False):
        lines.append(background(meta, False))
    lines.append(f"Language of the transcript: {langs.english_name(t.get('language'))}")
    lines.append(f"Source of the transcript: {DESCRIBE_EN.get(t.get('mode'), t.get('mode'))}")
    if t.get("mode") == "captions" and t.get("caption_kind") == "auto":
        lines.append("Note: made from automatic captions, so recognition errors are possible; treat obvious "
                     "mishearings by their sense.")
    if t.get("diarized") and t.get("speakers"):
        lines.append("Speakers: " + ", ".join(str(s) for s in t["speakers"]))
    return "\n".join(lines)


def _chunks(lines: list[str], max_chars: int) -> list[str]:
    parts, cur, size = [], [], 0
    for line in lines:
        if cur and size + len(line) + 1 > max_chars:
            parts.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append("\n".join(cur))
    return parts


def run(job_input: dict[str, Any], llm: Llm, model: str, stage: Callable[[str], None] = lambda s: None,
        max_chars: int = MAX_CHARS, progress: Callable[[dict[str, Any]], None] = lambda p: None,
        context_tokens: int | None = None) -> dict[str, Any]:
    t = job_input["transcript"]
    meta = job_input.get("meta") or {}
    target = langs.code(job_input.get("language"))
    hungarian = langs.is_source(target)  # the proven Hungarian instructions; the rest one English one (round 9)
    lang = LANGUAGE_NAMES.get(target, target) if hungarian else langs.english_name(target)
    notes_prompt = NOTES_PROMPT if hungarian else NOTES_PROMPT_EN
    lines = transcript_lines(t["segments"])
    if not lines:
        raise WorkError("Az átirat üres, nincs mit összefoglalni.")
    system = (SYSTEM if hungarian else SYSTEM_EN).format(language=lang)
    max_chars = chars_for(context_tokens, max_chars)  # a local model's context decides the part size
    body = "\n".join(lines)
    chunked = len(body) > max_chars
    usage = Usage()
    stage("summarize")
    if chunked:
        parts = _chunks(lines, max_chars)
        notes = []
        for i, part in enumerate(parts, 1):  # the last step (part parts+1) writes the summary itself
            progress({"part": i, "parts": len(parts) + 1, "model": model})
            notes.append(usage.add(llm.chat(model, system, notes_prompt.format(part=i, parts=len(parts),
                                                                               language=lang, text=part))).text)
        progress({"part": len(parts) + 1, "parts": len(parts) + 1, "model": model})
        while len("\n\n".join(notes)) > max_chars and len(notes) > 1:  # the notes themselves too long: merge
            merged = _chunks(notes, max_chars)
            notes = [usage.add(llm.chat(model, system, notes_prompt.format(part=i, parts=len(merged), language=lang,
                                                                           text=part))).text
                     for i, part in enumerate(merged, 1)]
        material = ("Az átirat túl hosszú volt, ezért részenként jegyzet készült róla. A jegyzetek:\n\n"
                    if hungarian else "The transcript was too long, so notes were made of it part by part. The "
                    "notes:\n\n") + "\n\n".join(notes)
    else:
        progress({"model": model})
        material = ("Átirat:\n" if hungarian else "Transcript:\n") + body
    has_speakers = bool(t.get("diarized") and t.get("speakers"))
    if hungarian:
        speakers = SPEAKERS_BLOCK if has_speakers else ""
        user = f"{_header(meta, t)}\n\n{FORMAT.format(speakers_block=speakers)}\n\n{material}"
    else:
        h_summary, h_points, h_speakers = langs.headings(target)
        example = str((t.get("speakers") or ["Speaker 1"])[0])
        speakers = SPEAKERS_BLOCK_EN.format(h_speakers=h_speakers, example=example) if has_speakers else ""
        form = FORMAT_EN.format(language=lang, h_summary=h_summary, h_points=h_points, speakers_block=speakers)
        user = f"{_header_en(meta, t)}\n\n{form}\n\n{material}"
    answer = usage.add(llm.chat(model, system, user))
    return {"text": answer.text, "model": model, "language": target, "prompt_version": PROMPT_VERSION,
            "chunked": chunked, "input_chars": len(body), **usage.facts()}
