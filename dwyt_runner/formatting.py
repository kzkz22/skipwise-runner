"""Pure helpers: words -> sentence-sized segments, speaker relabelling, paragraphs, times.
(Ported from the old yt-transcribe API.)"""
from __future__ import annotations

from . import langs

UNKNOWN_WORD = {"hu": "Ismeretlen", "en": "Unknown"}


def speaker_word(language: str | None) -> str:
    """The label word in the transcript's own language ("Szereplő 1", "Sprecher 1"; round 9)."""
    return langs.speaker_word(language)


def unknown_word(language: str | None) -> str:
    return UNKNOWN_WORD.get((language or "").lower(), "Unknown")


def _num(value, fallback: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if f != f else f  # NaN


# Uncertain words (plan 19c round A): below these a word is underlined on the item page, for the user to check.
# whisper.cpp: the lowest token probability of the word; AssemblyAI: the word confidence; WhisperX: the alignment score
# (a different, weaker signal: how well the word fits the sound). To be tuned on the test set (19c, Mérés).
LOW_P = 0.45
LOW_ALIGN = 0.35
MAX_LOW = 60  # marked words per segment


def _low_of_words(text: str, words: list) -> list[list[int]]:
    """Character ranges of the low-scoring words, found in the segment text one after the other."""
    out: list[list[int]] = []
    pos = 0
    for w in words if isinstance(words, list) else []:
        if not isinstance(w, dict):
            continue
        word = str(w.get("word") or w.get("text") or "").strip()
        if not word:
            continue
        at = text.find(word, pos)
        if at < 0:
            continue
        pos = at + len(word)
        p = w.get("p")
        limit = LOW_P
        if p is None and w.get("score") is not None:
            p, limit = w.get("score"), LOW_ALIGN
        try:
            low = p is not None and float(p) < limit
        except (TypeError, ValueError):
            low = False
        if low and len(out) < MAX_LOW:
            out.append([at, pos])
    return out


def _valid_low(low, text: str) -> list[list[int]]:
    out = []
    for r in low if isinstance(low, list) else []:
        if (isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) for x in r)
                and 0 <= r[0] < r[1] <= len(text)):
            out.append([r[0], r[1]])
    return out[:MAX_LOW]


def clean_segments(raw_segments: list[dict]) -> list[dict]:
    """Keep only start/end/text/speaker (+ low: uncertain word ranges), drop empty text, repair missing times."""
    out: list[dict] = []
    last_end = 0.0
    for seg in raw_segments:
        original = str(seg.get("text", ""))
        text = " ".join(original.split())
        if not text:
            continue
        start = _num(seg.get("start"), last_end)
        end = max(_num(seg.get("end"), start), start)
        item = {"start": start, "end": end, "text": text, "speaker": seg.get("speaker")}
        if seg.get("low") and original == text:
            low = _valid_low(seg.get("low"), text)
        else:
            low = _low_of_words(text, seg.get("words"))
        if low:
            item["low"] = low
        out.append(item)
        last_end = end
    return out


def segments_from_words(words: list[dict], max_chars: int = 220, max_seconds: float = 15.0) -> list[dict]:
    """Group words (text, start, end, optional speaker, optional p) into sentence-sized segments: a segment
    ends at a speaker change, after . ! ? …, or when it would grow past the limits. A word with p < LOW_P is
    kept as an uncertain range (`low`: [start, end] characters in the segment text)."""
    segments: list[dict] = []
    cur: dict | None = None
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        speaker = word.get("speaker")
        start, end = float(word.get("start") or 0), float(word.get("end") or 0)
        if cur is not None and (cur["speaker"] != speaker or len(cur["text"]) + 1 + len(text) > max_chars
                                or end - cur["start"] > max_seconds):
            segments.append(cur)
            cur = None
        if cur is None:
            cur = {"start": start, "end": end, "text": text, "speaker": speaker}
            at = 0
        else:
            at = len(cur["text"]) + 1
            cur["text"] += " " + text
            cur["end"] = end
        try:
            low = word.get("p") is not None and float(word["p"]) < LOW_P
        except (TypeError, ValueError):
            low = False
        if low and len(cur.setdefault("low", [])) < MAX_LOW:
            cur["low"].append([at, at + len(text)])
        if text[-1] in ".!?…":
            segments.append(cur)
            cur = None
    if cur is not None:
        segments.append(cur)
    for seg in segments:
        if not seg.get("low"):
            seg.pop("low", None)
    return segments


REPEAT_TIMES = 4          # a phrase said this many times in a row (or more) is a Whisper loop: kept once
REPEAT_SEGMENTS = 3       # the same segment this many times in a row: kept once
REPEAT_MAX_WORDS = 8


def _collapse(words: list[str]) -> tuple[list[str], int]:
    """Phrases of 1..REPEAT_MAX_WORDS words repeated REPEAT_TIMES or more in a row, kept once."""
    removed = 0
    n = 1
    while n <= REPEAT_MAX_WORDS:
        i, out, changed = 0, [], False
        while i < len(words):
            phrase = words[i:i + n]
            key = [w.lower().strip(".,!?…;:") for w in phrase]
            count = 1
            while (len(phrase) == n and
                   [w.lower().strip(".,!?…;:") for w in words[i + count * n:i + (count + 1) * n]] == key):
                count += 1
            if len(phrase) == n and count >= REPEAT_TIMES:
                out += words[i + (count - 1) * n:i + count * n]
                removed += count - 1
                i += count * n
                changed = True
            else:
                out.append(words[i])
                i += 1
        words = out
        n = n if changed else n + 1
    return words, removed


def drop_repeats(segments: list[dict]) -> tuple[list[dict], int]:
    """Whisper's repetition loops: a phrase said many times in a row inside a segment, and the same segment many
    times in a row. Returns the segments and how many repeats were dropped."""
    removed = 0
    out: list[dict] = []
    for seg in segments:
        words, n = _collapse(seg["text"].split())
        if n:
            seg = {**seg, "text": " ".join(words)}
            seg.pop("low", None)  # the ranges no longer fit the shorter text
            removed += n
        out.append(seg)
    kept: list[dict] = []
    i = 0
    while i < len(out):
        j = i + 1
        while (j < len(out) and out[j]["text"].lower() == out[i]["text"].lower()
               and out[j].get("speaker") == out[i].get("speaker")):
            j += 1
        if j - i >= REPEAT_SEGMENTS:
            kept.append({**out[i], "end": out[j - 1]["end"]})
            removed += j - i - 1
        else:
            kept += out[i:j]
        i = j
    return kept, removed


def relabel_speakers(segments: list[dict], word: str, unknown: str) -> list[dict]:
    """Rename provider speaker ids to '<word> 1..N' by first appearance; returns the summary list."""
    mapping: dict[str, str] = {}
    prev = None
    for seg in segments:
        raw = seg.get("speaker")
        if raw is None:
            seg["speaker"] = prev if prev is not None else unknown
        else:
            if raw not in mapping:
                mapping[raw] = f"{word} {len(mapping) + 1}"
            seg["speaker"] = mapping[raw]
        prev = seg["speaker"]
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg["speaker"]] = totals.get(seg["speaker"], 0.0) + (seg["end"] - seg["start"])
    spoken = sum(totals.values()) or 1.0
    order = list(mapping.values()) + ([unknown] if unknown in totals else [])
    return [{"label": label, "share": round(totals.get(label, 0.0) / spoken, 3)} for label in order]


def hms(seconds: float | None) -> str:
    s = int(max(seconds or 0, 0))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def turns(segments: list[dict], max_chars: int = 900, max_gap: float = 20.0) -> list[dict]:
    """Consecutive segments of one speaker as paragraphs."""
    out: list[dict] = []
    for seg in segments:
        cur = out[-1] if out else None
        if (cur is not None and cur["speaker"] == seg.get("speaker")
                and len(cur["text"]) + 1 + len(seg["text"]) <= max_chars and seg["start"] - cur["end"] <= max_gap):
            cur["text"] += " " + seg["text"]
            cur["end"] = seg["end"]
        else:
            out.append({"start": seg["start"], "end": seg["end"], "speaker": seg.get("speaker"), "text": seg["text"]})
    return out
