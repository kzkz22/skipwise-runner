"""Terms for the transcription (plan 19c round A, decision 19.30): names and technical words the user gave at submit.

The central sends them in the transcribe options (`terms`, a list of short strings). Whisper gets them as its initial
prompt (whisper.cpp `--prompt` with `--carry-initial-prompt`, WhisperX `initial_prompt`), AssemblyAI as
`keyterms_prompt`. Captions are not affected.

whisper.cpp keeps at most 224 prompt tokens and fills what the initial prompt leaves with the previous text. The
previous text is what `-mc 0` kept out (it starts repetition loops), so the prompt is made long enough to fill the
whole budget: the term list is repeated in front of itself, and whisper.cpp drops the front copies. One copy is at
most PROMPT_CHARS characters, which stays under 224 tokens even for accented Hungarian names (measured with Whisper's
own tokenizer: 1.5-2.6 characters per token for Hungarian names and technical term lists).
"""
from __future__ import annotations

import unicodedata
from typing import Any

MAX_TERMS = 60
TERM_CHARS = 60
PROMPT_CHARS = 330        # one copy of the list: <= 220 tokens at 1.5 characters per token
FILL_CHARS = 700          # the padded prompt: >= 224 tokens at 3 characters per token
KEYTERM_WORDS = 6         # AssemblyAI: at most 6 words per phrase
KEYTERMS_LIMIT = {"universal-2": 200, "universal-3-5-pro": 1000}  # words in all phrases together


def clean(raw: Any) -> list[str]:
    """The terms of a job as short, printable, distinct strings (the central is not trusted blindly)."""
    out: list[str] = []
    seen: set[str] = set()
    for value in raw if isinstance(raw, list) else []:
        if not isinstance(value, str):
            continue
        text = "".join(ch for ch in unicodedata.normalize("NFC", value) if ch.isprintable())
        text = " ".join(text.replace(",", " ").split())[:TERM_CHARS].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
        if len(out) >= MAX_TERMS:
            break
    return out


def prompt_list(terms: list[str]) -> tuple[str, int]:
    """(one copy of the list as a sentence, how many terms were left out because of the length)."""
    kept: list[str] = []
    for term in terms:
        if len(", ".join(kept + [term])) + 1 > PROMPT_CHARS:
            break
        kept.append(term)
    return (", ".join(kept) + "." if kept else ""), len(terms) - len(kept)


def whisper_cpp_prompt(terms: list[str]) -> tuple[str, int]:
    """The padded prompt for whisper.cpp (see the module text) and the number of terms left out."""
    core, left_out = prompt_list(terms)
    if not core:
        return "", left_out
    padded = core
    while len(padded) < FILL_CHARS:
        padded = core + " " + padded
    return padded, left_out


def keyterms(terms: list[str], api_model: str) -> list[str]:
    """The phrases for AssemblyAI's keyterms_prompt, within the model's word limit."""
    limit = KEYTERMS_LIMIT.get(api_model, 200)
    out, words = [], 0
    for term in terms:
        parts = term.split()
        if len(parts) > KEYTERM_WORDS:
            continue
        if words + len(parts) > limit:
            break
        out.append(term)
        words += len(parts)
    return out
