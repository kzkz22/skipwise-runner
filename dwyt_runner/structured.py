"""Answers that must be JSON (decision 19.35): the glossary, the minutes, the chapters, the categories, the
corrections and the slides' descriptions.

1. The request asks for JSON (`response_format: json_object`, Llm.complete(json_mode=True)); llama-server then only
   lets the model write valid JSON. A server that refuses the field is asked without it.
2. The answer is read tolerantly (`parse_object`): code fences, words around it, a second object or a note after
   it, trailing commas, curly quotes as JSON quotes; an answer cut off in the middle still gives its finished
   list items.
3. An unreadable answer is asked again once; then `Unreadable` (a WorkError with retry "later": the central hands the
   job out again) carries the beginning of the answer, which also goes to the runner's log. Jobs that read the
   transcript in parts (`ask_split`) first try the part again in two halves.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from .errors import WorkError
from .i18n import gettext as _

log = logging.getLogger(__name__)

SNIPPET = 300
_FENCE = re.compile(r"```(?:json)?\s*|```", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class Unreadable(WorkError):
    def __init__(self, what: str, answer: str) -> None:
        snippet = " ".join((answer or "").split())[:SNIPPET]
        super().__init__(_("A modell válaszából nem lett használható {what} (nem a kért JSON-alakban válaszolt; "
                           "később újrapróbálom). A válasz eleje: „{snippet}”").format(what=what, snippet=snippet),
                         "later")
        self.answer = answer


def _objects(text: str):
    """Every JSON object that starts at a `{` of text, in order (raw_decode reads one and ignores what follows)."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def _salvage(text: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """An answer cut off in the middle: the list items that were finished."""
    decoder = json.JSONDecoder()
    out: dict[str, Any] = {}
    for key in keys:
        m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', text)
        if not m:
            continue
        items, pos = [], m.end()
        while True:
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text) or text[pos] == "]":
                break
            try:
                value, pos = decoder.raw_decode(text, pos)
            except ValueError:
                break
            items.append(value)
        if items:
            out[key] = items
    return out or None


def parse_object(text: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """The JSON object of a model's answer that has a list under one of keys; None when there is none."""
    text = _FENCE.sub("", _THINK.sub("", text or ""))
    variants = [text, _TRAILING_COMMA.sub(r"\1", text),
                _TRAILING_COMMA.sub(r"\1", text.replace("“", '"').replace("”", '"'))]
    for candidate in variants:
        for value in _objects(candidate):
            if any(isinstance(value.get(k), list) for k in keys):
                return value
    for candidate in variants[1:]:
        found = _salvage(candidate, keys)
        if found is not None:
            return found
    return None


def first_object(text: str) -> dict[str, Any] | None:
    """The first JSON object of an answer, whatever its keys (categories: main/side)."""
    text = _FENCE.sub("", _THINK.sub("", text or ""))
    for candidate in (text, _TRAILING_COMMA.sub(r"\1", text)):
        for value in _objects(candidate):
            return value
    return None


def ask(llm: Any, model: str, messages: list[dict[str, Any]], usage: Any, hungarian: bool, what: str,
        read: Callable[[str], Any], temperature: float = 0.1) -> Any:
    """Ask for JSON; read(answer text) gives the value or None. Two tries, then Unreadable."""
    answer_text = ""
    for attempt in range(2):
        answer = llm.complete(model, messages, temperature=temperature, json_mode=True)
        if usage is not None:
            usage.add(answer)
        answer_text = answer.text
        value = read(answer_text)
        if value is not None:
            return value
        log.warning("unreadable %s from %s (try %s): %s", what, model, attempt + 1, answer_text[:2000])
        messages = messages + [{"role": "assistant", "content": answer_text[:2000]},
                               {"role": "user", "content": "Csak a JSON-objektumot írd, a megadott alakban."
                                if hungarian else "Write only the JSON object, in the given form."}]
    raise Unreadable(what, answer_text)


def ask_split(window: list[Any], run: Callable[[list[Any]], Any], min_lines: int = 20) -> list[Any]:
    """run(window) for one part of a transcript; an unreadable answer for a long part is tried again in two halves
    (a long list is where a model most often loses the form). Returns the values in order."""
    try:
        return [run(window)]
    except Unreadable:
        if len(window) < min_lines:
            raise
        half = len(window) // 2
        log.warning("unreadable answer for a part of %s lines: trying it in two halves", len(window))
        return [run(window[:half]), run(window[half:])]
