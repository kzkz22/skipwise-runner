"""Chat completions on any OpenAI-compatible server (LiteLLM, OpenRouter, llama-server, Ollama)."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable
from dataclasses import dataclass, field

import httpx

from .errors import WorkError
from .i18n import gettext as _

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.S)
STALL_S = 180  # the answer comes as a stream: this long without a byte means the connection is dead
LOCAL_STALL_S = 600  # a local server may still be loading a big model before the first token
_job = threading.local()


def watch_cancel(check: Callable[[], bool] | None) -> None:
    """The running job's "was it cancelled?" (set by the worker for its thread): a streamed answer stops
    at the next piece when the job is cancelled on the dwyt page."""
    _job.cancelled = check


def _cancelled() -> bool:
    check = getattr(_job, "cancelled", None)
    return bool(check and check())


@dataclass
class Answer:
    text: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    reasoning_tokens: int | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    models: list = field(default_factory=list)

    def add(self, answer: Answer) -> Answer:
        self.prompt_tokens += answer.prompt_tokens or 0
        self.completion_tokens += answer.completion_tokens or 0
        self.reasoning_tokens += answer.reasoning_tokens or 0
        self.seconds += answer.seconds
        if answer.model and answer.model not in self.models:
            self.models.append(answer.model)
        return answer

    def facts(self) -> dict:
        return {"model_used": ", ".join(self.models) or None, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "reasoning_tokens": self.reasoning_tokens or None,
                "duration_s": round(self.seconds, 1)}


RATE_TRIES = 4                  # "too many requests": wait and try again this many times within the job
RATE_WAIT_S = (20, 40, 60, 60)  # without a Retry-After header


class RateLimited(Exception):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__("HTTP 429")
        self.retry_after = retry_after


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        value = float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None
    return min(max(value, 1.0), 120.0)


def _wait(seconds: float) -> None:
    """Sleep, but stop at once when the job is cancelled (the next request then notices it)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _cancelled():
            raise WorkError(_("megszakítva"), "never")
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))


CHARS_PER_TOKEN = 2.5  # careful: Hungarian and timestamps take more tokens per character than English prose
PROMPT_TOKENS = 1200   # the instructions around the text


def chars_for(context_tokens: int | None, limit: int) -> int:
    """How many characters of text fit next to the instructions and the answer in a model's context.
    A quarter of the context (at least 1024 tokens) is kept for the answer, thinking included."""
    if not context_tokens:
        return limit
    answer = max(1024, context_tokens // 4)
    return max(2000, min(limit, int((context_tokens - answer - PROMPT_TOKENS) * CHARS_PER_TOKEN)))


def chat_url(base: str) -> str:
    base = base.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


class EmptyAnswer(WorkError):
    """Nothing but thinking (or nothing at all) came back."""


class JsonModeRefused(Exception):
    """The server or the model does not take response_format: asked again without it."""


NO_JSON_MODE: set = set()  # chat URLs (or URL + model) that refused response_format in this run


class Llm:
    def __init__(self, base_url: str, api_key: str, timeout: float = 2400,
                 extra: dict[str, Any] | None = None, stall_s: float = STALL_S,
                 fixed_temperature: float | None = None) -> None:
        self.url = chat_url(base_url)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout  # the whole answer at most
        self.stall_s = stall_s  # without any byte at most
        self.extra = extra or {}  # server-specific request fields, e.g. OpenRouter's "reasoning"
        # a model that only writes well with this temperature (AssemblyAI's small trial model): always sent
        self.fixed_temperature = fixed_temperature

    def chat(self, model: str, system: str, user: str, temperature: float = 0.3) -> Answer:
        return self.complete(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                             temperature)

    def complete(self, model: str, messages: list[dict], temperature: float | None = 0.3,
                 on_text: Callable[[str, bool], None] | None = None, json_mode: bool = False) -> Answer:
        """A whole conversation; temperature None leaves it to the server (the model's preset).

        The answer is asked for as a stream (every OpenAI-compatible server can): a dead connection shows up
        as silence (stall_s), and a job cancelled on the dwyt page stops at the next piece. A server that
        answers in one piece anyway is read the same way as before. "Too many requests" (HTTP 429, e.g. a free
        AssemblyAI account) is waited out a few times before the job is handed back for later.

        on_text(text so far, thinking): called as the answer comes, with the visible text (thinking left out) and
        whether the model is still thinking; used to show a question's answer as it is written (19a item 8).

        json_mode: the answer must be one JSON object (`response_format: json_object`; llama-server then only lets the
        model write valid JSON). A server or model that refuses the field is asked again without it, and this
        address is remembered for the rest of the run (decision 19.35)."""
        json_mode = json_mode and self.url not in NO_JSON_MODE and (self.url, model) not in NO_JSON_MODE
        attempt = 0
        while True:
            try:
                return self._complete(model, messages, temperature, on_text, json_mode)
            except JsonModeRefused:
                NO_JSON_MODE.add(self.url)
                json_mode = False
                continue
            except EmptyAnswer:
                if not json_mode:
                    raise
                # a thinking model whose JSON grammar left it nothing to write: once more without the grammar
                log.warning("empty answer from %s in JSON mode: asking again without it", model)
                NO_JSON_MODE.add((self.url, model))  # this model, for the rest of the run
                json_mode = False
                continue
            except RateLimited as exc:
                if attempt == RATE_TRIES:
                    raise WorkError(_("A(z) {model} modell percenkénti kerete elfogyott (HTTP 429); "
                                       "később újrapróbálom.").format(
                                      model=model), "later") from exc
                _wait(exc.retry_after or RATE_WAIT_S[min(attempt, len(RATE_WAIT_S) - 1)])
                attempt += 1

    def _complete(self, model: str, messages: list[dict], temperature: float | None,
                  on_text: Callable[[str, bool], None] | None = None, json_mode: bool = False) -> Answer:
        started = time.monotonic()
        body: dict = {"model": model, "messages": messages, "stream": True,
                      "stream_options": {"include_usage": True}, **self.extra}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.fixed_temperature is not None:
            temperature = self.fixed_temperature
        if temperature is not None:
            body["temperature"] = temperature
        try:
            with httpx.stream("POST", self.url, json=body, headers=self.headers,
                              timeout=httpx.Timeout(self.stall_s, connect=10)) as resp:
                if resp.status_code == 429:
                    resp.read()
                    raise RateLimited(_retry_after(resp))
                if resp.status_code >= 400:
                    resp.read()
                    if json_mode and resp.status_code in (400, 404, 415, 422) and re.search(
                            r"response_format|json_object|json mode|json_schema|grammar", _error(resp), re.I):
                        raise JsonModeRefused()
                    self._refused(resp, model)
                if "text/event-stream" not in resp.headers.get("content-type", ""):
                    resp.read()
                    return self._whole(resp, model, started)
                return self._streamed(resp, model, started, on_text)
        except httpx.TimeoutException as exc:
            raise WorkError(_("A(z) {model} modell {int} perce nem küldött semmit; a kapcsolat "
                               "valószínűleg megszakadt. Később újrapróbálom.").format(
                              model=model, int=int(self.stall_s // 60) or 1), "later") from exc
        except httpx.HTTPError as exc:
            raise WorkError(_("Az LLM-szerver kapcsolata megszakadt ({exc}); később újrapróbálom.").format(
                exc=exc), "later") from exc

    def _refused(self, resp: httpx.Response, model: str) -> None:
        if resp.status_code in (408, 429) or resp.status_code >= 500:
            raise WorkError(_("A(z) {model} modell most nem érhető el (HTTP {status_code}).").format(
                model=model, status_code=resp.status_code), "later")
        detail = _error(resp)
        if resp.status_code in (401, 403) or "No connected db" in detail:
            # LiteLLM answers "No connected db" when the key is not its master key (it would look
            # the key up in a database it does not have).
            raise WorkError(_("Az LLM-szerver nem fogadta el a kulcsot (HTTP {status_code}: "
                               "{detail}). Ellenőrizd: skipwise-runner config set llm_key ... "
                               "(LiteLLM-nél a LITELLM_MASTER_KEY értéke), majd indítsd újra a "
                               "futtatót.").format(
                              status_code=resp.status_code, detail=detail[:200]))
        raise WorkError(_("Az LLM-szerver elutasította a kérést (HTTP {status_code}): {detail}").format(
            status_code=resp.status_code, detail=detail))

    def _whole(self, resp: httpx.Response, model: str, started: float) -> Answer:
        try:
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise WorkError(_("Az LLM-szerver válasza nem értelmezhető."), "later") from exc
        return _answer(text, data.get("model"), data.get("usage") or {}, started)

    def _streamed(self, resp: httpx.Response, model: str, started: float,
                  on_text: Callable[[str, bool], None] | None = None) -> Answer:
        parts: list[str] = []
        usage: dict = {}
        used = None
        finished = False  # [DONE] or a finish_reason: otherwise the stream broke off in the middle
        for line in resp.iter_lines():
            if _cancelled():
                raise WorkError(_("megszakítva"), "never")
            if time.monotonic() - started > self.timeout:
                raise WorkError(_("A(z) {model} modell {int} perc alatt sem végzett; később "
                                   "újrapróbálom.").format(
                                  model=model, int=int(self.timeout // 60)), "later")
            if not line.startswith("data:"):
                continue  # SSE comments (": OPENROUTER PROCESSING"), event names, empty lines
            payload = line[5:].strip()
            if payload == "[DONE]":
                finished = True
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("error"):
                err = chunk["error"]
                text = err.get("message") if isinstance(err, dict) else err
                raise WorkError(_("A(z) {model} modell válasza félbeszakadt: {str}").format(
                    model=model, str=str(text)[:200]), "later")
            used = chunk.get("model") or used
            usage = chunk.get("usage") or usage
            reasoning = False
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
                reasoning = reasoning or bool(delta.get("reasoning_content") or delta.get("reasoning"))
                finished = finished or bool(choice.get("finish_reason"))
            if on_text is not None:
                on_text(*visible("".join(parts), reasoning))
        if not finished:
            raise WorkError(_("A(z) {model} modell válasza félbeszakadt (a kapcsolat megszakadt); "
                               "később újrapróbálom.").format(
                              model=model), "later")
        return _answer("".join(parts), used, usage, started)


def visible(text: str, reasoning: bool = False) -> tuple[str, bool]:
    """The part of an answer still being written that a reader may see, and whether the model is thinking:
    finished <think> blocks are dropped, an open one hides the rest (it is the model's thinking)."""
    text = _THINK.sub("", text)
    start = text.find("<think>")
    if start >= 0:
        return text[:start].strip(), True
    text = text.lstrip()
    return text, reasoning and not text


def _answer(text: str, model: str | None, usage: dict, started: float) -> Answer:
    text = _THINK.sub("", text).strip()
    if not text:
        raise EmptyAnswer(_("A modell üres választ adott."), "later")
    details = usage.get("completion_tokens_details") or {}
    return Answer(text=text, model=model, prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"), seconds=time.monotonic() - started,
                  reasoning_tokens=details.get("reasoning_tokens"))


def _error(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        return str(err.get("message") if isinstance(err, dict) else err or resp.text)[:400]
    except ValueError:
        return resp.text[:400]
