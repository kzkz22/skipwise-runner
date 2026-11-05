"""Where each step of a job runs (decision 19.34): the step list the central's item page, the Jobs page and this
window show — "Diák értelmezése · GPU-gép · RTX 3090 · Qwen3.6-35B".

At the start of a job `plan()` lists the steps the job will most likely go through, each with its place:
  where: this (the runner's machine) | gpu (the GPU host behind llama-swap) | cloud | llm (another LLM server)
  res:   cpu | gpu | net | disk | wait (what it loads there)
and a label for the place where known: the card (`gpu`), the provider (`via`), the model. The worker's stage
calls move along the list (`Tracker.enter`); a stage the plan did not foresee is added with its usual place, steps
that never came are dropped at the end. What only shows up while running (the GPU host's card, CPU instead of GPU,
the time of each step) is noted on the step (`note`). The list goes to the central with every heartbeat inside
`progress.steps`, and with the transcription's result as `steps` (the "Hogyan készült" row).
"""
from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .config import Config

MAX_STEPS = 16

# the usual place of every stage (what is not listed here runs on this machine)
TRANSFER = {"download", "upload", "upload-frames", "captions", "page", "write", "download-model"}
CPU = {"prepare-audio", "frames", "ocr", "load-audio"}
LLM_STAGES = {"summarize", "translate", "ask", "categorize", "chapters", "minutes", "knowledge", "refine", "vision"}
GPU_HOST = {"waiting-for-gpu", "starting", "transcribe", "align", "diarize", "load-audio"}

_local = threading.local()


def _step(sid: str, where: str, res: str, **extra: Any) -> dict[str, Any]:
    out = {"id": sid, "where": where, "res": res, "state": "next"}
    out.update({k: v for k, v in extra.items() if v})
    return out


def this_gpu() -> str | None:
    """This machine's card for the label (the last machine scan)."""
    try:
        from . import machine
        m = machine.load_snapshot()
        g = m.best_gpu if m else None
        return g.name[:60] if g and g.name else None
    except Exception:  # noqa: BLE001 - only a label
        return None


def llm_place(cfg: Config, model: str) -> dict[str, Any]:
    """Where a language model runs: from the model info (dwyt-airouter: local = the GPU host, cloud), a local
    Ollama / LM Studio (this machine), a known cloud address, else another server."""
    from . import gateway, localllm, modelinfo
    if localllm.uses_local(cfg):
        return {"where": "this", "res": "gpu", "gpu": this_gpu(), "model": model}
    url = (cfg.llm_url or "").lower()
    host = (urlsplit(url).hostname or "")
    if gateway.is_gateway(url):
        return {"where": "cloud", "res": "wait", "via": "AssemblyAI", "model": model}
    if host.endswith("openrouter.ai"):
        return {"where": "cloud", "res": "wait", "via": "OpenRouter", "model": model}
    try:
        info = (modelinfo.cached(cfg)[0] if cfg.llm_url else {}).get(model) or {}
    except Exception:  # noqa: BLE001 - only a label
        info = {}
    fallback = "cloud" if any(f.get("where") == "cloud" for f in info.get("fallback") or []) else None
    if info.get("where") == "local":
        return {"where": "gpu", "res": "gpu", "gpu": info.get("gpu"), "model": info.get("target_label") or model,
                "fallback": fallback}
    if info.get("where") == "cloud":
        return {"where": "cloud", "res": "wait", "via": info.get("provider"),
                "model": info.get("target_label") or model}
    if host in ("127.0.0.1", "localhost", "::1"):
        return {"where": "this", "res": "gpu", "gpu": this_gpu(), "model": model}
    return {"where": "llm", "res": "wait", "via": host[:60] or None, "model": model}


def _transcription(mode: str | None, cfg: Config, own_captions: bool = False) -> list[dict[str, Any]]:
    if mode == "captions":
        if own_captions:  # the subtitles inside an own file
            return [_step("captions", "this", "cpu")]
        return [_step("captions", "this", "net")]
    if mode == "cloud":
        return [_step("prepare-audio", "this", "cpu"), _step("upload", "this", "net"),
                _step("transcribe", "cloud", "wait", via="AssemblyAI")]
    if cfg.whisper_url:  # the GPU host behind llama-swap
        return [_step("prepare-audio", "this", "cpu"), _step("waiting-for-gpu", "gpu", "wait"),
                _step("transcribe", "gpu", "gpu"), _step("align", "gpu", "gpu"), _step("diarize", "gpu", "gpu")]
    card = this_gpu()
    return [_step("prepare-audio", "this", "cpu"), _step("waiting-for-gpu", "this", "wait", gpu=card),
            _step("transcribe", "this", "gpu" if card else "cpu", gpu=card), _step("diarize", "this", "cpu")]


def plan(job: dict[str, Any], cfg: Config, model: str | None = None) -> list[dict[str, Any]]:
    """The steps a job will most likely take (the first one is not started yet)."""
    kind, data = job.get("kind"), job.get("input") or {}
    if kind == "transcribe":
        source, options = data.get("source") or {}, data.get("options") or {}
        own = source.get("kind") in ("upload", "local")
        out: list[dict[str, Any]] = []
        if not own and source.get("kind") == "url":
            out.append(_step("page", "this", "net"))
        if own and source.get("kind") == "local":
            out.append(_step("download", "this", "disk"))
        else:
            out.append(_step("download", "this", "net"))
        if options.get("slides"):
            from . import slides
            out += [_step("frames", "this", "cpu"), _step("ocr", "this", "cpu")]
            vision = slides.vision_model(_effective(cfg), "private" if own else "public") \
                if cfg.slides_vision else ""
            if vision:
                out.append(_step("vision", **llm_place(_effective(cfg), vision)))
            out.append(_step("upload-frames", "this", "net"))
        out += _transcription(options.get("mode"), cfg, own_captions=own)
        out.append(_step("write", "this", "net"))
        return out
    if kind == "drop_original":
        return [_step("write", "this", "disk")]
    if kind in LLM_STAGES and model:
        stage = kind
        place = llm_place(_effective(cfg), model)
        out = [_step(stage, **place)]
        if kind == "summarize":
            if data.get("categories"):
                out.append(_step("categorize", **place))
            if data.get("chapters"):
                out.append(_step("chapters", **place))
        out.append(_step("write", "this", "net"))
        return out
    return []


def _effective(cfg: Config) -> Config:
    from . import localllm
    return localllm.effective(cfg)


def usual_place(stage: str, cfg: Config | None = None) -> dict[str, Any]:
    if stage in TRANSFER:
        return {"where": "this", "res": "net"}
    if stage in CPU:
        return {"where": "this", "res": "cpu"}
    if stage in GPU_HOST and cfg is not None and cfg.whisper_url:
        return {"where": "gpu", "res": "gpu"}
    return {"where": "this", "res": "cpu"}


class Tracker:
    def __init__(self, steps: list[dict[str, Any]], cfg: Config | None = None) -> None:
        self.steps = [dict(s) for s in steps[:MAX_STEPS]]
        self.cfg = cfg
        self.lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self.started = 0.0
        self.duration: float | None = None  # the recording's length (19.36)

    def enter(self, stage: str) -> None:
        with self.lock:
            if self.current is not None and self.current["id"] == stage:
                return
            found = next((s for s in self.steps if s["id"] == stage and s["state"] == "next"), None)
            if found is None and any(s["id"] == stage for s in self.steps):
                return  # done before (the audio fetched while the slides were looked at): the step stays as it is
            self._close()
            if found is None:
                if stage == "starting":  # the worker's first moment: no step of its own
                    return
                place = dict(self.current or {}) if stage in GPU_HOST and self.current and \
                    self.current.get("where") == "gpu" else usual_place(stage, self.cfg)
                found = _step(stage, place.get("where", "this"), place.get("res", "cpu"),
                              gpu=place.get("gpu"))
                at = next((i for i, s in enumerate(self.steps) if s["state"] == "next"), len(self.steps))
                if len(self.steps) < MAX_STEPS:
                    self.steps.insert(at, found)
            else:
                # steps planned before this one that never came are dropped (an earlier stage was skipped)
                at = self.steps.index(found)
                self.steps = [s for i, s in enumerate(self.steps) if i >= at or s["state"] != "next"]
            found["state"] = "now"
            self.current, self.started = found, time.monotonic()

    def note(self, step: str | None = None, **info: Any) -> None:
        """Facts known only while running: res, gpu, model, where, via (None values are ignored)."""
        with self.lock:
            target = self.current if step is None else next(
                (s for s in reversed(self.steps) if s["id"] == step and s["state"] != "next"), None)
            if target is None:
                return
            for key, value in info.items():
                if value is None:
                    continue
                if key == "gpu" and not value:
                    target.pop("gpu", None)
                    continue
                target[key] = str(value)[:60] if isinstance(value, str) else value

    def _close(self) -> None:
        if self.current is not None:
            self.current["state"] = "done"
            self.current["s"] = round(time.monotonic() - self.started, 1)
            self.current = None

    def finish(self) -> list[dict[str, Any]]:
        """The steps that ran (for the result): the current one closed, the ones that never came dropped."""
        with self.lock:
            self._close()
            self.steps = [s for s in self.steps if s["state"] == "done"]
            return [dict(s) for s in self.steps]

    def ran(self) -> list[dict[str, Any]]:
        """The steps done so far, without the delivery itself (the result's `steps`)."""
        with self.lock:
            out = []
            for s in self.steps:
                if s["state"] == "next" or s["id"] == "write":
                    continue
                row = dict(s, state="done")
                if s is self.current:
                    row["s"] = round(time.monotonic() - self.started, 1)
                out.append(row)
            return out

    def payload(self) -> list[dict[str, Any]]:
        with self.lock:
            out = []
            for s in self.steps:
                row = dict(s)
                if s is self.current:
                    row["s"] = round(time.monotonic() - self.started, 1)
                out.append(row)
            return out


def track(tracker: Tracker | None) -> None:
    """The tracker of the job this thread runs (note() from deep inside a module finds it)."""
    _local.tracker = tracker


def note(step: str | None = None, **info: Any) -> None:
    tracker = getattr(_local, "tracker", None)
    if tracker is not None:
        tracker.note(step, **info)


def begin(job: dict[str, Any], cfg: Config, model: str | None = None) -> None:
    """Called by work.run once the job's settings (and its model) are known: the step plan."""
    tracker = getattr(_local, "tracker", None)
    if tracker is None:
        return
    planned = plan(job, cfg, model)
    with tracker.lock:
        done = [s for s in tracker.steps if s["state"] != "next"]
        seen = {s["id"] for s in done}
        tracker.steps = (done + [s for s in planned if s["id"] not in seen])[:MAX_STEPS]


# ---- labels for this window (the central has its own) ------------------------------------------------------------

def _labels():
    from .i18n import N_, labels
    places = labels({"this": N_("Ez a gép"), "gpu": N_("GPU-gép"), "cloud": N_("Felhő"),
                     "llm": N_("Nyelvimodell-szerver"), "central": N_("Központ")})
    res = labels({"cpu": N_("CPU"), "gpu": N_("GPU"), "net": N_("hálózat"), "disk": N_("lemez"),
                  "wait": N_("helyben nincs terhelés")})
    return places, res


def place_label(step: dict[str, Any], machine_name: str = "") -> str:
    """"GPU-gép · RTX 3090", "Ez a gép · CPU", "Felhő · AssemblyAI" — what the window shows next to a step."""
    from .i18n import gettext as _
    places, res = _labels()
    where = step.get("where") or "this"
    head = places.get(where, where)
    if where == "this" and machine_name:
        head = f"{head} ({machine_name})"
    parts = [head]
    if step.get("via"):
        parts.append(str(step["via"]))
    if step.get("res") == "gpu" and step.get("gpu"):
        parts.append(str(step["gpu"]))
    elif step.get("res") and step.get("res") != "wait":
        parts.append(res.get(step["res"], step["res"]))
    text = " · ".join(parts)
    if step.get("fallback") == "cloud":
        text += " " + _("(ha nem elérhető: felhő)")
    return text


def seconds_text(s: float | None) -> str:
    """"45 mp", "2 p 10 mp", "1 ó 3 p"."""
    from .i18n import gettext as _
    s = int(round(s or 0))
    if s < 60:
        return _("{s} mp").format(s=s)
    if s < 3600:
        return _("{m} p {s} mp").format(m=s // 60, s=s % 60)
    return _("{h} ó {m} p").format(h=s // 3600, m=s % 3600 // 60)


def length(seconds: Any) -> None:
    """The recording's length, once known (sent with the heartbeats: the central's expected end, 19.36)."""
    tracker = getattr(_local, "tracker", None)
    if tracker is not None and isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds > 0:
        tracker.duration = float(seconds)
