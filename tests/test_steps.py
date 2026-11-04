"""Where each step of a job runs (decision 19.34): the plan, the tracker, the notes from the GPU host and the steps
sent with the heartbeats and the transcription's result."""
import yt_dlp

from dwyt_runner import modelinfo, steps
from test_runner import VIDEO, _run_once, _video


def ids(rows):
    return [r["id"] for r in rows]


def test_plans_name_each_place(cfg, monkeypatch):
    monkeypatch.setattr(steps, "this_gpu", lambda: "RTX 4070")
    cloud = steps.plan({"kind": "transcribe", "input": {"source": {"kind": "youtube", "url": VIDEO},
                                                        "options": {"mode": "cloud"}}}, cfg)
    assert ids(cloud) == ["download", "prepare-audio", "upload", "transcribe", "write"]
    assert cloud[3] == {"id": "transcribe", "where": "cloud", "res": "wait", "via": "AssemblyAI", "state": "next"}
    cfg.whisper_url = "http://127.0.0.1:8080/upstream/whisper"
    gpu = steps.plan({"kind": "transcribe", "input": {"source": {"kind": "url", "url": "https://x.hu/a"},
                                                      "options": {"mode": "local", "slides": True}}}, cfg)
    assert ids(gpu)[:5] == ["page", "download", "frames", "ocr", "upload-frames"]  # no vision model: no vision
    assert [(s["id"], s["where"]) for s in gpu[5:]] == [
        ("prepare-audio", "this"), ("waiting-for-gpu", "gpu"), ("transcribe", "gpu"), ("align", "gpu"),
        ("diarize", "gpu"), ("write", "this")]
    cfg.whisper_url = ""
    own = steps.plan({"kind": "transcribe", "input": {"source": {"kind": "local"}, "options": {"mode": "local"}}}, cfg)
    assert own[0] == {"id": "download", "where": "this", "res": "disk", "state": "next"}
    assert own[3] == {"id": "transcribe", "where": "this", "res": "gpu", "gpu": "RTX 4070", "state": "next"}


def test_language_models_are_placed_by_their_model_info(cfg, llm):
    llm.info = [{"model_name": "osszefoglalo", "model_info": {"dwyt": {
        "where": "local", "target_label": "Qwen3.6-35B", "gpu": "RTX 3090",
        "fallback": [{"name": "cloud/flash", "where": "cloud", "provider": "OpenRouter"}]}}},
        {"model_name": "felho", "model_info": {"dwyt": {"where": "cloud", "provider": "OpenRouter",
                                                        "target_label": "qwen/flash"}}}]
    modelinfo.reset()
    cfg.llm_models = ["felho"]
    assert steps.llm_place(cfg, "osszefoglalo") == {"where": "gpu", "res": "gpu", "gpu": "RTX 3090",
                                                     "model": "Qwen3.6-35B", "fallback": "cloud"}
    assert steps.llm_place(cfg, "felho") == {"where": "cloud", "res": "wait", "via": "OpenRouter",
                                             "model": "qwen/flash"}
    cfg.llm_url = "https://openrouter.ai/api/v1"
    assert steps.llm_place(cfg, "x")["via"] == "OpenRouter"
    plan = steps.plan({"kind": "summarize", "input": {"categories": [1], "chapters": True}}, cfg, "x")
    assert ids(plan) == ["summarize", "categorize", "chapters", "write"]


def test_the_tracker_follows_the_stages():
    t = steps.Tracker([steps._step("download", "this", "net"), steps._step("frames", "this", "cpu"),
                       steps._step("vision", "gpu", "gpu"), steps._step("transcribe", "cloud", "wait")])
    t.enter("starting")  # the worker's first moment: no step
    t.enter("download")
    t.enter("frames")
    t.note(res="cpu")
    t.enter("transcribe")  # no vision model answered: the vision step never came and is dropped
    t.enter("download")  # again (the audio fetched meanwhile): ignored
    t.enter("align")  # not foreseen: added after the current one, with its usual place
    rows = t.payload()
    assert ids(rows) == ["download", "frames", "transcribe", "align"]
    assert [r["state"] for r in rows] == ["done", "done", "done", "now"] and rows[0]["s"] >= 0
    t.note("transcribe", gpu="RTX 3090", res="gpu")
    assert t.payload()[2]["gpu"] == "RTX 3090"
    t.note("transcribe", gpu="")  # a CPU retry: the card's name goes
    assert "gpu" not in t.payload()[2]
    assert ids(t.ran()) == ids(rows) and all(r["state"] == "done" for r in t.ran())


def test_a_local_job_reports_the_gpu_hosts_card(cfg, central, whisper, monkeypatch):
    cfg.whisper_url = whisper.url
    yt_dlp.REGISTRY["videos"][VIDEO] = _video()
    central.add_job(90, "transcribe", {"source": {"kind": "youtube", "url": VIDEO},
                                       "options": {"mode": "local", "model": "large-v3", "language": "hu"}})
    _run_once(cfg, central)
    ran = central.results[90]["steps"]
    assert ids(ran) == ["download", "prepare-audio", "waiting-for-gpu", "transcribe", "diarize"]
    tr = next(s for s in ran if s["id"] == "transcribe")
    assert tr == {**tr, "where": "gpu", "res": "gpu", "gpu": "RTX 3090", "state": "done"} and "s" in tr
    beats = [p["steps"] for j, p in central.progress if j == 90]
    assert beats and any(s["state"] == "now" for s in beats[0])
