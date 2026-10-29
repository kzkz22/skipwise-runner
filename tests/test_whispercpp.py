"""Built-in local Whisper (decision 18.6): whisper.cpp (faked as a program) + sherpa-onnx (faked)."""
import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest
from conftest import Server
from fastapi import FastAPI
from fastapi.responses import Response

from dwyt_runner import config, local, media, whispercpp, work
from dwyt_runner.errors import WorkError

FAKE_CLI = r'''#!{python}
import json, os, sys
args = sys.argv[1:]
if "--help" in args:
    if not os.environ.get("FAKE_WHISPER_OLD"):
        sys.stderr.write("  --prompt PROMPT  [       ] initial prompt\n"
                         "  --carry-initial-prompt [false] always prepend\n")
    sys.exit(0)
log = os.environ["FAKE_WHISPER_LOG"]
open(log, "a", encoding="utf-8").write(" ".join(args) + "\n")
flag = os.environ.get("FAKE_WHISPER_GPU_FAIL")
if "-ng" not in args:
    sys.stderr.write("whisper_backend_init_gpu: using CUDA0 backend\n")
    for i in range(60):  # a long log: the backend line scrolls far up
        sys.stderr.write("whisper_model_load: some detail %d\n" % i)
    if flag and os.path.exists(flag):
        os.remove(flag)
        sys.stderr.write("CUDA error: out of memory\n")
        sys.exit(3)
for p in (10, 60, 100):
    sys.stderr.write("whisper_print_progress_callback: progress = %d%%\n" % p)
base = args[args.index("-of") + 1]
words = ["Sziasztok!", "Hova", "utaznál?", "Berlinbe", "megyek."]
probs = [0.9, 0.95, 0.8, 0.2, 0.9]  # "Berlinbe" is uncertain
segs = [{{"text": " " + w, "offsets": {{"from": i * 2000, "to": i * 2000 + 1500}},
         "tokens": [{{"text": "[_BEG_]", "p": 0.01}}, {{"text": " " + w, "p": probs[i]}}]}}
        for i, w in enumerate(words)]
segs.insert(0, {{"text": "[_BEG_]", "offsets": {{"from": 0, "to": 0}}}})
sys.stderr.write("whisper_print_timings:     load time =  1500.00 ms\n")
lang = args[args.index("-l") + 1]
json.dump({{"result": {{"language": "hu" if lang == "auto" else lang}}, "transcription": segs}},
          open(base + ".json", "w", encoding="utf-8"))
'''


@pytest.fixture
def builtin(tmp_path, monkeypatch):
    script = tmp_path / "fake_whisper_cli.py"
    script.write_text(FAKE_CLI.format(python=sys.executable), encoding="utf-8")
    if os.name == "nt":
        cli = tmp_path / "whisper-cli.cmd"
        cli.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        cli = script
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "cli.log"
    monkeypatch.setenv("FAKE_WHISPER_LOG", str(log))
    models = whispercpp.models_dir() / "whisper"
    models.mkdir(parents=True)
    for url in [*whispercpp.MODELS.values(), whispercpp.VAD_MODEL]:
        (models / Path(url).name).write_bytes(b"model")
    turns = []

    def fake_diarize(wav, speakers, threshold, stage):
        turns.append((speakers, threshold))
        stage("diarize")
        return [(0.0, 5.9, "S0"), (5.9, 10.0, "S1")]
    monkeypatch.setattr(whispercpp, "diarize", fake_diarize)
    monkeypatch.setattr(whispercpp, "speaker_models", lambda stage=None: (Path("seg.onnx"), Path("emb.onnx")))
    cfg = config.Config(whisper_builtin=True, whisper_cli=str(cli))
    return cfg, log, turns


class FileFetch(media.Fetch):
    def __init__(self, path):
        self.path = path

    def info(self):
        return {"title": "felvétel", "duration": 10}

    def get(self, workdir):
        return self.path


def test_local_job_on_this_machine(builtin, audio_file):
    cfg, log, turns = builtin
    stages = []
    out = local.run(FileFetch(audio_file), {"mode": "local", "language": None, "diarize": True}, cfg, stages.append)
    t = out["transcript"]
    assert (t["mode"], t["model"], t["language"], t["diarized"]) == ("local", "large-v3", "hu", True)
    assert [(s["speaker"], s["text"]) for s in t["segments"]] == [
        ("Szereplő 1", "Sziasztok!"), ("Szereplő 1", "Hova utaznál?"), ("Szereplő 2", "Berlinbe megyek.")]
    assert [s["label"] for s in t["speakers"]] == ["Szereplő 1", "Szereplő 2"]
    assert t["segments"][2]["low"] == [[0, 8]] and "low" not in t["segments"][0]  # "Berlinbe": p 0.2
    assert t["warnings"] == []  # it ran on the "video card"
    assert turns == [(None, 0.8)]
    args = log.read_text(encoding="utf-8").split()
    for flag in ("--vad", "-vm", "-mc", "-ml", "-sow", "-ojf", "-pp"):
        assert flag in args
    assert args[args.index("-l") + 1] == "auto" and args[args.index("-mc") + 1] == "0"
    assert "ggml-silero" in args[args.index("-vm") + 1]
    assert {"prepare-audio", "transcribe", "diarize"} <= set(stages)


def test_speaker_count_language_threshold_and_no_speakers(builtin, audio_file):
    cfg, log, turns = builtin
    cfg.diarize_threshold = 0.7
    whispercpp.transcribe(audio_file, {"language": "en", "num_speakers": 3}, "large-v3-turbo", cfg,
                          str(Path(log).parent), lambda s: None, lambda: False, 600)
    whispercpp.transcribe(audio_file, {"min_speakers": 2, "max_speakers": 2}, "large-v3", cfg,
                          str(Path(log).parent), lambda s: None, lambda: False, 600)
    assert turns == [(3, 0.7), (2, 0.7)]
    lines = log.read_text(encoding="utf-8").splitlines()
    assert "-l en" in lines[0] and "ggml-large-v3-turbo.bin" in lines[0]
    raw = whispercpp.transcribe(audio_file, {"diarize": False}, "large-v3", cfg, str(Path(log).parent),
                                lambda s: None, lambda: False, 600)
    assert raw["diarized"] is False and len(turns) == 2


def test_terms_fill_the_prompt_and_an_old_program_is_told(builtin, audio_file, monkeypatch):
    """19c round A: the user's terms go into the initial prompt, padded so no previous text gets in."""
    cfg, log, turns = builtin
    whispercpp._FLAGS.clear()
    out = whispercpp.transcribe(audio_file, {"terms": ["Kovács Ödön", "Kubernetes", "kubernetes", " OKR, KPI "]},
                                "large-v3", cfg, str(Path(log).parent), lambda s: None, lambda: False, 600)
    line = log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--carry-initial-prompt" in line and "-mc" not in line.split()
    assert "Kovács Ödön, Kubernetes, OKR KPI." in line and line.count("Kubernetes") >= 10  # repeated to fill
    assert out["warnings"] == []
    # an older whisper-cli without --carry-initial-prompt: no prompt, -mc 0 as before, and a warning
    whispercpp._FLAGS.clear()
    monkeypatch.setenv("FAKE_WHISPER_OLD", "1")
    out = whispercpp.transcribe(audio_file, {"terms": ["Kubernetes"]}, "large-v3", cfg, str(Path(log).parent),
                                lambda s: None, lambda: False, 600)
    args = log.read_text(encoding="utf-8").splitlines()[-1].split()
    assert "--prompt" not in args and args[args.index("-mc") + 1] == "0"
    assert any("régebbi" in w for w in out["warnings"])
    whispercpp._FLAGS.clear()


def test_failed_gpu_run_is_repeated_on_the_processor(builtin, audio_file, tmp_path, monkeypatch):
    cfg, log, _ = builtin
    flag = tmp_path / "gpu_fail"
    flag.write_text("x")
    monkeypatch.setenv("FAKE_WHISPER_GPU_FAIL", str(flag))
    out = local.run(FileFetch(audio_file), {"diarize": False}, cfg, lambda s: None)
    assert out["transcript"]["warnings"] == ["A videókártyán nem sikerült, ezért a processzoron futott (lassabb).",
                                             "Az átírás a processzoron futott."]
    assert log.read_text(encoding="utf-8").splitlines()[-1].endswith("-ng")


def test_blocked_program_and_unknown_model(builtin, audio_file, monkeypatch):
    cfg, _, _ = builtin
    with pytest.raises(WorkError, match="nem ismeri"):
        local.run(FileFetch(audio_file), {"model": "medium"}, cfg, lambda s: None)

    real = whispercpp.subprocess.Popen

    def blocked(args, *rest, **kwargs):
        if args[0] != cfg.whisper_cli:  # ffmpeg still runs
            return real(args, *rest, **kwargs)
        err = OSError(22, "Egy alkalmazásvezérlési házirend letiltotta ezt a fájlt")
        err.winerror = 4551
        raise err
    monkeypatch.setattr(whispercpp.subprocess, "Popen", blocked)
    with pytest.raises(WorkError, match="alkalmazásvezérlése") as exc:
        local.run(FileFetch(audio_file), {}, cfg, lambda s: None)
    assert exc.value.retry == "never"


def test_capabilities_offer_local_mode(builtin):
    cfg, _, _ = builtin
    caps = work.capabilities(cfg)
    assert "local" in caps["transcribe_modes"] and caps["whisper_models"] == ["large-v3", "large-v3-turbo"]
    assert "local" not in work.capabilities(config.Config())["transcribe_modes"]
    c = config.Config()
    config.set_value(c, "whisper_builtin", "igen")
    config.set_value(c, "diarize_threshold", "0,75")
    assert c.whisper_builtin is True and c.diarize_threshold == 0.75


def test_missing_program_is_explained(monkeypatch, tmp_path):
    monkeypatch.setattr(whispercpp.sys, "platform", "linux")
    monkeypatch.setattr(whispercpp.shutil, "which", lambda name: None)
    with pytest.raises(WorkError, match="config set whisper_cli"):
        whispercpp.whisper_cli(config.Config())
    with pytest.raises(WorkError, match="Nincs whisper-cli ebben a mappában"):
        whispercpp.whisper_cli(config.Config(whisper_cli=str(tmp_path)))
    (tmp_path / "Release").mkdir()
    exe = tmp_path / "Release" / "whisper-cli.exe"
    exe.write_bytes(b"")
    assert whispercpp.whisper_cli(config.Config(whisper_cli=str(tmp_path))) == str(exe)


def test_words_get_the_speaker_they_overlap_most():
    words = [{"text": "a", "start": 0.0, "end": 1.0}, {"text": "b", "start": 1.8, "end": 2.4},
             {"text": "c", "start": 9.0, "end": 9.5}]
    whispercpp.assign_speakers(words, [(0.0, 2.0, "S0"), (2.0, 5.0, "S1")])
    assert [w["speaker"] for w in words] == ["S0", "S1", "S1"]


@pytest.fixture(scope="module")
def files():
    app = FastAPI()
    body = b"ggml model bytes"

    @app.get("/ok.bin")
    def ok():
        return Response(body)

    @app.get("/missing.bin")
    def missing():
        return Response(status_code=404)
    s = Server(app)
    yield s.url, body
    s.stop()


def test_downloads_mirrors_and_checksum(files, tmp_path):
    url, body = files
    seen = []
    dest = whispercpp.download([f"{url}/missing.bin", f"{url}/ok.bin"], tmp_path / "m.bin", seen.append)
    assert dest.read_bytes() == body and seen == ["download-model", "download-model"]
    assert whispercpp.download(f"{url}/missing.bin", dest) == dest  # already there: no download
    with pytest.raises(WorkError, match="ellenőrzőösszege"):
        whispercpp.download(f"{url}/ok.bin", tmp_path / "bad.bin", sha256="0" * 64)
    good = hashlib.sha256(body).hexdigest()
    assert whispercpp.download(f"{url}/ok.bin", tmp_path / "good.bin", sha256=good).exists()
    with pytest.raises(WorkError) as exc:
        whispercpp.download(f"{url}/missing.bin", tmp_path / "none.bin")
    assert exc.value.retry == "later" and not (tmp_path / "none.bin.part").exists()


def test_try_command(builtin, audio_file, capsys):
    cfg, _, _ = builtin
    config.save(config.Config(whisper_cli=cfg.whisper_cli))
    from dwyt_runner import cli
    assert cli.whisper_cmd("try", "large-v3", audio_file, None, None) == 0
    out = capsys.readouterr().out
    assert "large-v3 · hu · 2 beszélő" in out
    assert "[00:00] Szereplő 1: Sziasztok! Hova utaznál?" in out and "Szereplő 2: Berlinbe megyek." in out


def test_the_windows_build_is_picked_from_the_release():
    names = ["whisper-bin-Win32.zip", "whisper-bin-x64.zip", "whisper-blas-bin-x64.zip",
             "whisper-cublas-11.8.0-bin-x64.zip", "whisper-cublas-12.4.0-bin-x64.zip", "whisper-bin-ubuntu-x64.tar.gz",
             "whisper-v1.9.5-xcframework.zip", "whisper-bin-win-arm64.zip"]
    assets = [{"name": n, "browser_download_url": f"https://example/{n}"} for n in names]
    assert whispercpp.pick_asset(assets, True)["name"] == "whisper-cublas-12.4.0-bin-x64.zip"
    assert whispercpp.pick_asset(assets, False)["name"] == "whisper-blas-bin-x64.zip"
    other = [{"name": n, "browser_download_url": "u"} for n in ("whisper-bin-win-cuda-12.4.0-x64.zip",
                                                                  "whisper-bin-win-cpu-x64.zip")]
    assert whispercpp.pick_asset(other, True)["name"] == "whisper-bin-win-cuda-12.4.0-x64.zip"
    assert whispercpp.pick_asset(other, False)["name"] == "whisper-bin-win-cpu-x64.zip"
    assert whispercpp.pick_asset([], True) is None


def test_the_newest_release_with_a_windows_build_is_used():
    releases = [{"tag_name": "v1.9.5", "assets": []}, {"tag_name": "b7001", "draft": True, "assets": [
        {"name": "whisper-bin-win-cuda-12.4.0-x64.zip", "browser_download_url": "draft"}]},
        {"tag_name": "b7000", "assets": [{"name": "whisper-bin-win-cuda-12.4.0-x64.zip", "browser_download_url": "u"},
                                         {"name": "whisper-blas-bin-x64.zip", "browser_download_url": "u2"}]}]
    tag, asset = whispercpp.pick_release(releases, True)
    assert (tag, asset["browser_download_url"]) == ("b7000", "u")
    assert whispercpp.pick_release(releases, False)[1]["name"] == "whisper-blas-bin-x64.zip"
    assert whispercpp.pick_release(releases[:1], True) == ("", None)
