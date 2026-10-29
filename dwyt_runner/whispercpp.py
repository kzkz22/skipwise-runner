"""Built-in local Whisper on this machine (plan 15b, decision 18.6): whisper.cpp for the text and
sherpa-onnx for the speakers, no PyTorch, no Hugging Face token, no separate GPU host.

The settings measured on 2026-10-13 (dwyt-runner/tools/whisper_probe.py, a Hungarian street interview
with 8 speakers, against a hand-corrected transcript): whisper.cpp large-v3 with Silero speech detection
(--vad) and no previous-text context (-mc 0) gave 7.7% word errors (WhisperX: 5.7%) and no invented
sentences; sherpa-onnx with the pyannote 3.0 segmentation, NeMo TitaNet speaker embeddings and automatic
clustering at threshold 0.8 found all 8 speakers and gave 95.8% of the words the right one (WhisperX:
94.6%). Both models have free licences (MIT, CC-BY-4.0); the better-scoring reverb-v1 segmentation is
non-commercial and is not used.

whisper.cpp runs as its official program (whisper-cli). On Windows the runner downloads the official
release itself (the newest release that carries Windows builds: the CUDA build when an NVIDIA card is
present, else the OpenBLAS build), checked against the SHA-256 that GitHub publishes; elsewhere whisper-cli
comes from PATH or the whisper_cli setting.
Windows application control may refuse the unsigned program: that is said plainly, never worked around.
A failed GPU run is repeated on the processor with a warning (that is still local mode, not a switch).
Models live in the runner's models folder and are downloaded once.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import wave
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from . import machine, tools
from .config import Config
from .errors import WorkError
from .i18n import gettext as _

HF = "https://huggingface.co"
RELEASES_API = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=30"
MODELS = {"large-v3": f"{HF}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
          "large-v3-turbo": f"{HF}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"}
VAD_MODEL = f"{HF}/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = (f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
                f"{HF}/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx")
EMBEDDING = (f"{SHERPA}/speaker-recongition-models/nemo_en_titanet_large.onnx",  # the typo is upstream's
             f"{HF}/csukuangfj/speaker-embedding-models/resolve/main/nemo_en_titanet_large.onnx")
DEFAULT_THRESHOLD = 0.8
PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")
LOAD_TIME = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
BLOCKED = (4551, 1260)  # Windows: "an application control policy has blocked this file"


# the setup window follows downloads: (file name, bytes so far, total bytes or 0)
ON_DOWNLOAD: Callable[[str, int, int], None] | None = None


def models_dir() -> Path:
    return machine.models_dir()


def available(cfg: Config) -> bool:
    return bool(cfg.whisper_builtin)


# ---- downloads ---------------------------------------------------------------------------------

def download(urls: str | list[str], dest: Path, stage: Callable[[str], None] | None = None,
             sha256: str | None = None) -> Path:
    """Fetch once into the models folder (several URLs = mirrors). A partial file never counts."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    errors = []
    for url in [urls] if isinstance(urls, str) else urls:
        if stage:
            stage("download-model")
        digest = hashlib.sha256()
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=120),
                              headers={"User-Agent": "dwyt-runner"}) as r:
                if r.status_code != 200:
                    errors.append(f"{url}: HTTP {r.status_code}")
                    continue
                total = int(r.headers.get("content-length") or 0)
                got = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        digest.update(chunk)
                        got += len(chunk)
                        if ON_DOWNLOAD:
                            ON_DOWNLOAD(dest.name, got, total)
        except httpx.HTTPError as exc:
            errors.append(f"{url}: {exc}")
            continue
        if sha256 and digest.hexdigest() != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise WorkError(_("A letöltött fájl ellenőrzőösszege nem egyezik: {name}. Nem használom.").format(
                name=dest.name))
        tmp.replace(dest)
        return dest
    tmp.unlink(missing_ok=True)
    raise WorkError(_("Nem sikerült letölteni: {name} ({join}); később újrapróbálom.").format(
        name=dest.name, join='; '.join(errors)[:400]),
                    "later")


def _unpack_tar(archive: Path, into: Path) -> None:
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(into, filter="data")
        except TypeError:  # Python without extraction filters
            tf.extractall(into)


def whisper_model(name: str, stage: Callable[[str], None] | None = None) -> Path:
    if name not in MODELS:
        raise WorkError(_("Ismeretlen Whisper-modell: {name} ({known}).").format(name=name, known=", ".join(MODELS)))
    return download(MODELS[name], models_dir() / "whisper" / Path(MODELS[name]).name, stage)


def vad_model(stage: Callable[[str], None] | None = None) -> Path:
    return download(VAD_MODEL, models_dir() / "whisper" / Path(VAD_MODEL).name, stage)


def speaker_models(stage: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    folder = models_dir() / "speakers"
    seg = folder / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if not seg.exists():
        try:
            _unpack_tar(download(SEGMENTATION[0], folder / Path(SEGMENTATION[0]).name, stage), folder)
            (folder / Path(SEGMENTATION[0]).name).unlink(missing_ok=True)
        except WorkError:
            download(SEGMENTATION[1], seg, stage)
        if not seg.exists():
            raise WorkError(_("A beszélő-szétválasztó modell csomagjában nincs model.onnx."))
    emb = download(list(EMBEDDING), folder / Path(EMBEDDING[0]).name, stage)
    return seg, emb


def prepared(cfg: Config, model: str) -> bool:
    """Everything the built-in Whisper needs is here already (nothing is downloaded to find out)."""
    folder = models_dir()
    files = [folder / "whisper" / Path(MODELS[model]).name if model in MODELS else None,
             folder / "whisper" / Path(VAD_MODEL).name,
             folder / "speakers" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx",
             folder / "speakers" / Path(EMBEDDING[0]).name]
    if not all(f is not None and f.exists() and f.stat().st_size > 0 for f in files):
        return False
    if cfg.whisper_cli:
        return Path(cfg.whisper_cli).exists()
    if sys.platform == "win32":
        cli = folder / "whisper.cpp"
        return cli.exists() and any(cli.glob("*/**/whisper-cli.exe"))
    return shutil.which("whisper-cli") is not None


def prepare(cfg: Config, model: str, step: Callable[[str], None] | None = None) -> None:
    """Fetch what the built-in Whisper needs (the setup window's Hozzávalók step, `whisper prepare`)."""
    for label, get in (("whisper.cpp program", lambda: whisper_cli(cfg)),
                       (_("beszédfelismerő modell"), lambda: whisper_model(model)),
                       (_("beszédérzékelő (VAD)"), vad_model),
                       (_("beszélő-szétválasztó modellek"), speaker_models)):
        if step:
            step(label)
        get()


def pick_release(releases: list[dict], nvidia: bool) -> tuple[str, dict | None]:
    """The newest published release that has a fitting Windows build: whisper.cpp's version tags
    (v1.9.5) may come without files, the builds sit on other releases."""
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        asset = pick_asset(rel.get("assets") or [], nvidia)
        if asset:
            return str(rel.get("tag_name") or "whisper.cpp"), asset
    return "", None


def pick_asset(assets: list[dict], nvidia: bool) -> dict | None:
    """The Windows x64 build to use: CUDA 12 (then any CUDA) with an NVIDIA card, else OpenBLAS, else plain."""
    zips = [a for a in assets if isinstance(a, dict) and str(a.get("name", "")).endswith(".zip")
            and a.get("browser_download_url")]
    win = [a for a in zips if re.search(r"x64", a["name"], re.I) and not re.search(r"ubuntu|linux|mac|xcframework|"
                                                                                   r"arm64|win32", a["name"], re.I)]
    order = ([r"(cuda|cublas)-?12", r"cuda|cublas"] if nvidia else []) + [r"blas", r"."]
    for pattern in order:
        for a in win:
            name = a["name"].lower()
            if re.search(pattern, name) and (nvidia or not re.search(r"cuda|cublas", name)) and \
                    (pattern != "blas" or "cublas" not in name):
                return a
    return None


def whisper_cli(cfg: Config, stage: Callable[[str], None] | None = None) -> str:
    """The whisper-cli program: the setting, then PATH, then (Windows) the official release."""
    if cfg.whisper_cli:
        path = Path(cfg.whisper_cli)
        if path.is_dir():
            hits = sorted(path.rglob("whisper-cli.exe")) + sorted(path.rglob("whisper-cli"))
            if not hits:
                raise WorkError(_("Nincs whisper-cli ebben a mappában: {path} (whisper_cli).").format(path=path))
            path = hits[0]
        if not path.exists():
            raise WorkError(_("Nem található: {path} (whisper_cli).").format(path=path))
        return str(path)
    folder = models_dir() / "whisper.cpp"
    if sys.platform == "win32":
        done = sorted(folder.glob("*/**/whisper-cli.exe")) if folder.exists() else []
        if done:
            return str(done[0])
        try:  # the releases' files and their SHA-256 as GitHub lists them, newest first
            r = httpx.get(RELEASES_API, timeout=30, headers={"User-Agent": "dwyt-runner"})
            releases = r.json() if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError) as exc:
            raise WorkError(_("Nem érem el a whisper.cpp kiadásait a GitHubon ({exc}); később "
                               "újrapróbálom.").format(
                              exc=exc),
                            "later") from exc
        tag, asset = pick_release(releases if isinstance(releases, list) else [], bool(machine.nvidia_smi()))
        if asset is None:
            raise WorkError(_("A whisper.cpp legutóbbi kiadásaiban nem találok Windowsos csomagot. Töltsd le kézzel, "
                            "és add meg a mappáját: skipwise-runner config set whisper_cli MAPPA"))
        name = asset["name"]
        digest = str(asset.get("digest") or "")
        target = folder / f"{tag}-{name[:-4]}"
        archive = download(asset["browser_download_url"], folder / name, stage,
                           sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else None)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        archive.unlink(missing_ok=True)
        hits = sorted(target.rglob("whisper-cli.exe"))
        if hits:
            return str(hits[0])
        raise WorkError(_("A whisper.cpp csomagjában nincs whisper-cli.exe ({name}).").format(name=name))
    found = shutil.which("whisper-cli")
    if found:
        return found
    raise WorkError(_("Ezen a gépen nincs whisper-cli. Telepítsd (például a whisper.cpp csomagjából), vagy add meg a "
                    "helyét: skipwise-runner config set whisper_cli /út/whisper-cli"))


# ---- running ------------------------------------------------------------------------------------

def to_wav(src: str, workdir: str) -> str:
    """whisper.cpp and sherpa-onnx both read 16 kHz mono 16-bit WAV."""
    dst = os.path.join(workdir, "audio.wav")
    try:
        subprocess.run([tools.ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", dst], check=True, capture_output=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        raise WorkError(_("Az ffmpeg nem tudta kinyerni a hangot: ") + exc.stderr.decode("utf-8", "replace")[-300:]) \
            from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkError(_("Az ffmpeg nem indult el vagy nem végzett: {exc}").format(exc=exc)) from exc
    return dst


def _no_window() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def run_cli(args: list[str], stage: Callable[[str], None], cancelled: Callable[[], bool],
            deadline: float) -> tuple[int, str]:
    """Run whisper-cli, following its progress; (exit code, the end of its log)."""
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", **_no_window())
    except OSError as exc:
        if getattr(exc, "winerror", None) in BLOCKED:
            raise WorkError(_("A Windows alkalmazásvezérlése (Intelligens alkalmazásvezérlés vagy céges házirend) "
                            "nem engedi futtatni a whisper.cpp programját. Ezen a gépen a helyi átírás így nem megy; "
                            "kérj felhős átírást, vagy használd egy másik gépen.")) from exc
        raise WorkError(_("A whisper.cpp nem indult el: {exc}").format(exc=exc)) from exc
    tail: list[str] = []
    notes: list[str] = []  # the backend lines come first and would scroll out of the tail
    last = -1
    try:
        for line in proc.stderr:
            tail = (tail + [line.rstrip()])[-40:]
            if len(notes) < 20 and re.search(r"backend|CUDA|Vulkan|Metal|ggml_cuda", line):
                notes.append(line.rstrip())
            m = PROGRESS.search(line)
            if m and int(m.group(1)) != last:
                last = int(m.group(1))
                stage("transcribe")
            if cancelled():
                raise WorkError(_("megszakítva"))
            if time.monotonic() > deadline:
                raise WorkError(_("A helyi átírás túl sokáig tartott; a munka leállt (whisper_timeout_min)."))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return proc.returncode, "\n".join(notes + tail)


def device_of(log: str) -> str:
    return "cpu" if not re.search(r"using (CUDA|Vulkan|Metal)\S* backend", log) else "gpu"


_FLAGS: dict[str, bool] = {}


def supports(cli: str, flag: str) -> bool:
    """Whether this whisper-cli knows a flag (its --help lists it); asked once per program."""
    key = f"{cli}\0{flag}"
    if key not in _FLAGS:
        try:
            out = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=30,
                                 encoding="utf-8", errors="replace", **_no_window())
            _FLAGS[key] = flag in (out.stdout or "") + (out.stderr or "")
        except (OSError, subprocess.SubprocessError):
            _FLAGS[key] = False
    return _FLAGS[key]


def word_p(seg: dict) -> float | None:
    """A word's probability: the lowest probability of its text tokens (-ojf output); None without tokens."""
    ps = []
    for tok in seg.get("tokens") or []:
        text = str(tok.get("text") or "")
        if text.startswith("[_") or not text.strip() or tok.get("p") is None:
            continue
        try:
            ps.append(float(tok["p"]))
        except (TypeError, ValueError):
            continue
    return min(ps) if ps else None


def transcribe_text(wav: str, model: Path, vad: Path, language: str | None, cfg: Config, workdir: str,
                    stage: Callable[[str], None], cancelled: Callable[[], bool],
                    deadline: float, term_list: list[str] | None = None
                    ) -> tuple[list[dict], str, str, list[str], float]:
    """(words with start/end/p, language, device, warnings, model load seconds)."""
    from . import terms
    cli = whisper_cli(cfg, stage)
    base = os.path.join(workdir, "whisper")
    args = [cli, "-m", str(model), "-f", wav, "-l", language or "auto", "-ml", "1", "-sow", "-ojf", "-of", base,
            "-pp", "--vad", "-vm", str(vad)]
    warnings: list[str] = []
    prompt, left_out = terms.whisper_cpp_prompt(term_list or [])
    if prompt and supports(cli, "--carry-initial-prompt"):
        # the terms fill the whole prompt budget, so no previous text gets in (that is what -mc 0 kept out)
        args += ["--prompt", prompt, "--carry-initial-prompt"]
        if left_out:
            warnings.append(_("A szakszavak közül {n} nem fért a Whisper promptjába.").format(n=left_out))
    else:
        args += ["-mc", "0"]
        if prompt:
            warnings.append(_("Ez a whisper.cpp régebbi, ezért a szakszavakat nem kapta meg; frissítsd a futtatót "
                              "vagy a whisper-cli-t."))
    if cfg.whisper_threads:
        args += ["-t", str(cfg.whisper_threads)]
    stage("transcribe")
    code, log = run_cli(args, stage, cancelled, deadline)
    device = device_of(log)
    if code != 0 and re.search(r"CUDA|Vulkan|Metal", log):  # the video card failed: once more on the processor
        warnings.append(_("A videókártyán nem sikerült, ezért a processzoron futott (lassabb)."))
        code, log = run_cli(args + ["-ng"], stage, cancelled, deadline)
        device = "cpu"
    if code != 0:
        raise WorkError(_("A whisper.cpp hibával állt le ({code}): {log}").format(code=code, log=log[-600:]))
    data = json.loads(Path(base + ".json").read_text(encoding="utf-8", errors="replace"))
    found = str((data.get("result") or {}).get("language") or language or "").lower()
    words = []
    for seg in data.get("transcription") or []:
        text = str(seg.get("text") or "").strip()
        if not text or text.startswith("[_") or text.startswith("[BLANK"):
            continue
        off = seg.get("offsets") or {}
        words.append({"text": text, "start": off.get("from", 0) / 1000, "end": off.get("to", 0) / 1000,
                      "p": word_p(seg)})
    load = LOAD_TIME.search(log)  # whisper_print_timings: the model's loading, not the audio's length
    return words, found, device, warnings, float(load.group(1)) / 1000 if load else 0.0


def diarize(wav: str, speakers: int | None, threshold: float, stage: Callable[[str], None]) -> list[tuple]:
    """sherpa-onnx speaker turns: (start, end, speaker id)."""
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as exc:
        raise WorkError(_("A beszélők szétválasztásához hiányzik a sherpa-onnx. Telepítsd a futtatót a helyi "
                        "kiegészítővel: uv tool install --force --python 3.13 \"dwyt-runner[local] @ "
                        "https://github.com/kzkz22/skipwise-runner/archive/refs/heads/main.zip\"")) from exc
    seg, emb = speaker_models(stage)
    stage("diarize")
    threads = max(1, (os.cpu_count() or 2) - 1)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=speakers or -1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise WorkError(_("A beszélő-szétválasztás beállítása hibás."))
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    with wave.open(wav, "rb") as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return [(float(r.start), float(r.end), f"S{r.speaker}") for r in sd.process(samples).sort_by_start_time()]


def assign_speakers(words: list[dict], turns: list[tuple]) -> list[dict]:
    """Each word gets the speaker of the turn it overlaps most, else the nearest turn."""
    for w in words:
        s, e = w["start"], max(w["end"], w["start"])
        best, best_overlap, best_gap = None, 0.0, float("inf")
        for ts, te, who in turns:
            overlap = min(e, te) - max(s, ts)
            if overlap > best_overlap:
                best, best_overlap = who, overlap
            elif best_overlap <= 0:
                gap = ts - e if ts > e else s - te
                if gap < best_gap:
                    best, best_gap = who, gap
        w["speaker"] = best
    return words


def transcribe(src: str, options: dict[str, Any], model_name: str, cfg: Config, workdir: str,
               stage: Callable[[str], None], cancelled: Callable[[], bool], limit_s: int) -> dict[str, Any]:
    """The same raw shape as the GPU worker's result: language, segments (raw speaker ids), diarized, device."""
    from .formatting import segments_from_words
    deadline = time.monotonic() + limit_s
    model = whisper_model(model_name, stage)
    vad = vad_model(stage)
    stage("prepare-audio")
    wav = to_wav(src, workdir)
    from . import gpuswap
    swap_started = time.monotonic()
    gpuswap.before_whisper(cfg, stage, cancelled)  # one video card: the language model makes room
    swap_s = time.monotonic() - swap_started
    try:
        from . import terms
        words, language, device, warnings, load_s = transcribe_text(wav, model, vad, options.get("language"), cfg,
                                                            workdir, stage, cancelled, deadline,
                                                            terms.clean(options.get("terms")))
    finally:
        gpuswap.after_whisper()
    diarized = False
    if words and options.get("diarize", True):
        count = options.get("num_speakers") or (options.get("min_speakers") if options.get("min_speakers") and
                                                options.get("min_speakers") == options.get("max_speakers") else None)
        turns = diarize(wav, int(count) if count else None, cfg.diarize_threshold or DEFAULT_THRESHOLD, stage)
        if turns:
            assign_speakers(words, turns)
            diarized = True
    return {"language": language, "segments": segments_from_words(words), "diarized": diarized, "device": device,
            "load_s": load_s, "swap_s": swap_s,
            "warnings": warnings}
