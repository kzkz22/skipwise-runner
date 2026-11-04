"""Plan 19c round C (decision 19.33): the slides of a video before the transcription — keyframes, OCR, the vision
model, terms for Whisper / AssemblyAI, the frames sent to the central service."""
import hashlib
import json
import subprocess

import pytest
import yt_dlp

from dwyt_runner import slides, tools, work
from test_runner import VIDEO, _run_once, _video


@pytest.fixture(scope="session")
def slide_video(tmp_path_factory) -> str:
    """36 s: three different pictures, 12 s each, with a tone."""
    path = tmp_path_factory.mktemp("slides") / "eloadas.mp4"
    src = ["smptebars", "testsrc", "rgbtestsrc"]
    cmd = [tools.ffmpeg(), "-nostdin", "-loglevel", "error", "-y"]
    for name in src:
        cmd += ["-f", "lavfi", "-i", f"{name}=s=320x240:r=5:d=12"]
    cmd += ["-f", "lavfi", "-i", "sine=f=440:d=36", "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


@pytest.fixture
def fake_ocr(monkeypatch):
    """Tesseract stand-in: every frame reads a slide text (no binary needed on any OS)."""
    texts = {1: "Negyedéves OKR áttekintés\nA Kubernetes klaszter és az S3 tároló",
             2: "Kovács Ödön bemutatja a GitHub Actions folyamatot", 3: ""}
    monkeypatch.setattr(slides, "tesseract", lambda cfg: "tesseract")
    monkeypatch.setattr(slides, "ocr_langs", lambda cmd, language: ("hun+eng", None))
    monkeypatch.setattr(slides, "ocr", lambda cmd, path, langs, data=None: texts[int(path[-8:-4]) % 3 + 1]
                        if path[-8:-4].isdigit() else "")
    return texts


def test_candidates_fill_long_gaps_and_keep_distance():
    assert slides.candidates([], 150) == [0.0, 60.0, 120.0]
    assert slides.candidates([2.0, 30.0, 31.0, 100.0], 120) == [0.0, 30.0, 90.0, 100.0]
    assert slides.candidates([], 0) == [0.0]


def test_keyframes_find_the_three_pictures(slide_video, tmp_path):
    ok, duration = slides.has_video(slide_video)
    assert ok and duration == pytest.approx(36, abs=0.5)
    frames = slides.keyframes(slide_video, duration, str(tmp_path), 300)
    assert [round(f.start) for f in frames] == [0, 12, 24]
    assert [f.n for f in frames] == [1, 2, 3] and frames[0].end == frames[1].start
    assert slides.distance(frames[0].hash, frames[1].hash) > slides.SAME_HASH
    with open(frames[0].path, "rb") as fh:
        assert fh.read(2) == b"\xff\xd8"  # a JPEG
    (tmp_path / "x").mkdir()
    two = slides.keyframes(slide_video, duration, str(tmp_path / "x"), 2)
    assert len(two) == 2 and [f.n for f in two] == [1, 2]


def test_a_sound_file_has_no_slides(audio_file, tmp_path, cfg):
    out = slides.run(audio_file, {}, cfg, str(tmp_path), "public", lambda s: None)
    assert out.frames == [] and "nincs videókép" in out.warnings[0]


def test_terms_put_the_users_first_and_never_whole_sentences():
    f1 = slides.Frame(n=1, start=0, text="Negyedéves OKR áttekintés\nA Kubernetes klaszter és az S3 tároló",
                      terms=["Kubernetes", "Ez egy nagyon hosszú mondat a dián, amely sosem lehet szakszó, mert túl "
                             "hosszú és Whisper nyelvét elvinné"])
    f2 = slides.Frame(n=2, start=10, text="OKR\nGitHub és iPhone, H.264")
    out = slides.terms([f1, f2], ["Kovács Ödön", "kubernetes"])
    assert out[:2] == ["Kovács Ödön", "kubernetes"]
    assert "Kubernetes" not in out[2:] and "OKR" in out and "S3" in out and "GitHub" in out and "H.264" in out
    assert not any(len(t) > 60 for t in out) and "és" not in out and "Negyedéves" not in out
    assert len(slides.terms([f1, f2], [f"Szó{i}" for i in range(100)])) == slides.TERMS_FOR_WHISPER


def test_vision_model_choice_follows_the_role(cfg, llm):
    from dwyt_runner import modelinfo
    assert slides.vision_model(cfg, "public") == ""  # no model info: nobody says it can see
    cfg.llm_vision_model = "latas"
    assert slides.vision_model(cfg, "public") == "latas"
    assert slides.vision_model(cfg, "private") == ""  # own recordings: only the private model, if it can see
    llm.info = [{"model_name": "osszefoglalo-helyi", "model_info": {"dwyt": {"where": "local", "vision": True}}}]
    modelinfo.reset()
    assert slides.vision_model(cfg, "private") == "osszefoglalo-helyi"


def test_the_step_reads_describes_and_uploads(slide_video, tmp_path, cfg, llm, fake_ocr):
    cfg.llm_vision_model = "latas"
    llm.vision_answers = [json.dumps({"kind": "slide", "title": "OKR", "text": "Célok",
                                      "terms": ["OKR", "Kubernetes", "x"]}),
                          "nem JSON, csak szöveg"]
    sent, stages = [], []
    out = slides.run(slide_video, {"language": "hu", "terms": ["Kovács Ödön"]}, cfg, str(tmp_path), "public",
                     stages.append, upload=lambda f: sent.append(f.n) or True)
    assert stages == ["frames", "ocr", "vision", "upload-frames"]
    assert len(out.frames) == 3 and sent == [1, 2, 3] and all(f.uploaded for f in out.frames)
    assert len(llm.vision_calls) == 2  # the third frame has no text: no request
    msg = llm.vision_calls[0]["messages"][0]["content"]
    assert msg[1]["image_url"]["url"].startswith("data:image/jpeg;base64,") and "OCR szerint" in msg[0]["text"]
    assert out.frames[0].kind == "slide" and out.frames[0].terms == ["OKR", "Kubernetes"]
    assert out.frames[1].kind == "" and out.frames[1].terms == []  # an unreadable answer leaves the frame plain
    assert out.terms[0] == "Kovács Ödön" and "Kubernetes" in out.terms and "GitHub" in out.terms
    fields = slides.result_fields(out)
    assert fields["slides"] == {"count": 3, "ocr": "tesseract (hun+eng)", "vision": "latas",
                                "terms": out.terms, "uploaded": 3}
    assert fields["frames"][0]["title"] == "OKR" and fields["frames"][2]["text"] == ""


def test_no_ocr_no_vision_and_a_refused_upload_only_warn(slide_video, tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(slides, "tesseract", lambda cfg: None)
    out = slides.run(slide_video, {}, cfg, str(tmp_path), "private", lambda s: None, upload=lambda f: False)
    text = " ".join(out.warnings)
    assert "Tesseract" in text and "látó modell" in text and "nem fogadta el" in text
    assert len(out.frames) == 3 and not any(f.uploaded for f in out.frames)


def test_url_job_with_slides_gives_terms_to_the_cloud(slide_video, cfg, central, assembly, llm, fake_ocr):
    cfg.llm_vision_model = "latas"
    yt_dlp.REGISTRY["videos"][VIDEO] = _video()
    yt_dlp.REGISTRY["audio"] = slide_video
    central.add_job(80, "transcribe", {"source": {"kind": "youtube", "url": VIDEO},
                                       "options": {"mode": "cloud", "model": "auto", "language": "hu",
                                                   "slides": True, "terms": ["Kovács Ödön"]}})
    _run_once(cfg, central)
    res = central.results[80]
    assert res["slides"]["count"] == 3 and res["slides"]["uploaded"] == 3
    assert [f["n"] for f in res["frames"]] == [1, 2, 3] and res["frames"][0]["title"] == "Negyedéves OKR"
    ctype, body = central.frames[(80, 1)]
    assert ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
    keyterms = assembly.requests[0]["keyterms_prompt"]
    assert keyterms[0] == "Kovács Ödön" and "OKR" in keyterms and "Kubernetes" in keyterms
    formats = [c[2].get("format") for c in yt_dlp.REGISTRY["calls"] if c[1]]
    assert any("height<=720" in (f or "") for f in formats) and len(formats) == 2  # the picture and the sound
    caps = central.claims[0]["capabilities"]
    assert caps["slides"] == {"ocr": True, "vision": True}


def test_captions_keep_their_own_text_but_get_the_slides(slide_video, cfg, central, fake_ocr):
    yt_dlp.REGISTRY["videos"][VIDEO] = _video()
    yt_dlp.REGISTRY["captions"]["https://captions.example/hu"] = json.dumps(
        {"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Jó napot."}]}]}).encode()
    yt_dlp.REGISTRY["audio"] = slide_video
    central.frames_refused = True
    central.add_job(81, "transcribe", {"source": {"kind": "youtube", "url": VIDEO},
                                       "options": {"mode": "captions", "language": None, "slides": True}})
    _run_once(cfg, central)
    res = central.results[81]
    assert res["transcript"]["mode"] == "captions" and res["slides"]["count"] == 3
    assert res["slides"]["uploaded"] == 0 and any("nem fogadta el" in w for w in res["transcript"]["warnings"])
    assert len([c for c in yt_dlp.REGISTRY["calls"] if c[1]]) == 1  # only the picture: captions need no sound


def test_a_failed_video_download_still_transcribes(cfg, central, assembly, monkeypatch):
    from dwyt_runner import media
    from dwyt_runner.errors import WorkError

    def broken(url, cfg, outdir):
        raise WorkError("nincs ilyen formátum")
    monkeypatch.setattr(media, "download_video", broken)
    yt_dlp.REGISTRY["videos"][VIDEO] = _video()
    central.add_job(82, "transcribe", {"source": {"kind": "youtube", "url": VIDEO},
                                       "options": {"mode": "cloud", "model": "auto", "language": "hu",
                                                   "slides": True}})
    _run_once(cfg, central)
    res = central.results[82]
    assert res["transcript"]["mode"] == "cloud" and res["slides"]["count"] == 0
    assert "A diák feldolgozása nem sikerült: nincs ilyen formátum" in res["transcript"]["warnings"][0]


def test_own_file_slides_use_only_the_local_model(slide_video, cfg, central, whisper, llm, fake_ocr):
    cfg.whisper_url = whisper.url
    cfg.llm_vision_model = "felho-latas"  # a cloud model: never for an own recording
    data = open(slide_video, "rb").read()
    central.files[83] = data
    central.add_job(83, "transcribe", {"source": {"kind": "upload", "sha256": hashlib.sha256(data).hexdigest(),
                                                  "name": "megbeszeles.mp4", "size": len(data),
                                                  "media": "/runner/v1/jobs/83/media"},
                                       "options": {"mode": "local", "model": "large-v3", "language": "hu",
                                                   "slides": True}})
    _run_once(cfg, central)
    res = central.results[83]
    assert res["slides"]["count"] == 3 and res["slides"]["vision"] is None and llm.vision_calls == []
    assert "OKR" in whisper.requests[0]["query"]["prompt"]
    assert len(central.frames) == 3


def test_capabilities_say_what_reads_the_slides(cfg, monkeypatch):
    monkeypatch.setattr(slides.shutil, "which", lambda cmd: None)
    assert work.capabilities(cfg)["slides"] == {"ocr": False, "vision": True}
    cfg.slides_ocr = "off"
    cfg.slides_vision = False
    assert work.capabilities(cfg)["slides"] == {"ocr": False, "vision": False}


def test_tesseract_is_found_where_the_windows_installer_puts_it(cfg, monkeypatch, tmp_path):
    exe = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(slides.shutil, "which", lambda cmd: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setattr(slides.os, "name", "nt")
    assert slides.tesseract(cfg) == str(exe)
    cfg.tesseract_cmd = str(tmp_path / "nincs.exe")  # an explicit path is not second-guessed
    assert slides.tesseract(cfg) is None


def test_slide_text_is_background_for_the_language_model_jobs(cfg, central, llm):
    central.add_job(84, "summarize", {"role": "public", "language": "hu",
                                      "meta": {"title": "Előadás", "slides_text": "[00:00] Negyedéves OKR\nCélok"},
                                      "transcript": {"language": "hu", "mode": "cloud", "diarized": False,
                                                     "segments": [{"start": 0, "end": 5, "text": "Jó napot."}]}})
    _run_once(cfg, central)
    assert 84 in central.results
    user = llm.calls[0]["messages"][-1]["content"]
    assert "A videón látszó diák szövege" in user and "Negyedéves OKR" in user


def test_corrections_see_what_was_on_screen():
    from dwyt_runner import refine
    segs = [{"idx": 0, "start": 0, "end": 5, "text": "Az ókáer célok jók.", "low": [[3, 9]]},
            {"idx": 1, "start": 200, "end": 205, "text": "Más téma."}]
    job = {"slides": [{"start": 0, "end": 60, "text": "Negyedéves OKR", "terms": ["OKR"]},
                      {"start": 190, "end": 300, "text": "Kubernetes klaszter", "terms": []}]}
    found = refine.spots(segs, [])
    text = slides.on_screen(job, segs, found, True)
    assert "OKR" in text and "Negyedéves OKR" in text and "Kubernetes" not in text
    assert slides.on_screen({}, segs, found, True) == ""


def test_ocr_languages_come_from_the_installed_tesseract_or_the_runners_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(slides, "tessdata_dir", lambda: str(tmp_path))
    monkeypatch.setattr(slides, "installed_langs", lambda cmd: {"eng", "hun", "osd"})
    assert slides.ocr_langs("tesseract", "hu") == ("hun+eng", None)  # apt's tesseract-ocr-hun: its own data
    fetched = []
    monkeypatch.setattr(slides, "installed_langs", lambda cmd: {"eng"})
    monkeypatch.setattr(slides, "fetch_tessdata", lambda code, folder: fetched.append(code) or True)
    assert slides.ocr_langs("tesseract", "de") == ("deu+eng", str(tmp_path))  # the runner fetches German
    assert fetched == ["deu", "eng"]
    monkeypatch.setattr(slides, "fetch_tessdata", lambda code, folder: False)  # offline
    assert slides.ocr_langs("tesseract", "de") == ("eng", None)
    monkeypatch.setattr(slides, "installed_langs", lambda cmd: set())
    assert slides.ocr_langs("tesseract", "de") == ("", None)


def test_tessdata_is_checked_before_use(monkeypatch, tmp_path):
    import hashlib

    import httpx
    good = b"traineddata"
    monkeypatch.setitem(slides.TESSDATA_SHA256, "hun", hashlib.sha256(good).hexdigest())
    answers = [b"tampered", good]
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, content=answers.pop(0)))
    assert slides.fetch_tessdata("hun", str(tmp_path)) is False and not (tmp_path / "hun.traineddata").exists()
    assert slides.fetch_tessdata("hun", str(tmp_path)) is True
    assert (tmp_path / "hun.traineddata").read_bytes() == good
    assert slides.fetch_tessdata("hun", str(tmp_path)) is True  # already there: no download
    assert slides.fetch_tessdata("xyz", str(tmp_path)) is False
    assert set(slides.TESSDATA_SHA256) >= set(slides.TESSERACT_LANGS.values())


def test_the_windows_package_carries_tesseract(cfg, monkeypatch, tmp_path):
    import sys
    exe = tmp_path / "tesseract" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "python"))
    monkeypatch.setattr(slides.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(slides.os, "name", "nt")
    assert slides.tesseract(cfg) == str(exe)
