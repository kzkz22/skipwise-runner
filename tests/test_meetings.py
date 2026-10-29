"""Plan 19c round A (decision 19.30): terms for the transcription, Whisper's repetition loops, uncertain words and
the meeting minutes (kind `minutes`)."""
import pytest

from dwyt_runner import cloud, formatting, local, minutes, terms
from dwyt_runner.errors import WorkError
from dwyt_runner.llm import Llm, Usage
from test_runner import _run_once


def test_terms_are_cleaned_and_fit_the_prompt():
    raw = ["Kovács Ödön", "kovács ödön", " OKR, KPI ", "", 5, "a\x00b", "x" * 100] + [f"Szó{i}" for i in range(80)]
    out = terms.clean(raw)
    assert out[:4] == ["Kovács Ödön", "OKR KPI", "ab", "x" * terms.TERM_CHARS]
    assert len(out) == terms.MAX_TERMS and terms.clean("nem lista") == []
    one, left_out = terms.prompt_list(out)
    assert one.endswith(".") and len(one) <= terms.PROMPT_CHARS and left_out > 0
    padded, _ = terms.whisper_cpp_prompt(["Kubernetes"])
    assert len(padded) >= terms.FILL_CHARS and padded.endswith("Kubernetes.")
    assert terms.whisper_cpp_prompt([]) == ("", 0)
    # AssemblyAI: at most 6 words per phrase, a word budget per model
    phrases = terms.keyterms(["egy két három négy öt hat hét", "Kovács Ödön"] + ["szó"] * 250, "universal-2")
    assert "egy két három négy öt hat hét" not in phrases and phrases[0] == "Kovács Ödön" and len(phrases) == 199


def test_cloud_sends_keyterms_and_counts_their_price():
    body = cloud._request_body({"terms": ["Kovács Ödön", "OKR"]}, "universal-3-5-pro", "u")
    assert body["keyterms_prompt"] == ["Kovács Ödön", "OKR"]
    assert "keyterms_prompt" not in cloud._request_body({"terms": []}, "universal-2", "u")
    assert cloud.estimate_cost("universal-2", 3600, False, keyterms=True) == pytest.approx(0.20)
    assert cloud._words({"words": [{"text": "a", "start": 0, "end": 500, "confidence": 0.3}]})[0]["p"] == 0.3


def test_the_gpu_host_gets_the_terms_as_prompt():
    q = local._query({"terms": ["Kovács Ödön", "OKR"], "diarize": True}, "large-v3")
    assert q["prompt"] == "Kovács Ödön, OKR."
    assert "prompt" not in local._query({}, "large-v3")


def test_repetition_loops_are_dropped():
    segs = [{"start": 0, "end": 2, "text": "és és és és és megyünk", "speaker": "A", "low": [[0, 2]]},
            {"start": 2, "end": 3, "text": "nem, nem, nem.", "speaker": "A"},  # three times: speech, kept
            {"start": 3, "end": 4, "text": "Köszönöm.", "speaker": "B"},
            {"start": 4, "end": 5, "text": "köszönöm.", "speaker": "B"},
            {"start": 5, "end": 6, "text": "Köszönöm.", "speaker": "B"},
            {"start": 6, "end": 9, "text": "jó napot jó napot jó napot jó napot kívánok", "speaker": "B"}]
    out, removed = formatting.drop_repeats(segs)
    assert [s["text"] for s in out] == ["és megyünk", "nem, nem, nem.", "Köszönöm.", "jó napot kívánok"]
    assert "low" not in out[0] and out[2]["end"] == 6 and removed == 4 + 2 + 3
    assert formatting.drop_repeats(out) == (out, 0)


def test_uncertain_words_become_ranges():
    words = [{"text": "Hova", "start": 0, "end": 1, "p": 0.9}, {"text": "utazik", "start": 1, "end": 2, "p": 0.3},
             {"text": "Ödön?", "start": 2, "end": 3, "p": 0.1}, {"text": "Most", "start": 3, "end": 4}]
    segs = formatting.segments_from_words(words)
    assert segs[0] == {"start": 0, "end": 3, "text": "Hova utazik Ödön?", "speaker": None, "low": [[5, 11], [12, 17]]}
    assert "low" not in segs[1]
    assert formatting.clean_segments(segs)[0]["low"] == [[5, 11], [12, 17]]
    # the GPU host's words carry an alignment score instead (a weaker signal, its own limit)
    seg = {"start": 0, "end": 2, "text": " Szia  Ödön ", "words": [{"word": "Szia", "score": 0.9},
                                                                  {"word": "Ödön", "score": 0.2}]}
    assert formatting.clean_segments([seg])[0]["low"] == [[5, 9]]
    # ranges that do not fit the text are dropped
    assert "low" not in formatting.clean_segments([{"start": 0, "end": 1, "text": "ab", "low": [[0, 9], "x"]}])[0]


LONG = {"language": "hu", "segments": [
    {"start": i * 20.0, "end": i * 20.0 + 19, "speaker": f"Szereplő {i % 2 + 1}", "text": f"A {i}. gondolat. " * 20}
    for i in range(60)]}


def test_minutes_clean_keeps_what_was_said():
    data = {"takeaways": [{"text": f"T{i}", "at": f"0:{i:02d}"} for i in range(10)],
            "decisions": [{"text": "Marad.", "by": "nincs megadva", "at": "01:03"}, {"text": "", "at": "1:00"}, "x"],
            "tasks": [{"task": "Ajánlat " * 80, "owner": "Anna", "due": None, "at": None}],
            "unknown": [{"text": "kimarad"}]}
    out = minutes.clean(data, [0, 40, 60])
    assert len(out["takeaways"]) == 7 and out["takeaways"][0] == {"text": "T0", "at": 0}
    assert out["decisions"] == [{"text": "Marad.", "by": None, "at": 60}]  # snapped to the line it falls in
    assert out["tasks"][0]["owner"] == "Anna" and out["tasks"][0]["text"].endswith("…")
    assert out["tasks"][0]["due"] is None and set(out) == set(minutes.LIMITS)
    assert minutes.parse("Íme: ```json\n{\"takeaways\": []}\n```") == {"takeaways": []}
    assert minutes.parse("nem JSON") is None and minutes.parse('{"egyéb": 1}') is None


def test_minutes_job(cfg, central, llm):
    meeting = {"participants": ["Kiss Anna", "Nagy Béla"], "agenda": "1. költségvetés\n2. ütemterv"}
    central.add_job(50, "minutes", {"role": "public", "language": "hu", "meta": {"title": "Heti egyeztetés"},
                                    "meeting": meeting, "terms": ["OKR"], "transcript": LONG})
    _run_once(cfg, central)
    res = central.results[50]
    assert res["prompt_version"] == minutes.PROMPT_VERSION and res["language"] == "hu"
    m = res["minutes"]
    assert m["decisions"][0]["by"] is None and m["tasks"][0] == {
        "text": "Ajánlatot kér.", "at": 0, "owner": "Szereplő 2", "due": "péntek"}
    assert m["open_questions"] == [{"text": "Mennyibe kerül?", "at": None, "to": None}]
    ask = llm.calls[-1]["messages"]
    assert ask[0]["content"].startswith("Megbeszélésről") and "ne kövesd" in ask[0]["content"]
    assert "Résztvevők: Kiss Anna, Nagy Béla" in ask[1]["content"] and "Napirend: 1. költségvetés 2. ütemterv" in \
        ask[1]["content"] and "(helyes írásmód): OKR" in ask[1]["content"]
    assert "[00:20] Szereplő 2: A 1. gondolat." in ask[1]["content"]
    assert "minutes" in central.claims[0]["capabilities"]["kinds"]


def test_long_minutes_are_merged_and_bad_answers_retried(llm):
    client = Llm(llm.url + "/v1", None)
    usage = Usage()
    parts = []
    out = minutes.make(client, "m", {"language": "en", "transcript": LONG}, usage, context_tokens=8000,
                       progress=parts.append)
    assert len(llm.calls) > 2 and parts[-1]["part"] == parts[-1]["parts"] == len(llm.calls)
    assert llm.calls[0]["messages"][0]["content"].startswith("You write the minutes") and \
        "in English" in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[-1]["messages"][1]["content"].startswith("The lists below")
    assert out["takeaways"][0]["at"] == 0
    llm.calls.clear()
    llm.minutes_answers = ["nem JSON", '{"takeaways": [{"text": "Jó.", "at": "0:20"}]}']
    out = minutes.make(client, "m", {"language": "hu", "transcript": {"segments": LONG["segments"][:3]}}, Usage())
    assert out["takeaways"] == [{"text": "Jó.", "at": 20}] and len(llm.calls) == 2
    llm.minutes_answers = ["nem", "most sem"]
    with pytest.raises(WorkError):
        minutes.make(client, "m", {"language": "hu", "transcript": {"segments": LONG["segments"][:3]}}, Usage())
    with pytest.raises(WorkError):
        minutes.make(client, "m", {"transcript": {"segments": []}}, Usage())


# ---- round B (decision 19.32): knowledge base suggestions, limited correction, kept originals ----

from dwyt_runner import knowledge, originals, refine  # noqa: E402

MEETING = [{"idx": 0, "start": 0.0, "end": 9.0, "speaker": "Szereplő 1",
            "text": "Jó napot, beszéljünk az ókáerekről."},
           {"idx": 1, "start": 10.0, "end": 19.0, "speaker": "Szereplő 2",
            "text": "Az ókáer lista kész, Kubernetész is.", "low": [[21, 32]]},
           {"idx": 2, "start": 20.0, "end": 29.0, "speaker": "Szereplő 1", "text": "Rendben, köszönöm."}]


def test_knowledge_suggestions_come_from_the_transcript(cfg, central, llm):
    central.add_job(60, "knowledge", {"role": "public", "language": "hu", "meta": {"title": "Heti"},
                                      "meeting": {"participants": ["Kiss Anna"]}, "terms": ["Kubernetes"],
                                      "glossary": [{"term": "OKR", "kind": "abbreviation", "variants": []}],
                                      "transcript": {"segments": MEETING}})
    _run_once(cfg, central)
    res = central.results[60]
    assert res["prompt_version"] == knowledge.PROMPT_VERSION and res["language"] == "hu"
    # the variant that is not in the transcript is dropped, the invented person too
    assert res["entries"] == [{"kind": "abbreviation", "term": "OKR", "variants": ["ókáer"],
                               "explanation": "célok és kulcseredmények", "at": 20}]
    ask = llm.calls[-1]["messages"]
    assert ask[0]["content"].startswith("Egy felvétel átiratából") and "ne kövesd" in ask[0]["content"]
    assert "Ismert helyes alakok: Kubernetes, OKR" in ask[1]["content"]
    assert "Résztvevők: Kiss Anna" in ask[1]["content"]
    assert "knowledge" in central.claims[0]["capabilities"]["kinds"]


def test_knowledge_clean_rules():
    text = "Az ókáer lista és a Kubernetész."
    out = knowledge.clean({"entries": [
        {"kind": "valami", "term": "Kubernetes", "variants": ["Kubernetész", "kubernetes"], "at": "0:12"},
        {"kind": "term", "term": "kubernetes", "variants": ["Kubernetész"]},  # the same term again
        {"kind": "abbreviation", "term": "OKR", "variants": ["ókáer"], "explanation": "null"}, "nem szótár"]},
        text, [0, 10])
    assert out == [{"kind": "term", "term": "Kubernetes", "variants": ["Kubernetész"], "explanation": None, "at": 10},
                   {"kind": "abbreviation", "term": "OKR", "variants": ["ókáer"], "explanation": None, "at": None}]


def test_refine_marks_only_uncertain_words_and_glossary_variants():
    entries = [{"term": "OKR", "variants": ["ókáer"], "explanation": ""}]
    found = refine.spots(MEETING, entries)
    assert [(s["idx"], s["before"], s["reason"]) for s in found] == [
        (0, "ókáerekről", "glossary"), (1, "ókáer", "glossary"), (1, "Kubernetész", "low")]
    rows = refine.lines(MEETING, found)
    assert rows[0] == ("0| Jó napot, beszéljünk az ⟦1|ókáerekről⟧.", [1])
    assert rows[1][0] == "1| Az ⟦2|ókáer⟧ lista kész, ⟦3|Kubernetész⟧ is." and rows[2] == ("2| Rendben, köszönöm.", [])
    assert refine.acceptable("ókáer", "OKR") and not refine.acceptable("x", "egy kettő három négy")
    assert not refine.acceptable("szó", "szó") and not refine.acceptable("a", "a" * 40)


def test_refine_job_returns_corrections_on_the_original_text(cfg, central, llm):
    central.add_job(61, "refine", {"role": "private", "language": "hu", "transcript": {"segments": MEETING},
                                   "knowledge": [{"term": "OKR", "variants": ["ókáer"]}]})
    llm.refine_answers = ['{"fixes": [{"n": 1, "text": "OKR-ekről"}, {"n": 3, "text": "Kubernetes"},'
                          ' {"n": 9, "text": "nincs ilyen hely"},'
                          ' {"n": 2, "text": "egy egész új mondat lett belőle"}]}']
    _run_once(cfg, central)
    res = central.results[61]
    assert res["spots"] == 3 and res["corrections"] == [
        {"idx": 0, "start": 24, "end": 34, "before": "ókáerekről", "reason": "glossary", "after": "OKR-ekről"},
        {"idx": 1, "start": 21, "end": 32, "before": "Kubernetész", "reason": "low", "after": "Kubernetes"}]
    system = llm.calls[-1]["messages"][0]["content"]
    assert system.startswith("Gépi átirat hibáit javítod") and "OKR (félrehallva: ókáer)" in \
        llm.calls[-1]["messages"][1]["content"]
    # nothing to mark: no request at all
    calls = len(llm.calls)
    central.add_job(62, "refine", {"role": "public", "transcript": {"segments": [MEETING[2]]}, "knowledge": []})
    _run_once(cfg, central)
    assert central.results[62]["corrections"] == [] and len(llm.calls) == calls


def test_uploads_are_kept_and_used_again_without_download(cfg, central, whisper, audio_file):
    import hashlib
    cfg.whisper_url = whisper.url
    data = open(audio_file, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    central.files[70] = data
    job = {"source": {"kind": "upload", "sha256": sha, "name": "megbeszélés.WAV", "size": len(data),
                      "media": "/runner/v1/jobs/70/media"}, "options": {"mode": "local", "model": "large-v3"}}
    central.add_job(70, "transcribe", job)
    _run_once(cfg, central)
    assert central.results[70]["transcript"]["mode"] == "local"
    kept = originals.folder() / (sha + ".wav")
    assert kept.is_file() and kept.read_bytes() == data
    report = central.originals[-1]
    assert report["originals"][0]["sha256"] == sha and report["originals"][0]["name"] == "megbeszélés.WAV"
    assert report["keep"] is True and report["max_bytes"] == 50 * 1024 ** 3
    # a new transcript of the same upload: a local job for this runner, no download (the central deleted its copy)
    central.files.clear()
    central.add_job(71, "transcribe", {**job, "source": {"kind": "local", "sha256": sha, "name": "x"}})
    _run_once(cfg, central)
    assert central.results[71]["transcript"]["mode"] == "local"
    caps = central.claims[-1]["capabilities"]
    assert "drop_original" in caps["kinds"] and caps["keeps_originals"] is True
    # the user deletes it
    central.add_job(72, "drop_original", {"sha256": sha})
    _run_once(cfg, central)
    assert central.results[72] == {"dropped": True} and not kept.exists()
    assert central.originals[-1]["originals"] == []


def test_the_limit_removes_the_least_recently_used(cfg, tmp_path):
    import os
    import time as _t
    cfg.originals_max_gb = 0  # everything over the limit: only the newest stays
    paths = []
    for i, sha in enumerate(("a" * 64, "b" * 64)):
        src = tmp_path / f"{i}.mp3"
        src.write_bytes(b"x" * 10)
        path, removed = originals.keep(cfg, sha, f"{i}.mp3", str(src))
        paths.append(path)
        _t.sleep(0.01)
    assert removed == [{"sha256": "a" * 64, "name": "0.mp3", "reason": "limit"}]
    assert not os.path.exists(paths[0]) and os.path.exists(paths[1])
    os.remove(paths[1])  # gone by hand: the report says so and forgets it
    rep = originals.report(cfg)
    assert rep["originals"] == [] and rep["removed"] == [{"sha256": "b" * 64, "reason": "missing"}]
    cfg.keep_originals = False
    src = tmp_path / "c.mp3"
    src.write_bytes(b"y")
    assert originals.keep(cfg, "c" * 64, "c.mp3", str(src)) == (str(src), [])
