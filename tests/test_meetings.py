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
