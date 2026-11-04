"""Answers that must be JSON (decision 19.35): JSON mode in the request, tolerant reading, a second try, a part in
two halves, and a clear 'later' failure with the beginning of the answer."""
import json

import pytest

from dwyt_runner import structured
from dwyt_runner.llm import Answer
from test_meetings import MEETING
from test_runner import _run_once

GOOD = {"entries": [{"kind": "abbreviation", "term": "OKR", "variants": ["ókáer"], "explanation": "célok",
                     "at": "00:20"}]}


@pytest.mark.parametrize("text", [
    '```json\n{"entries": [{"term": "OKR"}]}\n```',
    'Íme a szójegyzék: {"entries": [{"term": "OKR"}]} Remélem, segít. {megjegyzés}',
    '{"entries": [{"term": "OKR"},],}',                          # trailing commas
    '{“entries”: [{“term”: “OKR”}]}',                              # curly quotes as JSON quotes
    '<think>{"nem": 1}</think>{"x": {"y": 1}} {"entries": [{"term": "OKR"}]}',  # the right object later
])
def test_answers_are_read_tolerantly(text):
    assert structured.parse_object(text, ("entries",))["entries"][0]["term"] == "OKR"


def test_a_cut_off_answer_keeps_its_finished_items():
    text = '{"entries": [{"term": "OKR", "at": 1}, {"term": "Kubernetes"}, {"term": "Kovács Ö'
    assert structured.parse_object(text, ("entries",)) == {"entries": [{"term": "OKR", "at": 1},
                                                                      {"term": "Kubernetes"}]}
    assert structured.parse_object("nincs itt semmi", ("entries",)) is None
    assert structured.first_object('Válasz: {"main": 3, "side": []}') == {"main": 3, "side": []}


def _knowledge_job(central, job_id):
    central.add_job(job_id, "knowledge", {"role": "public", "language": "hu", "meta": {"title": "Heti"},
                                          "transcript": {"segments": MEETING}})


def test_the_request_asks_for_json_and_falls_back_when_refused(cfg, central, llm):
    _knowledge_job(central, 61)
    _run_once(cfg, central)
    assert central.results[61]["entries"] and llm.calls[-1]["response_format"] == {"type": "json_object"}
    llm.reset()
    llm.no_json_mode = True  # an older server: asked again without the field, and not asked with it again
    _knowledge_job(central, 62)
    _knowledge_job(central, 63)
    _run_once(cfg, central)
    _run_once(cfg, central)
    assert central.results[62]["entries"] and central.results[63]["entries"]
    assert [("response_format" in c) for c in llm.calls] == [True, False, False]


def test_an_unreadable_answer_is_tried_again_then_handed_back(cfg, central, llm):
    llm.knowledge_answers = ["Sajnos most nem tudok JSON-t írni.", json.dumps(GOOD, ensure_ascii=False)]
    _knowledge_job(central, 64)
    _run_once(cfg, central)
    assert central.results[64]["entries"][0]["term"] == "OKR"  # the second try was good
    llm.knowledge_answers = ["Első rossz válasz", "Második rossz válasz"]
    _knowledge_job(central, 65)
    _run_once(cfg, central)
    failure = central.failures[65]
    assert failure["retry"] == "later" and "szójegyzék" in failure["error"]
    assert "A válasz eleje: „Második rossz válasz”" in failure["error"]


def test_a_garbled_long_part_is_asked_again_in_halves():
    calls = []

    def run(lines):
        calls.append(len(lines))
        if len(lines) == 40:
            raise structured.Unreadable("szójegyzék", "x")
        return len(lines)
    assert structured.ask_split(list(range(40)), run) == [20, 20] and calls == [40, 20, 20]

    def always_bad(lines):
        raise structured.Unreadable("szójegyzék", "x")
    with pytest.raises(structured.Unreadable):  # a short part is not split
        structured.ask_split(list(range(10)), always_bad)


def test_ask_counts_every_try():
    class FakeLlm:
        def __init__(self):
            self.answers = ["nem JSON", '{"fixes": []}']

        def complete(self, model, messages, temperature=None, json_mode=False):
            assert json_mode
            return Answer(text=self.answers.pop(0), model=model, prompt_tokens=10, completion_tokens=2, seconds=0)

    class Usage:
        n = 0

        def add(self, answer):
            self.n += 1
            return answer
    usage = Usage()
    assert structured.ask(FakeLlm(), "m", [], usage, True, "javítás",
                          lambda t: structured.parse_object(t, ("fixes",))) == {"fixes": []}
    assert usage.n == 2


def test_an_empty_json_mode_answer_is_asked_again_without_it(cfg, central, llm):
    # a thinking model under a JSON grammar may give nothing but its thinking: once more without the grammar
    llm.knowledge_answers = ["<think>hosszú gondolkodás…</think>", json.dumps(GOOD, ensure_ascii=False)]
    _knowledge_job(central, 66)
    _run_once(cfg, central)
    assert central.results[66]["entries"][0]["term"] == "OKR"
    assert [("response_format" in c) for c in llm.calls] == [True, False]
    _knowledge_job(central, 67)  # the same model is not asked in JSON mode again
    _run_once(cfg, central)
    assert "response_format" not in llm.calls[-1]
