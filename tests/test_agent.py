from mha import agent as agent_mod
from mha.agent import Agent
from mha.llm import Step, ToolCall


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, steps, completions=()):
        self.steps, self.completions = list(steps), list(completions)
        self.convs, self.prompts = [], []

    def start(self, system, history, question):
        conv = {"system": system, "history": history, "question": question, "results": []}
        self.convs.append(conv)
        return conv

    def step(self, conv, tools, max_tokens=2500):
        return self.steps.pop(0)

    def add_tool_results(self, conv, results):
        conv["results"].append(results)

    def complete(self, prompt, max_tokens=2000):
        self.prompts.append(prompt)
        return self.completions.pop(0)


def call(name, args, cid="t1"):
    return Step("", [ToolCall(cid, name, args)])


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, queries, top_k=6, year_from=None, year_to=None):
        self.calls.append((queries, year_from, year_to))
        return [
            {"id": "src:0", "text": "1206 онд Тэмүжин хаан өргөмжлөгдсөн.", "title": "Шастир", "author": "A", "year": "1240",
             "lang": "mn", "kind": "primary", "section": "Нэг", "chunk": 0}
        ]


def make(steps, tmp_path, monkeypatch, completions=()):
    monkeypatch.setattr(agent_mod, "GAPS", tmp_path / "gaps.jsonl")
    s = FakeSearcher()
    return Agent(llm=FakeLLM(steps, completions), searcher=s, prompt="v2"), s


def test_search_then_answer(tmp_path, monkeypatch):
    a, s = make([call("search_history", {"queries": ["q"], "year_from": 1200}), Step("1206 онд хаан болсон [S1].")], tmp_path, monkeypatch)
    r = a.ask("Хэзээ?")
    assert r["searched"] and not r["gap"]
    assert list(r["sources"]) == ["S1"] and r["sources"]["S1"]["section"] == "Нэг"
    assert r["invalid_citations"] == []
    assert s.calls == [(["q"], 1200, None)]
    # хайлтын үр дүн эх сурвалжийн шошготойгоор загварт буцсан
    sent = a.llm.convs[0]["results"][0][0][1]
    assert sent.startswith("[S1] Шастир") and "1206" in sent


def test_invalid_citation_detected(tmp_path, monkeypatch):
    a, _ = make([call("search_history", {"queries": ["q"]}), Step("Юм [S1]. Өөр юм [S9].")], tmp_path, monkeypatch)
    assert a.ask("?")["invalid_citations"] == ["S9"]


def test_gap_is_logged(tmp_path, monkeypatch):
    a, _ = make([call("search_history", {"queries": ["q"]}), call("report_gap", {"topic": "Зуд"}, "t2"), Step("Архивт алга.")], tmp_path, monkeypatch)
    r = a.ask("Зуд?")
    assert r["gap"]
    assert "Зуд" in (tmp_path / "gaps.jsonl").read_text(encoding="utf-8")


def test_verify_revises_unsupported(tmp_path, monkeypatch):
    a, _ = make(
        [call("search_history", {"queries": ["q"]}), Step("1207 онд хаан болсон [S1].")],
        tmp_path, monkeypatch,
        completions=['{"unsupported": [{"claim": "1207", "reason": "passage says 1206"}]}', "1206 онд хаан болсон [S1]."],
    )
    r = a.ask("?", verify=True)
    assert "1206" in r["answer"] and len(r["unsupported_removed"]) == 1
    assert "1206 онд Тэмүжин" in a.llm.prompts[0]  # шалгагч эх сурвалжийн бодит текстийг харсан


def test_verify_skipped_when_nothing_to_check(tmp_path, monkeypatch):
    a, _ = make([Step("Сайн уу")], tmp_path, monkeypatch)
    assert a.ask("Сайн уу", verify=True)["unsupported_removed"] == []
    assert a.llm.prompts == []


def test_level_and_history_reach_model(tmp_path, monkeypatch):
    a, _ = make([Step("Сайн уу")], tmp_path, monkeypatch)
    hist = [{"role": "user", "content": "өмнөх"}, {"role": "assistant", "content": "хариу"}]
    r = a.ask("Сайн уу", level="student", history=hist)
    conv = a.llm.convs[0]
    assert "student" in conv["system"] and conv["history"] == hist
    assert r["searched"] is False


def test_max_turns_fallback_and_empty_answer(tmp_path, monkeypatch):
    a, _ = make([call("search_history", {"queries": ["q"]}, f"t{i}") for i in range(3)], tmp_path, monkeypatch)
    assert "дуусгаж чадсангүй" in a.ask("?", max_turns=3)["answer"]
    b, _ = make([Step("")], tmp_path, monkeypatch)
    assert "хариулт өгсөнгүй" in b.ask("?")["answer"]
