from types import SimpleNamespace as NS

from mha import agent as agent_mod
from mha.agent import Agent


def text(t):
    return NS(content=[NS(type="text", text=t)], stop_reason="end_turn")


def tool(name, inp, tid="t1"):
    return NS(content=[NS(type="tool_use", id=tid, name=name, input=inp)], stop_reason="tool_use")


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, queries, top_k=6, year_from=None, year_to=None):
        self.calls.append((queries, year_from, year_to))
        return [
            {"id": "src:0", "text": "1206 онд Тэмүжин хаан өргөмжлөгдсөн.", "title": "Шастир", "author": "A", "year": "1240",
             "lang": "mn", "kind": "primary", "section": "Нэг", "chunk": 0}
        ]


def make(script, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "GAPS", tmp_path / "gaps.jsonl")
    s = FakeSearcher()
    return Agent(client=FakeClient(script), searcher=s, prompt="v2"), s


def test_search_then_answer(tmp_path, monkeypatch):
    a, s = make([tool("search_history", {"queries": ["q"], "year_from": 1200}), text("1206 онд хаан болсон [S1].")], tmp_path, monkeypatch)
    r = a.ask("Хэзээ?")
    assert r["searched"] and not r["gap"]
    assert list(r["sources"]) == ["S1"] and r["sources"]["S1"]["section"] == "Нэг"
    assert r["invalid_citations"] == []
    assert s.calls == [(["q"], 1200, None)]


def test_invalid_citation_detected(tmp_path, monkeypatch):
    a, _ = make([tool("search_history", {"queries": ["q"]}), text("Юм [S1]. Өөр юм [S9].")], tmp_path, monkeypatch)
    r = a.ask("?")
    assert r["invalid_citations"] == ["S9"]


def test_gap_is_logged(tmp_path, monkeypatch):
    a, _ = make([tool("search_history", {"queries": ["q"]}), tool("report_gap", {"topic": "Зуд"}, "t2"), text("Архивт алга.")], tmp_path, monkeypatch)
    r = a.ask("Зуд?")
    assert r["gap"]
    assert "Зуд" in (tmp_path / "gaps.jsonl").read_text(encoding="utf-8")


def test_verify_revises_unsupported(tmp_path, monkeypatch):
    script = [
        tool("search_history", {"queries": ["q"]}),
        text("1207 онд хаан болсон [S1]."),
        text('{"unsupported": [{"claim": "1207", "reason": "passage says 1206"}]}'),
        text("1206 онд хаан болсон [S1]."),
    ]
    a, _ = make(script, tmp_path, monkeypatch)
    r = a.ask("?", verify=True)
    assert "1206" in r["answer"] and len(r["unsupported_removed"]) == 1


def test_level_and_history_reach_model(tmp_path, monkeypatch):
    a, _ = make([text("Сайн уу")], tmp_path, monkeypatch)
    r = a.ask("Сайн уу", level="student", history=[{"role": "user", "content": "өмнөх"}, {"role": "assistant", "content": "хариу"}])
    call = a.client.calls[0]
    assert "student" in call["system"] and call["messages"][2] == {"role": "user", "content": "Сайн уу"}
    assert r["searched"] is False


def test_max_turns_fallback(tmp_path, monkeypatch):
    a, _ = make([tool("search_history", {"queries": ["q"]}, f"t{i}") for i in range(3)], tmp_path, monkeypatch)
    r = a.ask("?", max_turns=3)
    assert "дуусгаж чадсангүй" in r["answer"]
