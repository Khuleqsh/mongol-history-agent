from types import SimpleNamespace as NS

import pytest

from mha.agent import TOOLS
from mha.llm import AnthropicLLM, GeminiLLM, QuotaExceeded, ToolCall, make_llm, to_gemini_declarations, to_gemini_schema


# ------------------------------------------------------------ schema хөрвүүлэлт
def test_gemini_schema_strips_unsupported_keys():
    decls = to_gemini_declarations(TOOLS)
    blob = str(decls)
    assert "default" not in blob and "minItems" not in blob
    assert decls[0]["name"] == "search_history"
    assert decls[0]["parameters"]["required"] == ["queries"]
    assert to_gemini_schema({"a": [{"default": 1, "b": 2}]}) == {"a": [{"b": 2}]}


def test_make_llm_unknown_provider():
    with pytest.raises(ValueError):
        make_llm("nope")


# ------------------------------------------------------------------ Anthropic
class FakeAnthropic:
    def __init__(self, script):
        self.script, self.calls, self.messages = list(script), [], self

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def test_anthropic_tool_roundtrip():
    tool_resp = NS(content=[NS(type="tool_use", id="t1", name="search_history", input={"queries": ["q"]})])
    final = NS(content=[NS(type="text", text="Хариу [S1]")])
    llm = AnthropicLLM("m", client=FakeAnthropic([tool_resp, final, NS(content=[NS(type="text", text="done")])]))
    conv = llm.start("sys", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "асуулт")
    step = llm.step(conv, TOOLS)
    assert step.tool_calls == [ToolCall("t1", "search_history", {"queries": ["q"]})]
    llm.add_tool_results(conv, [(step.tool_calls[0], "out")])
    assert conv["messages"][-1]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "out"}
    assert llm.step(conv, TOOLS).text == "Хариу [S1]"
    assert llm.complete("p") == "done"


# --------------------------------------------------------------------- Gemini
genai_types = pytest.importorskip("google.genai.types")
T = genai_types


def resp(*parts):
    return T.GenerateContentResponse(candidates=[T.Candidate(content=T.Content(role="model", parts=list(parts)))])


class FakeGenai:
    def __init__(self, script):
        self.script, self.calls, self.models = list(script), [], self

    def generate_content(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class APIErr(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def gem(script, **kw):
    return GeminiLLM("gemini-test", client=FakeGenai(script), sleep=lambda s: None, **kw)


def test_gemini_tool_roundtrip_with_real_types():
    fc = T.Part(function_call=T.FunctionCall(name="search_history", args={"queries": ["Тэмүжин"], "year_from": 1200}))
    llm = gem([resp(fc), resp(T.Part(text="1206 онд [S1]"))])
    conv = llm.start("sys", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "асуулт")
    assert [c.role for c in conv["contents"]] == ["user", "model", "user"]

    step = llm.step(conv, TOOLS)
    assert step.tool_calls[0].name == "search_history"
    assert step.tool_calls[0].args == {"queries": ["Тэмүжин"], "year_from": 1200}
    assert step.tool_calls[0].id == "call_0"  # Gemini id өгөөгүй үед үүсгэнэ
    assert conv["contents"][-1].role == "model"  # загварын хариулт (thought signature-тай нь) хадгалагдсан

    llm.add_tool_results(conv, [(step.tool_calls[0], "олдсон текст")])
    fr = conv["contents"][-1].parts[0].function_response
    assert conv["contents"][-1].role == "user" and fr.name == "search_history" and fr.response == {"result": "олдсон текст"}

    assert llm.step(conv, TOOLS).text == "1206 онд [S1]"
    cfg = llm.client.calls[0]["config"]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.tools[0].function_declarations[0].name == "search_history"
    assert cfg.system_instruction == "sys"


def test_gemini_ignores_thought_parts_and_empty_candidates():
    llm = gem([resp(T.Part(text="сэтгэж байна", thought=True), T.Part(text="Хариу")), T.GenerateContentResponse(candidates=[])])
    conv = llm.start("s", [], "q")
    assert llm.step(conv, TOOLS).text == "Хариу"
    assert llm.step(conv, TOOLS).text == ""


def test_gemini_complete():
    assert gem([resp(T.Part(text="JSON"))]).complete("prompt") == "JSON"


def test_gemini_retries_transient_errors():
    sleeps = []
    llm = GeminiLLM("m", client=FakeGenai([APIErr(503, "overloaded"), APIErr(429, "rate limit per minute"), resp(T.Part(text="ok"))]), sleep=sleeps.append)
    assert llm.complete("p") == "ok"
    assert sleeps == [5, 10]


def test_gemini_daily_quota_raises_friendly_error():
    llm = gem([APIErr(429, "RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier limit: 20")])
    with pytest.raises(QuotaExceeded) as e:
        llm.complete("p")
    assert "gemini-test" in str(e.value)


def test_gemini_other_errors_propagate_and_retries_are_bounded():
    with pytest.raises(APIErr):
        gem([APIErr(400, "bad request")]).complete("p")
    with pytest.raises(APIErr):
        gem([APIErr(503, "x")] * 4, retries=4).complete("p")
