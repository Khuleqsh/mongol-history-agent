"""LLM provider давхарга. Агент зөвхөн энэ neutral интерфейсээр ярина:

    conv = llm.start(system, history, question)
    step = llm.step(conv, TOOLS)           # Step(text, tool_calls)
    llm.add_tool_results(conv, [(call, output), ...])
    llm.complete(prompt)                   # tool-гүй нэг удаагийн дуудлага (verify/revise)

Tool schema нь Anthropic хэлбэртэй (name, description, input_schema); Gemini-д хөрвүүлнэ.
"""
import time
from dataclasses import dataclass, field
from typing import Any

from .config import MODEL, PROVIDER


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Step:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class QuotaExceeded(RuntimeError):
    """Өдрийн (эсвэл тухайн загварт квот байхгүй) хязгаар дууссан үед."""


# ---------------------------------------------------------------- Anthropic
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def start(self, system: str, history: list[dict], question: str) -> dict:
        return {"system": system, "messages": list(history) + [{"role": "user", "content": question}]}

    def step(self, conv: dict, tools: list[dict], max_tokens: int = 2500) -> Step:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=conv["system"], tools=tools, messages=conv["messages"]
        )
        conv["messages"].append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return Step(text, calls)

    def add_tool_results(self, conv: dict, results: list[tuple[ToolCall, str]]) -> None:
        conv["messages"].append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": c.id, "content": out} for c, out in results],
            }
        )

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in resp.content if b.type == "text")


# ------------------------------------------------------------------- Gemini
_UNSUPPORTED_SCHEMA_KEYS = {"default", "minItems", "maxItems"}


def to_gemini_schema(schema: Any) -> Any:
    """Gemini-ийн OpenAPI дэд олонлогт байхгүй түлхүүрүүдийг (default, minItems...) хасна."""
    if isinstance(schema, dict):
        return {k: to_gemini_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(schema, list):
        return [to_gemini_schema(v) for v in schema]
    return schema


def to_gemini_declarations(tools: list[dict]) -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "parameters": to_gemini_schema(t["input_schema"])}
        for t in tools
    ]


def _is_daily_quota(msg: str) -> bool:
    return "PerDay" in msg or "limit: 0" in msg


class GeminiLLM:
    name = "gemini"

    def __init__(self, model: str, client=None, retries: int = 4, sleep=time.sleep) -> None:
        if client is None:
            from google import genai

            client = genai.Client()  # GEMINI_API_KEY (эсвэл GOOGLE_API_KEY)-г орчноос уншина
        self.client, self.model, self.retries, self.sleep = client, model, retries, sleep

    # -- туслах
    def _call(self, **kw):
        for attempt in range(self.retries):
            try:
                return self.client.models.generate_content(model=self.model, **kw)
            except Exception as e:  # noqa: BLE001 - SDK-ийн APIError-ийг код-оор нь ялгана
                code, msg = getattr(e, "code", None), str(e)
                if code == 429 and _is_daily_quota(msg):
                    raise QuotaExceeded(
                        f"'{self.model}' загварын үнэгүй квот дууссан эсвэл энэ түлхүүрт нээгдээгүй байна. "
                        "Маргааш дахин оролдох, эсвэл MHA_MODEL-оор өөр (жишээ нь Flash-Lite) загвар сонгоно уу."
                    ) from e
                if code in (429, 503) and attempt < self.retries - 1:
                    self.sleep(min(5 * 2**attempt, 60))
                    continue
                raise

    @staticmethod
    def _parse(resp) -> Step:
        cands = getattr(resp, "candidates", None) or []
        if not cands or cands[0].content is None:
            return Step("")
        text, calls = [], []
        for i, part in enumerate(cands[0].content.parts or []):
            fc = getattr(part, "function_call", None)
            if fc:
                calls.append(ToolCall(fc.id or f"call_{i}", fc.name, dict(fc.args or {})))
            elif part.text and not getattr(part, "thought", False):
                text.append(part.text)
        return Step("".join(text), calls)

    # -- интерфейс
    def start(self, system: str, history: list[dict], question: str) -> dict:
        from google.genai import types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        return {"system": system, "contents": contents}

    def step(self, conv: dict, tools: list[dict], max_tokens: int = 2500) -> Step:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=conv["system"],
            tools=[types.Tool(function_declarations=to_gemini_declarations(tools))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=max_tokens * 3,  # "thinking" token-ууд мөн энэ хязгаарт тоологддог
        )
        resp = self._call(contents=conv["contents"], config=config)
        cands = getattr(resp, "candidates", None) or []
        if cands and cands[0].content is not None:
            # Загварын хариултыг хэвээр нь (thought signature-тай нь) буцааж өгөх шаардлагатай
            conv["contents"].append(cands[0].content)
        return self._parse(resp)

    def add_tool_results(self, conv: dict, results: list[tuple[ToolCall, str]]) -> None:
        from google.genai import types

        parts = [types.Part.from_function_response(name=c.name, response={"result": out}) for c, out in results]
        conv["contents"].append(types.Content(role="user", parts=parts))

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        from google.genai import types

        resp = self._call(contents=prompt, config=types.GenerateContentConfig(max_output_tokens=max_tokens * 3))
        return self._parse(resp).text


def make_llm(provider: str | None = None, model: str | None = None):
    provider = (provider or PROVIDER).lower()
    model = model or MODEL
    if provider == "gemini":
        return GeminiLLM(model or "gemini-3.5-flash-lite")
    if provider == "anthropic":
        return AnthropicLLM(model or "claude-sonnet-5-5")
    raise ValueError(f"Үл мэдэгдэх provider: {provider!r} (gemini эсвэл anthropic)")
