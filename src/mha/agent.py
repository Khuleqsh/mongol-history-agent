import json
import re
from datetime import datetime, timezone

from .config import GAPS, MODEL, PROMPT, PROMPTS
from .search import Searcher
from .verify import revise_answer, verify_answer

LEVELS = {
    "standard": "",
    "brief": "\n\nLevel: brief. At most ~120 words: a short answer and a 3-step timeline. Skip follow-up questions.",
    "student": (
        "\n\nLevel: student (age 14-18). Use simple words, define unfamiliar terms (khan, yassa, ger ...), "
        "explain why each event mattered, and give a short timeline."
    ),
    "researcher": (
        "\n\nLevel: researcher. Be precise about dating, variant readings and source provenance; "
        "state the type of each source and list contested points explicitly."
    ),
}

TOOLS = [
    {
        "name": "search_history",
        "description": (
            "Hybrid (semantic + keyword) multilingual search over the history archive. "
            "The FIRST query must be the user's own question. Name variants are expanded automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "top_k": {"type": "integer", "default": 6},
                "year_from": {"type": "integer", "description": "Only passages mentioning a year >= this"},
                "year_to": {"type": "integer", "description": "Only passages mentioning a year <= this"},
            },
            "required": ["queries"],
        },
    },
    {
        "name": "report_gap",
        "description": "Log that the archive lacks sources for this topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


class Agent:
    def __init__(self, client=None, searcher=None, prompt: str | None = None, model: str | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.searcher = searcher or Searcher()
        self.prompt_name = prompt or PROMPT
        self.system_base = load_prompt(self.prompt_name)
        self.model = model or MODEL

    def ask(
        self,
        question: str,
        level: str = "standard",
        history: list[dict] | None = None,
        verify: bool = False,
        max_turns: int = 6,
    ) -> dict:
        system = self.system_base + LEVELS.get(level, "")
        messages = list(history or []) + [{"role": "user", "content": question}]
        labels: dict[str, str] = {}
        cited: dict[str, dict] = {}
        passages: dict[str, str] = {}
        state = {"gap": False, "searched": False, "question": question}

        answer = ""
        for _ in range(max_turns):
            resp = self.client.messages.create(
                model=self.model, max_tokens=2500, system=system, tools=TOOLS, messages=messages
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if b.type == "text")
                break
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = self._run(b.name, b.input, labels, cited, passages, state)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
        else:
            answer = "Хайлт хэт удаан үргэлжилсэн тул дуусгаж чадсангүй. Асуултаа жижиглэн дахин асууна уу."

        unsupported: list[dict] = []
        if verify and passages and re.search(r"\[S\d+\]", answer):
            unsupported = verify_answer(self.client, self.model, answer, passages)
            if unsupported:
                answer = revise_answer(self.client, self.model, answer, passages, unsupported)

        used = set(re.findall(r"\[(S\d+)\]", answer))
        return {
            "answer": answer,
            "sources": {k: v for k, v in cited.items() if k in used},
            "invalid_citations": sorted(used - set(cited)),
            "gap": state["gap"],
            "searched": state["searched"],
            "unsupported_removed": unsupported,
        }

    def _run(self, name, args, labels, cited, passages, state) -> str:
        if name == "report_gap":
            state["gap"] = True
            GAPS.parent.mkdir(parents=True, exist_ok=True)
            with GAPS.open("a", encoding="utf-8") as f:
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "question": state["question"],
                    "topic": args["topic"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return "Gap logged."
        if name != "search_history":
            return f"Unknown tool: {name}"
        state["searched"] = True
        hits = self.searcher.search(
            args["queries"],
            top_k=args.get("top_k", 6),
            year_from=args.get("year_from"),
            year_to=args.get("year_to"),
        )
        if not hits:
            return "No results."
        lines = []
        for h in hits:
            label = labels.setdefault(h["id"], f"S{len(labels) + 1}")
            cited[label] = {k: h.get(k, "") for k in ("title", "author", "year", "lang", "kind", "section", "chunk")}
            passages[label] = h["text"]
            where = f', {h["section"]}' if h.get("section") else ""
            head = f'[{label}] {h["title"]} — {h["author"]} {h["year"]} ({h["lang"]}, {h.get("kind", "")}){where}, chunk {h["chunk"]}'
            lines.append(head + "\n" + h["text"])
        return "\n\n".join(lines)
