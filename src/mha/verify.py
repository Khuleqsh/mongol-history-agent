"""Хариултын ишлэл бүр эх сурвалжаар үнэхээр баталгаажиж байгааг тусдаа дуудлагаар шалгана."""
import json
import re

VERIFY_PROMPT = """You are a strict fact-checker. Below are SOURCE PASSAGES and an ANSWER that cites them with labels like [S1].
For every factual claim in the ANSWER, check that the cited passage actually supports it.
Return ONLY JSON: {"unsupported": [{"claim": "...", "reason": "..."}]}. Use an empty list if every claim is supported.
Be strict: a claim whose date, name or number differs from the passage is unsupported."""

REVISE_PROMPT = """Rewrite the ANSWER below. Remove or correct every claim listed under UNSUPPORTED so that everything left is
supported by the SOURCE PASSAGES. Keep the same language, structure and [S#] citations. Do not add new facts."""


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _passages_block(answer: str, passages: dict[str, str]) -> str:
    used = set(re.findall(r"\[(S\d+)\]", answer))
    return "\n\n".join(f"[{k}] {v}" for k, v in passages.items() if k in used)


def verify_answer(llm, answer: str, passages: dict[str, str]) -> list[dict]:
    body = f"{VERIFY_PROMPT}\n\nSOURCE PASSAGES:\n{_passages_block(answer, passages)}\n\nANSWER:\n{answer}"
    data = extract_json(llm.complete(body, max_tokens=1200))
    if isinstance(data, dict) and isinstance(data.get("unsupported"), list):
        return data["unsupported"]
    return []


def revise_answer(llm, answer: str, passages: dict[str, str], unsupported: list[dict]) -> str:
    body = (
        f"{REVISE_PROMPT}\n\nSOURCE PASSAGES:\n{_passages_block(answer, passages)}\n\n"
        f"UNSUPPORTED:\n{json.dumps(unsupported, ensure_ascii=False)}\n\nANSWER:\n{answer}"
    )
    return llm.complete(body, max_tokens=2500) or answer
