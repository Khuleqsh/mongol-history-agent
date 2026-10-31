"""Өөрийгөө сайжруулах мөчлөгийн хэмжүүр: eval асуултууд, baseline, түүх."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check(item: dict, r: dict) -> bool:
    """Нэг асуултын шалгуур.
    - expect_gap: архивт байхгүй асуултад агент "мэдэхгүй" гэж gap бүртгэсэн байх
    - must_include: бүх үг хариултад байх; any_of: ядаж нэг нь байх
    - хайлт хийсэн, [S#] ишлэлтэй, ишлэл бүр бодит эх сурвалжтай байх
    """
    if item.get("expect_gap"):
        return bool(r["gap"])
    ans = r["answer"].lower()
    ok = all(t.lower() in ans for t in item.get("must_include", []))
    anys = item.get("any_of", [])
    if anys:
        ok = ok and any(t.lower() in ans for t in anys)
    return bool(ok and r["searched"] and re.search(r"\[S\d+\]", r["answer"]) and not r["invalid_citations"])


def run_eval(agent, items: list[dict], verbose: bool = True) -> float:
    passed = 0
    for it in items:
        ok = check(it, agent.ask(it["q"]))
        passed += ok
        if verbose:
            print(("PASS" if ok else "FAIL"), it["q"])
    return passed / len(items) if items else 0.0


def read_baseline(path: Path) -> float | None:
    return json.loads(path.read_text())["score"] if path.exists() else None


def regressed(score: float, baseline: float | None, tolerance: float = 0.02) -> bool:
    return baseline is not None and score < baseline - tolerance


def append_history(path: Path, score: float, prompt: str, model: str, total: int, provider: str = "") -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "provider": provider,
        "model": model,
        "score": round(score, 4),
        "total": total,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
