import json
from datetime import datetime, timezone

from . import config
from .config import EVAL_DIR, FEEDBACK

CANDIDATES = EVAL_DIR / "candidates.jsonl"


def log_feedback(question: str, answer: str, verdict: str, note: str = "") -> None:
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "verdict": verdict,
        "note": note,
    }
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if verdict == "bad":
        # Буруу гэж тэмдэглэсэн асуулт eval-ийн нэр дэвшигч болно. Зөв хариултыг ХҮН нөхөж баталгаажуулна.
        CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
        with CANDIDATES.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"q": question, "must_include": [], "note": note}, ensure_ascii=False) + "\n")
