import json
from collections import Counter
from pathlib import Path


def summarize_gaps(path: Path) -> list[tuple[str, int, str]]:
    """Архивт дутсан сэдвүүдийг давтамжаар нь эрэмбэлнэ: (сэдэв, тоо, жишээ асуулт)."""
    if not path.exists():
        return []
    counts: Counter = Counter()
    example: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec["topic"].strip().casefold()
        counts[key] += 1
        example.setdefault(key, rec.get("question", ""))
    return [(t, c, example[t]) for t, c in counts.most_common()]
