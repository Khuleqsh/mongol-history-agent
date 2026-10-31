import json
from pathlib import Path


def load_aliases(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def expand_queries(queries: list[str], groups: list[list[str]], max_total: int = 14) -> list[str]:
    """Асуултад нэрийн хувилбар (Тэмүжин/Temüjin/铁木真...) байвал бусад хувилбарыг нь query болгон нэмнэ.
    Анхны query-нууд үргэлж эхэнд, дараалал хадгалагдана."""
    out: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        key = q.casefold()
        if q and key not in seen:
            seen.add(key)
            out.append(q)

    for q in queries:
        add(q)
    for q in list(out):
        ql = q.casefold()
        for g in groups:
            if any(len(t) >= 3 and t.casefold() in ql for t in g):
                for t in g:
                    add(t)
    return out[:max_total]
