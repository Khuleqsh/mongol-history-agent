import hashlib
import json
from pathlib import Path

from .config import EMBED_MODEL, INDEX, RAW, SOURCES
from .text import chunk_text, extract_years, split_sections

REQUIRED = ("id", "file", "title", "lang", "license")
ALLOWED_LICENSES = {
    "public-domain", "cc0", "cc-by", "cc-by-sa", "open", "own", "permission",
    "personal-use",  # зөвхөн өөрийн машинд; data/raw/personal/ дотор байлгаж commit хийхгүй
}


def validate_sources(sources: list[dict], raw_dir: Path = RAW) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for n, s in enumerate(sources):
        label = s.get("id", f"#{n}")
        for k in REQUIRED:
            if not s.get(k):
                errors.append(f"{label}: '{k}' талбар дутуу")
        if s.get("id") in seen:
            errors.append(f"{label}: id давхардсан")
        seen.add(s.get("id", ""))
        if s.get("license") and s["license"] not in ALLOWED_LICENSES:
            errors.append(f"{label}: license '{s['license']}' зөвшөөрөгдсөн жагсаалтад алга")
        if s.get("file") and not (raw_dir / s["file"]).exists():
            errors.append(f"{label}: {raw_dir / s['file']} файл олдсонгүй")
    return errors


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def build_records(source: dict, text: str) -> list[dict]:
    """Нэг эх сурвалжийг chunk болгож metadata-тай нь буцаана (model шаардахгүй, тест хийх боломжтой)."""
    records = []
    n = 0
    for heading, body in split_sections(text) or [("", text)]:
        for chunk in chunk_text(body):
            header = source["title"] + (f" — {heading}" if heading else "")
            records.append(
                {
                    "id": f'{source["id"]}:{n}',
                    "document": chunk,
                    "embed_text": f"{header}\n{chunk}",  # chunk бүрийн өмнө эх сурвалж, бүлгийн нэр наана (contextual retrieval)
                    "meta": {
                        "source_id": source["id"],
                        "title": source["title"],
                        "author": source.get("author", ""),
                        "year": str(source.get("year", "")),
                        "lang": source.get("lang", ""),
                        "kind": source.get("kind", ""),
                        "license": source.get("license", ""),
                        "section": heading,
                        "years": ",".join(str(y) for y in extract_years(chunk)),
                        "chunk": n,
                        "sha": hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12],
                    },
                }
            )
            n += 1
    return records


def build_index() -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    errors = validate_sources(sources)
    if errors:
        raise SystemExit("sources.json алдаатай:\n- " + "\n- ".join(errors))

    model = SentenceTransformer(EMBED_MODEL)
    INDEX.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX))
    try:
        client.delete_collection("history")
    except Exception:
        pass
    col = client.create_collection("history", metadata={"hnsw:space": "cosine"})

    for s in sources:
        records = build_records(s, load_text(RAW / s["file"]))
        if not records:
            continue
        embs = model.encode(
            [r["embed_text"] for r in records], normalize_embeddings=True, batch_size=16, show_progress_bar=True
        ).tolist()
        col.add(
            ids=[r["id"] for r in records],
            documents=[r["document"] for r in records],
            embeddings=embs,
            metadatas=[r["meta"] for r in records],
        )
        print(f'{s["id"]}: {len(records)} chunk')
