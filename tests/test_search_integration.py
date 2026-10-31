"""Бодит chromadb + rank_bm25 дээр Searcher-ийн логикийг шалгана. Embedding model-ийг хуурамч (hash) загвараар солино."""
import hashlib
import uuid

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")

from mha.ingest import build_records  # noqa: E402
from mha.search import Searcher  # noqa: E402
from mha.text import tokenize  # noqa: E402


class FakeEmbedder:
    DIM = 128

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            v = np.zeros(self.DIM)
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


class FakeReranker:
    def predict(self, pairs):
        return [1.0 if "ЗОРИУД" in d else 0.0 for _, d in pairs]


DOCS = {
    "shastir": ("Шастир", "# Төрөлт\nТэмүжин Онон мөрний эрэг дээр төрсөн.\n\n# Хаан өргөмжлөл\n1206 онд Тэмүжин Чингис хаан хэмээн өргөмжлөгдсөн.\n\n# Эцэс\n1227 онд Чингис хаан таалал төгссөн."),
    "ger": ("Гэр бүл", "# Гэр\nМонгол гэр нь модон тулгуур, эсгийгээр хийгддэг байшин.\n\n# Хоол\nАйраг нь гүүний сүүгээр исгэсэн ундаа."),
    "zh": ("中文史料", "# 记载\n铁木真于1206年被推举为成吉思汗。"),
}


def build(tmp_aliases=None, reranker=None):
    emb = FakeEmbedder()
    col = chromadb.EphemeralClient().create_collection(f"history_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    for sid, (title, text) in DOCS.items():
        recs = build_records({"id": sid, "file": "x", "title": title, "lang": "mn", "license": "own"}, text)
        col.add(
            ids=[r["id"] for r in recs],
            documents=[r["document"] for r in recs],
            embeddings=emb.encode([r["embed_text"] for r in recs]).tolist(),
            metadatas=[r["meta"] for r in recs],
        )
    s = Searcher(rerank=False, model=emb, collection=col, reranker=reranker)
    s.aliases = tmp_aliases if tmp_aliases is not None else []
    return s


def test_keyword_and_vector_find_relevant_chunk():
    s = build()
    hits = s.search(["Чингис хаан хэмээн өргөмжлөгдсөн"], top_k=3)
    assert hits[0]["id"] == "shastir:1"
    assert hits[0]["section"] == "Хаан өргөмжлөл"


def test_year_filter():
    s = build()
    hits = s.search(["Чингис хаан"], top_k=5, year_from=1220, year_to=1230)
    assert hits and all(1227 in s.years[h["id"]] for h in hits)
    assert s.search(["Чингис хаан"], year_from=1900) == []


def test_alias_expansion_finds_other_language_source():
    s = build([["Тэмүжин", "铁木真"]])
    # нэг эх сурвалжаас дээд тал нь 2 хит авбал хятад эх сурвалж crowded out болохгүй
    ids = [h["id"] for h in s.search(["Тэмүжин хэзээ хаан болсон"], top_k=3, max_per_source=2)]
    assert "zh:0" in ids


def test_cap_per_source_limits_domination():
    s = build()
    hits = s.search(["Чингис Тэмүжин хаан онд"], top_k=6, max_per_source=1)
    sources = [h["source_id"] for h in hits]
    assert len(sources) == len(set(sources))


def test_reranker_reorders():
    class R:
        def predict(self, pairs):
            return [10.0 if "айраг" in d.lower() else 0.0 for _, d in pairs]

    s = build(reranker=R())
    hits = s.search(["Чингис хаан"], top_k=3)
    assert "айраг" in hits[0]["text"].lower()
