from .aliases import expand_queries, load_aliases
from .config import ALIASES, EMBED_MODEL, INDEX, RERANK_MODEL
from .fusion import cap_per_source, rrf
from .text import tokenize, years_overlap


class Searcher:
    """Hybrid хайлт: нэрийн хувилбараар query өргөжүүлэх -> vector + BM25 -> RRF -> (reranker) -> эх сурвалжаар тэнцвэржүүлэх."""

    def __init__(self, rerank: bool = True, model=None, collection=None, reranker=None) -> None:
        from rank_bm25 import BM25Okapi

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL)
        if collection is None:
            import chromadb

            collection = chromadb.PersistentClient(path=str(INDEX)).get_collection("history")
        self.model, self.col = model, collection
        data = self.col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.pos = {i: n for n, i in enumerate(self.ids)}
        self.years = {
            cid: [int(y) for y in m.get("years", "").split(",") if y] for cid, m in zip(self.ids, self.metas)
        }
        self.bm25 = BM25Okapi([tokenize(f'{m["title"]} {m.get("section", "")} {d}') for d, m in zip(self.docs, self.metas)])
        self.aliases = load_aliases(ALIASES)
        self.reranker = reranker
        if self.reranker is None and rerank and RERANK_MODEL:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(RERANK_MODEL)

    def search(
        self,
        queries: list[str],
        top_k: int = 6,
        year_from: int | None = None,
        year_to: int | None = None,
        max_per_source: int = 3,
    ) -> list[dict]:
        if not self.ids:
            return []
        queries = expand_queries(queries, self.aliases)
        n = min(30, len(self.ids))
        qembs = self.model.encode(queries, normalize_embeddings=True).tolist()
        rank_lists = []
        for q, qe in zip(queries, qembs):
            rank_lists.append(self.col.query(query_embeddings=[qe], n_results=n)["ids"][0])
            bm = self.bm25.get_scores(tokenize(q))
            order = sorted(range(len(bm)), key=lambda j: -bm[j])[:n]
            rank_lists.append([self.ids[j] for j in order if bm[j] > 0])

        scores = rrf(rank_lists)
        cands = sorted(scores, key=scores.get, reverse=True)[:40]
        if year_from is not None or year_to is not None:
            cands = [c for c in cands if years_overlap(self.years[c], year_from, year_to)]
        if self.reranker and cands:
            rs = self.reranker.predict([(queries[0], self.docs[self.pos[c]]) for c in cands])
            cands = [c for c, _ in sorted(zip(cands, rs), key=lambda x: -float(x[1]))]
        cands = cap_per_source(cands, lambda c: self.metas[self.pos[c]]["source_id"], max_per_source)[:top_k]
        return [{"id": c, "text": self.docs[self.pos[c]], **self.metas[self.pos[c]]} for c in cands]
