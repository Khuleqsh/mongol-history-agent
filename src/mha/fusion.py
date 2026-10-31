from typing import Callable, Hashable, Iterable


def rrf(rank_lists: Iterable[list[Hashable]], k: int = 60) -> dict:
    """Reciprocal Rank Fusion: олон эрэмбэлсэн жагсаалтыг нэг оноонд нэгтгэнэ."""
    scores: dict = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def cap_per_source(ids: list, source_of: Callable, cap: int) -> list:
    """Нэг эх сурвалж үр дүнг дарамтлахаас сэргийлж тус бүрээс дээд тал нь `cap` авна."""
    counts: dict = {}
    out = []
    for i in ids:
        s = source_of(i)
        if counts.get(s, 0) >= cap:
            continue
        counts[s] = counts.get(s, 0) + 1
        out.append(i)
    return out
