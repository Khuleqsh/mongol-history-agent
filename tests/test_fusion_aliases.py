from mha.aliases import expand_queries
from mha.fusion import cap_per_source, rrf

GROUPS = [["Чингис хаан", "Тэмүжин", "Genghis Khan", "铁木真"]]


def test_rrf_prefers_items_in_multiple_lists():
    scores = rrf([["a", "b", "c"], ["c", "b"], ["b"]])
    order = sorted(scores, key=scores.get, reverse=True)
    assert order[0] == "b"
    assert set(order) == {"a", "b", "c"}


def test_cap_per_source():
    ids = ["x1", "x2", "x3", "y1"]
    out = cap_per_source(ids, lambda i: i[0], 2)
    assert out == ["x1", "x2", "y1"]


def test_expand_adds_variants_and_keeps_originals_first():
    out = expand_queries(["Чингис хаан хэзээ төрсөн бэ?", "birth date"], GROUPS)
    assert out[:2] == ["Чингис хаан хэзээ төрсөн бэ?", "birth date"]
    assert "铁木真" in out and "Genghis Khan" in out


def test_expand_no_match_and_dedupe_and_cap():
    assert expand_queries(["Монгол улс"], GROUPS) == ["Монгол улс"]
    assert expand_queries(["a", "A"], []) == ["a"]
    many = [[f"term{i}" * 2 for i in range(30)]]
    assert len(expand_queries(["term1term1 ?"], many, max_total=10)) == 10
