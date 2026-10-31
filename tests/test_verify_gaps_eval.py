import json

from mha import evalkit, feedback
from mha.gaps import summarize_gaps
from mha.ingest import build_records, validate_sources
from mha.verify import extract_json


def test_extract_json():
    assert extract_json('text {"unsupported": []} more') == {"unsupported": []}
    assert extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json("no json") is None
    assert extract_json("{broken") is None


def test_summarize_gaps(tmp_path):
    p = tmp_path / "gaps.jsonl"
    rows = [
        {"topic": "Зуд 1944", "question": "q1"},
        {"topic": "зуд 1944", "question": "q2"},
        {"topic": "Хүннү", "question": "q3"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = summarize_gaps(p)
    assert out[0] == ("зуд 1944", 2, "q1")
    assert summarize_gaps(tmp_path / "none.jsonl") == []


def good(**kw):
    r = {"answer": "1206 онд хаан болсон [S1].", "sources": {"S1": {}}, "invalid_citations": [], "gap": False, "searched": True}
    r.update(kw)
    return r


def test_check_rules():
    item = {"q": "?", "must_include": ["1206"]}
    assert evalkit.check(item, good())
    assert not evalkit.check(item, good(answer="1207 онд [S1]"))
    assert not evalkit.check(item, good(answer="1206 онд хаан болсон"))  # ишлэлгүй
    assert not evalkit.check(item, good(invalid_citations=["S9"]))
    assert not evalkit.check(item, good(searched=False))
    assert evalkit.check({"q": "?", "any_of": ["1206", "1207"]}, good())
    assert evalkit.check({"q": "?", "expect_gap": True}, good(gap=True, answer="мэдэхгүй"))
    assert not evalkit.check({"q": "?", "expect_gap": True}, good(gap=False))


def test_baseline_regression(tmp_path):
    assert not evalkit.regressed(0.5, None)
    assert evalkit.regressed(0.7, 0.9)
    assert not evalkit.regressed(0.89, 0.9)
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"score": 0.8}))
    assert evalkit.read_baseline(p) == 0.8
    h = tmp_path / "h.jsonl"
    evalkit.append_history(h, 0.8, "v2", "m", 10)
    assert json.loads(h.read_text())["prompt"] == "v2"


def test_feedback_bad_creates_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK", tmp_path / "fb.jsonl")
    monkeypatch.setattr(feedback, "CANDIDATES", tmp_path / "cand.jsonl")
    feedback.log_feedback("q?", "a", "good")
    assert not (tmp_path / "cand.jsonl").exists()
    feedback.log_feedback("q2?", "a", "bad", "буруу он")
    assert json.loads((tmp_path / "cand.jsonl").read_text())["q"] == "q2?"


def test_validate_sources(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    ok = {"id": "a", "file": "a.md", "title": "T", "lang": "mn", "license": "public-domain"}
    assert validate_sources([ok], tmp_path) == []
    bad = [
        {"id": "a", "file": "missing.md", "title": "T", "lang": "mn", "license": "wtf"},
        {"id": "a", "file": "a.md", "title": "", "lang": "mn", "license": "own"},
    ]
    errs = validate_sources(bad, tmp_path)
    joined = "\n".join(errs)
    assert "license" in joined and "олдсонгүй" in joined and "дутуу" in joined and "давхардсан" in joined


def test_build_records_metadata():
    src = {"id": "s", "file": "s.md", "title": "Шастир", "lang": "mn", "license": "own", "kind": "primary"}
    text = "# Нэг\n1206 онд юм болов.\n\n# Хоёр\n1227 онд өөр юм болов."
    recs = build_records(src, text)
    assert [r["meta"]["section"] for r in recs] == ["Нэг", "Хоёр"]
    assert recs[0]["meta"]["years"] == "1206"
    assert recs[1]["embed_text"].startswith("Шастир — Хоёр")
    assert [r["id"] for r in recs] == ["s:0", "s:1"]
