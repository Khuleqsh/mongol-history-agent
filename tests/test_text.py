from mha.text import chunk_text, extract_years, split_sections, tokenize, years_overlap


def test_chunks_respect_size_and_keep_text():
    text = "\n\n".join(f"Догол мөр {i}. " + "а" * 200 for i in range(10))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert "Догол мөр 0." in chunks[0]
    assert "Догол мөр 9." in chunks[-1]
    assert all(len(c) < 800 for c in chunks)


def test_chunk_empty():
    assert chunk_text("") == []


def test_tokenize_cyrillic_and_latin():
    assert tokenize("Чингис хаан, Temüjin!") == ["чингис", "хаан", "temüjin"]


def test_extract_years():
    assert extract_years("1206 онд хаан өргөмжлөгдөв. 1227 оны намар. 2024-д.") == [1206, 1227, 2024]
    assert extract_years("Тэр 209 онд болсон") == [209]
    assert extract_years("Тэнд 12345 хүн байв") == []


def test_years_overlap():
    assert years_overlap([1206, 1227], 1200, 1210)
    assert not years_overlap([1206], 1300, None)
    assert years_overlap([], None, None)
    assert not years_overlap([], 1200, 1300)


def test_split_sections():
    text = "# Нэг\nэхний\n\n## Хоёр\nхоёрдугаар\n"
    secs = split_sections(text)
    assert [h for h, _ in secs] == ["Нэг", "Хоёр"]
    assert "эхний" in secs[0][1]
    assert split_sections("гарчиггүй текст") == [("", "гарчиггүй текст")]


def test_tokenize_cjk_per_character():
    assert tokenize("铁木真于1206年") == ["铁", "木", "真", "于", "1206", "年"]
