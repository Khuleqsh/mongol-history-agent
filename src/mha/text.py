"""Цэвэр (model-гүй) текст боловсруулалт: tokenizer, chunker, бүлэг, он илрүүлэлт."""
import re

_WORD = re.compile(r"\w+", re.UNICODE)
_YEAR4 = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")
_YEAR3 = re.compile(r"(?<!\d)(\d{3})\s?(?:оны|онд|оноос|он|AD|CE)(?!\w)")
_HEADING = re.compile(r"^#{1,4}\s+(.*)$")


_CJK = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])")


def tokenize(text: str) -> list[str]:
    """Кирилл/Латин үгээр, хятад/япон/солонгос бичгийг тэмдэгт тус бүрээр нь token болгоно (BM25 CJK-д ажиллахын тулд)."""
    return _WORD.findall(_CJK.sub(r" \1 ", text.lower()))


def extract_years(text: str) -> list[int]:
    years = {int(y) for y in _YEAR4.findall(text)}
    years |= {int(y) for y in _YEAR3.findall(text)}
    return sorted(years)


def years_overlap(years: list[int], lo: int | None, hi: int | None) -> bool:
    if lo is None and hi is None:
        return True
    return any((lo is None or y >= lo) and (hi is None or y <= hi) for y in years)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Markdown гарчгаар (#, ##) тасалж (гарчиг, агуулга) буцаана."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf)))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf)))
    return sections


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Догол мөрөөр нь нийлүүлж ~size тэмдэгттэй хэсгүүд болгоно."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= size:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            tail = buf[-overlap:]
            buf = (tail + "\n\n" + p) if len(p) < size else p
    if buf:
        chunks.append(buf)
    return chunks
