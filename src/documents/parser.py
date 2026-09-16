"""Deterministic HTML-to-text normalization for SEC primary filing documents."""

import re

from bs4 import BeautifulSoup


def parse_filing_html(html: str) -> str:
    """Remove non-content markup while retaining paragraphs, headings, and table rows."""
    if not isinstance(html, str) or not html.strip():
        raise ValueError("Filing HTML must be a non-empty string.")
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        element.decompose()
    for element in soup.find_all(
        lambda tag: tag.name and tag.name.lower() in {"ix:header", "ix:hidden"}
    ):
        element.decompose()
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [_normalize_inline(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    text = soup.get_text("\n")
    lines = [_normalize_inline(line) for line in text.splitlines()]
    return _normalize_lines(lines)


def _normalize_inline(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _normalize_lines(lines: list[str]) -> str:
    normalized = []
    blank = False
    for line in lines:
        if not line:
            if normalized and not blank:
                normalized.append("")
            blank = True
            continue
        normalized.append(line)
        blank = False
    return "\n".join(normalized).strip()
