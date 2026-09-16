"""Best-effort deterministic SEC Part/Item section detection."""

from dataclasses import dataclass
import re


ITEM_PATTERN = re.compile(r"^ITEM\s+(\d+[A-Z]?)\s*[.\-:—–]*\s*(.*)$", re.IGNORECASE)
PART_PATTERN = re.compile(r"^PART\s+(I|II)\b\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class FilingSection:
    section: str
    title: str
    text: str


def detect_sections(text: str, document_type: str) -> list[FilingSection]:
    """Split normalized filing text on likely SEC headings, retaining unknown content."""
    if document_type not in {"10-K", "10-Q"}:
        raise ValueError("Section detection supports only 10-K and 10-Q documents.")
    lines = text.splitlines()
    candidates = _heading_candidates(lines)
    candidates = _remove_likely_toc(candidates)
    if not candidates:
        return [FilingSection("UNKNOWN", "Unknown", text.strip())] if text.strip() else []

    sections = []
    first_line = candidates[0][0]
    preamble = "\n".join(lines[:first_line]).strip()
    if preamble:
        sections.append(FilingSection("UNKNOWN", "Unknown", preamble))
    for index, (line_number, section, title) in enumerate(candidates):
        next_line = candidates[index + 1][0] if index + 1 < len(candidates) else len(lines)
        body = "\n".join(lines[line_number + 1:next_line]).strip()
        if body:
            sections.append(FilingSection(section, title, body))
    return sections or [FilingSection("UNKNOWN", "Unknown", text.strip())]


def _heading_candidates(lines: list[str]) -> list[tuple[int, str, str]]:
    candidates = []
    current_part = None
    for index, line in enumerate(lines):
        normalized = " ".join(line.split())
        part = PART_PATTERN.match(normalized)
        if part and len(normalized) <= 100:
            if "|" in part.group(2):
                continue
            current_part = "PART " + part.group(1).upper()
            candidates.append((index, current_part, normalized.title()))
            continue
        item = ITEM_PATTERN.match(normalized)
        if item and len(normalized) <= 180:
            item_number = item.group(1).upper()
            title = item.group(2).strip() or "Item " + item_number
            # Table-of-contents rows preserved from HTML tables often become
            # ``Item 9C | Title | page``. They are not document section starts.
            if title.startswith("|"):
                continue
            section = "ITEM " + item_number
            if current_part:
                section = current_part + " " + section
            candidates.append((index, section, title))
    return candidates


def _remove_likely_toc(candidates: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """Drop an early dense run of headings, a common table-of-contents pattern."""
    if len(candidates) < 5:
        return candidates
    early = candidates[:5]
    if all("ITEM" in candidate[1] for candidate in early) and early[-1][0] - early[0][0] <= 20:
        cutoff = early[-1][0]
        return [candidate for candidate in candidates if candidate[0] > cutoff]
    return candidates
