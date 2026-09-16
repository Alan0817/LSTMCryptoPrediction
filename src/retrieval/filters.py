"""Shared deterministic metadata-filter helpers for local retrieval."""

from documents.types import FinancialDocumentChunk


def normalize_section(value: str) -> str:
    """Normalize a section identifier or title for exact comparison."""
    return " ".join(value.split()).casefold()


def section_matches(chunk: FinancialDocumentChunk, section: str | None) -> bool:
    """Match a requested section against its canonical identifier or title."""
    if section is None:
        return True

    requested = normalize_section(section)
    return requested in {
        normalize_section(chunk.section),
        normalize_section(chunk.section_title),
    }
