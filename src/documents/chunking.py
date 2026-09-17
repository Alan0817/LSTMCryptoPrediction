"""Deterministic paragraph-aware chunks that never cross detected section boundaries."""

from .sections import FilingSection
from .types import FinancialDocument, FinancialDocumentChunk, make_chunk_id


def chunk_document(
    document: FinancialDocument,
    sections: list[FilingSection],
    target_size: int = 1800,
    max_size: int = 2200,
    overlap: int = 200,
) -> list[FinancialDocumentChunk]:
    """Create section-aware chunks using paragraphs first and character splits only as a fallback."""
    if target_size <= 0 or max_size < target_size or overlap < 0 or overlap >= max_size:
        raise ValueError("Chunk sizes must satisfy 0 < target_size <= max_size and 0 <= overlap < max_size.")
    chunks = []
    chunk_index = 0
    for section in sections:
        for text in _chunk_section_text(section.text, target_size, max_size, overlap):
            chunks.append(
                FinancialDocumentChunk(
                    chunk_id=make_chunk_id(document.document_id, section.section, chunk_index),
                    document_id=document.document_id,
                    company=document.company,
                    ticker=document.ticker,
                    cik=document.cik,
                    accession_number=document.accession_number,
                    document_type=document.document_type,
                    filing_date=document.filing_date,
                    period_end=document.period_end,
                    source=document.source,
                    source_url=document.source_url,
                    section=section.section,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            chunk_index += 1
    return chunks


def _chunk_section_text(text: str, target_size: int, max_size: int, overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    units = []
    for paragraph in paragraphs:
        units.extend(_split_oversized_paragraph(paragraph, max_size))
    chunks = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        if current and (len(current) + len(separator) + len(unit) > max_size or len(current) >= target_size):
            chunks.append(current)
            prefix = current[-overlap:].lstrip() if overlap else ""
            current = (prefix + "\n\n" if prefix else "") + unit
        else:
            current += separator + unit
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_paragraph(paragraph: str, max_size: int) -> list[str]:
    if len(paragraph) <= max_size:
        return [paragraph]
    pieces = []
    remainder = paragraph
    while len(remainder) > max_size:
        boundary = remainder.rfind(" ", 0, max_size + 1)
        boundary = boundary if boundary > max_size // 2 else max_size
        pieces.append(remainder[:boundary].strip())
        remainder = remainder[boundary:].lstrip()
    if remainder:
        pieces.append(remainder)
    return pieces
