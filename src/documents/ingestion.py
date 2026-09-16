"""Deterministic SEC discovery, parsing, chunking, cache, and JSONL orchestration."""

from dataclasses import dataclass

from .chunking import chunk_document
from .parser import parse_filing_html
from .sec_client import SECClient
from .sections import detect_sections
from .storage import CorpusStorage, RawFilingCache
from .types import CompanyIdentity, FinancialDocument, FinancialDocumentChunk, FilingReference, make_document_id


INITIAL_COMPANIES = (
    CompanyIdentity("NVIDIA Corporation", "NVDA", "0001045810"),
    CompanyIdentity("Apple Inc.", "AAPL", "0000320193"),
    CompanyIdentity("Strategy Inc.", "MSTR", "0001050446"),
)


@dataclass(frozen=True)
class IngestionResult:
    company: CompanyIdentity
    documents: list[FinancialDocument]
    chunks: list[FinancialDocumentChunk]
    missing_forms: list[str]


def ingest_company_filings(
    company_identity: CompanyIdentity,
    sec_client: SECClient,
    storage: CorpusStorage,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    latest_per_form: int = 1,
    raw_cache: RawFilingCache | None = None,
    target_chunk_size: int = 1800,
    max_chunk_size: int = 2200,
    chunk_overlap: int = 200,
) -> IngestionResult:
    """Ingest selected official primary filings; missing requested forms remain explicit."""
    references = sec_client.find_filings(company_identity.cik, forms=forms, latest_per_form=latest_per_form)
    found_forms = {reference.form for reference in references}
    documents = []
    chunks = []
    for reference in references:
        document, document_chunks = ingest_filing(
            company_identity,
            reference,
            sec_client,
            raw_cache=raw_cache,
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents.append(document)
        chunks.extend(document_chunks)
    storage.write_documents(documents)
    storage.write_chunks(chunks, replace_document_ids=[document.document_id for document in documents])
    return IngestionResult(
        company=company_identity,
        documents=documents,
        chunks=chunks,
        missing_forms=sorted(set(forms).difference(found_forms)),
    )


def ingest_filing(
    company_identity: CompanyIdentity,
    reference: FilingReference,
    sec_client: SECClient,
    raw_cache: RawFilingCache | None = None,
    target_chunk_size: int = 1800,
    max_chunk_size: int = 2200,
    chunk_overlap: int = 200,
) -> tuple[FinancialDocument, list[FinancialDocumentChunk]]:
    """Process one primary SEC filing HTML document using no model-derived content."""
    document_id = make_document_id(reference.cik, reference.accession_number, reference.primary_document)
    html = raw_cache.load(document_id) if raw_cache else None
    if html is None:
        html = sec_client.download_filing(reference)
        if raw_cache:
            raw_cache.write(document_id, html)
    text = parse_filing_html(html)
    document = FinancialDocument(
        document_id=document_id,
        company=company_identity.company,
        ticker=company_identity.ticker,
        cik=company_identity.cik,
        accession_number=reference.accession_number,
        primary_document=reference.primary_document,
        document_type=reference.form,
        filing_date=reference.filing_date,
        period_end=reference.period_end,
        source="SEC EDGAR",
        source_url=sec_client.filing_url(reference),
        text=text,
    )
    sections = detect_sections(text, reference.form)
    return document, chunk_document(document, sections, target_chunk_size, max_chunk_size, chunk_overlap)
