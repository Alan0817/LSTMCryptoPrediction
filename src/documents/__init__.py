"""Deterministic SEC filing ingestion and local corpus storage."""

from .ingestion import INITIAL_COMPANIES, ingest_company_filings
from .storage import CorpusStorage, validate_corpus
from .types import CompanyIdentity, FinancialDocument, FinancialDocumentChunk

__all__ = [
    "CompanyIdentity",
    "CorpusStorage",
    "FinancialDocument",
    "FinancialDocumentChunk",
    "INITIAL_COMPANIES",
    "ingest_company_filings",
    "validate_corpus",
]
