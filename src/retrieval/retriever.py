"""Deterministic filtered exact-cosine search over a SemanticIndex."""

from datetime import date
from typing import Sequence

import numpy as np

from documents.types import FinancialDocumentChunk

from .embeddings import EmbeddingModel
from .index import SemanticIndex
from .types import RetrievalResult


class SemanticRetriever:
    def __init__(self, index: SemanticIndex, chunks: Sequence[FinancialDocumentChunk], embedding_model: EmbeddingModel):
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("Retriever chunks contain duplicate chunk IDs.")
        if set(index.chunk_ids) != set(by_id):
            raise ValueError("Retriever chunks do not match semantic index chunk IDs.")
        self._index = index
        self._chunks = tuple(by_id[chunk_id] for chunk_id in index.chunk_ids)
        self._embedding_model = embedding_model

    @staticmethod
    def _validate_date(value: str | None, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("{} must be an ISO date string.".format(name))
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("{} must be an ISO date string.".format(name)) from error
        return value

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
        document_type: str | None = None,
        section: str | None = None,
        filing_date_from: str | None = None,
        filing_date_to: str | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if ticker is not None and (not isinstance(ticker, str) or not ticker.strip()):
            raise ValueError("ticker must be a non-empty string when provided.")
        if document_type is not None and document_type not in {"10-K", "10-Q"}:
            raise ValueError("document_type must be '10-K' or '10-Q'.")
        if section is not None and (not isinstance(section, str) or not section.strip()):
            raise ValueError("section must be a non-empty string when provided.")
        filing_date_from = self._validate_date(filing_date_from, "filing_date_from")
        filing_date_to = self._validate_date(filing_date_to, "filing_date_to")
        if filing_date_from and filing_date_to and filing_date_from > filing_date_to:
            raise ValueError("filing_date_from must not be later than filing_date_to.")

        candidates = [
            index for index, chunk in enumerate(self._chunks)
            if (ticker is None or chunk.ticker == ticker.upper())
            and (document_type is None or chunk.document_type == document_type)
            and (section is None or chunk.section == section)
            and (filing_date_from is None or chunk.filing_date >= filing_date_from)
            and (filing_date_to is None or chunk.filing_date <= filing_date_to)
        ]
        if not candidates:
            return []
        query_vector = np.asarray(self._embedding_model.embed_query(query), dtype=np.float64)
        if query_vector.ndim != 1 or query_vector.shape[0] != self._index.dimension:
            raise ValueError("Query embedding dimension does not match the semantic index.")
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding must contain only finite values.")
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Query embedding must not be a zero vector.")
        scores = self._index.vectors[candidates] @ (query_vector / norm)
        ordered = sorted(zip(candidates, scores), key=lambda item: (-float(item[1]), self._chunks[item[0]].chunk_id))[:top_k]
        return [
            RetrievalResult(
                rank=rank,
                score=float(score),
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                ticker=chunk.ticker,
                company=chunk.company,
                document_type=chunk.document_type,
                filing_date=chunk.filing_date,
                period_end=chunk.period_end,
                section=chunk.section,
                section_title=chunk.section_title,
                text=chunk.text,
                source=chunk.source,
                source_url=chunk.source_url,
            )
            for rank, (candidate, score) in enumerate(ordered, start=1)
            for chunk in [self._chunks[candidate]]
        ]
