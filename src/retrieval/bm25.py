"""Small deterministic BM25 retriever for local SEC chunks."""

from collections import Counter
from datetime import date
import math
import re

from documents.types import FinancialDocumentChunk

from .filters import section_matches
from .types import RetrievalResult


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.text) for chunk in self.chunks]
        self.tf = [Counter(document) for document in self.docs]
        self.lengths = [len(document) for document in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        document_frequency = Counter(
            term for document in self.docs for term in set(document)
        )
        document_count = len(self.docs)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query,
        top_k=5,
        ticker=None,
        document_type=None,
        section=None,
        filing_date_from=None,
        filing_date_to=None,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if document_type is not None and document_type not in {"10-K", "10-Q"}:
            raise ValueError("document_type must be '10-K' or '10-Q'.")
        for value in (filing_date_from, filing_date_to):
            if value is not None:
                date.fromisoformat(value)

        terms = tokenize(query)
        candidates = []
        for index, chunk in enumerate(self.chunks):
            if (
                (ticker and chunk.ticker != ticker.upper())
                or (document_type and chunk.document_type != document_type)
                or not section_matches(chunk, section)
                or (filing_date_from and chunk.filing_date < filing_date_from)
                or (filing_date_to and chunk.filing_date > filing_date_to)
            ):
                continue

            score = sum(
                self.idf.get(term, 0)
                * self.tf[index][term]
                * (self.k1 + 1)
                / (
                    self.tf[index][term]
                    + self.k1
                    * (1 - self.b + self.b * self.lengths[index] / self.avgdl)
                )
                for term in terms
                if self.tf[index][term]
            )
            candidates.append((score, chunk))

        candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(
                rank,
                float(score),
                chunk.chunk_id,
                chunk.document_id,
                chunk.ticker,
                chunk.company,
                chunk.document_type,
                chunk.filing_date,
                chunk.period_end,
                chunk.section,
                chunk.section_title,
                chunk.text,
                chunk.source,
                chunk.source_url,
            )
            for rank, (score, chunk) in enumerate(candidates[:top_k], 1)
        ]
