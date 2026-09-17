"""RRF hybrid and optional reranked retrieval with dense-compatible search."""

from .fusion import reciprocal_rank_fusion
from .types import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        reranker=None,
        candidate_k=20,
        rrf_k=60,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.last_diagnostics = []

    def search(self, query, top_k=5, **filters):
        dense = self.dense.search(query, top_k=self.candidate_k, **filters)
        bm25 = self.bm25.search(query, top_k=self.candidate_k, **filters)
        fused = reciprocal_rank_fusion([dense, bm25], self.rrf_k)
        self.last_diagnostics = [
            {
                "chunk_id": chunk.chunk_id,
                "dense_rank": next(
                    (item.rank for item in dense if item.chunk_id == chunk.chunk_id),
                    None,
                ),
                "bm25_rank": next(
                    (item.rank for item in bm25 if item.chunk_id == chunk.chunk_id),
                    None,
                ),
                "hybrid_score": score,
            }
            for chunk, score in fused
        ]
        if self.reranker:
            reranked = self.reranker.rerank(query, fused[:self.candidate_k], top_k)
            selected = [(pair[0][0], float(pair[1])) for pair in reranked]
        else:
            selected = fused[:top_k]

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
            for rank, (chunk, score) in enumerate(selected, 1)
        ]
