from documents.types import FinancialDocumentChunk
from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.types import RetrievalResult


def make_chunk(chunk_id, text, ticker="MSTR"):
    return FinancialDocumentChunk(
        chunk_id,
        "d" + chunk_id,
        "Co",
        ticker,
        "0000000001",
        "0000000001-26-000001",
        "10-K",
        "2026-01-01",
        None,
        "SEC",
        "url",
        "PART I ITEM 1A",
        "Risk Factors",
        0,
        text,
    )


def make_result(chunk_id, rank):
    return RetrievalResult(
        rank,
        1,
        chunk_id,
        "d" + chunk_id,
        "Co",
        "MSTR",
        "10-K",
        "2026-01-01",
        None,
        "S",
        "S",
        "t",
        "SEC",
        "url",
    )


class FakeRetriever:
    def __init__(self, items):
        self.items = items

    def search(self, *args, top_k=5, **kwargs):
        return self.items[:top_k]


class ReverseReranker(Reranker):
    def rerank(self, query, candidates, top_k):
        return [
            (candidate, 10 - rank)
            for rank, candidate in enumerate(reversed(candidates), 1)
        ][:top_k]


def test_bm25_tokenization_ranking_filters_and_provenance():
    assert tokenize("Bitcoin-CUSTODY!") == ["bitcoin", "custody"]

    retriever = BM25Retriever(
        [
            make_chunk("b", "supply chain"),
            make_chunk("a", "bitcoin custody risk"),
            make_chunk("z", "bitcoin", ticker="AAPL"),
        ]
    )

    results = retriever.search("Bitcoin custody", ticker="MSTR", top_k=5)

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert results[0].source_url == "url"
    assert retriever.search("bitcoin", ticker="NVDA") == []


def test_bm25_section_filters_match_canonical_values_and_titles():
    retriever = BM25Retriever(
        [
            make_chunk("a", "bitcoin custody risk"),
            make_chunk("b", "supply chain"),
        ]
    )

    canonical = retriever.search("bitcoin custody", section="PART I ITEM 1A")
    title = retriever.search("bitcoin custody", section="risk factors")

    assert [item.chunk_id for item in canonical] == [item.chunk_id for item in title]
    assert retriever.search("bitcoin custody", section="unrelated") == []


def test_rrf_and_hybrid_reranking_are_deterministic():
    fused = reciprocal_rank_fusion(
        [
            [make_result("a", 1), make_result("b", 2)],
            [make_result("b", 1), make_result("c", 2)],
        ]
    )
    assert [item[0].chunk_id for item in fused] == ["b", "a", "c"]

    hybrid = HybridRetriever(
        FakeRetriever([make_result("a", 1), make_result("b", 2)]),
        FakeRetriever([make_result("b", 1), make_result("c", 2)]),
        candidate_k=2,
    )
    assert [item.chunk_id for item in hybrid.search("q", top_k=3)] == ["b", "a", "c"]

    reranked_hybrid = HybridRetriever(
        FakeRetriever([make_result("a", 1), make_result("b", 2)]),
        FakeRetriever([make_result("b", 1), make_result("c", 2)]),
        reranker=ReverseReranker(),
        candidate_k=3,
    )
    assert len(reranked_hybrid.search("q", top_k=2)) == 2
    assert reranked_hybrid.last_diagnostics[0]["chunk_id"] == "b"
