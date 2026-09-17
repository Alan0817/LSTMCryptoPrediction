import json

import pytest

from documents.types import FinancialDocumentChunk
from retrieval.evaluation import RetrievalJudgmentCase, evaluate, validate_cases, write_artifact
from retrieval.types import RetrievalResult


def chunk(chunk_id):
    return FinancialDocumentChunk(chunk_id, "doc", "Company", "TEST", "0000000001", "0000000001-26-000001", "10-K", "2026-01-01", None, "SEC", "https://sec.test", "PART I ITEM 1A", "Risk Factors", 0, "text")


def result(chunk_id, rank, score):
    return RetrievalResult(rank, score, chunk_id, "doc", "TEST", "Company", "10-K", "2026-01-01", None, "PART I ITEM 1A", "Risk Factors", "preview", "SEC", "https://sec.test")


class FakeRetriever:
    def __init__(self, results): self.results = results; self.calls = []
    def search(self, query, top_k=5, **filters): self.calls.append((query, top_k, filters)); return self.results.get(query, [])[:top_k]


def test_manual_judgments_metrics_diagnostics_groups_and_artifact(tmp_path):
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    cases = (
        RetrievalJudgmentCase("one", "one", "lexical", {"ticker": "TEST"}, ("a", "b")),
        RetrievalJudgmentCase("two", "two", "semantic", {"ticker": "TEST", "document_type": "10-K"}, ("c",)),
        RetrievalJudgmentCase("negative", "none", "negative", {"ticker": "NONE"}),
    )
    retriever = FakeRetriever({"one": [result("a", 1, .9), result("c", 2, .8), result("b", 3, .7)], "two": [result("a", 1, .8), result("c", 2, .5)], "none": []})
    evaluation = evaluate(retriever, chunks, cases)

    assert evaluation["positive_case_count"] == 2 and evaluation["negative_case_count"] == 1
    assert evaluation["metrics"]["hit_at_1"] == .5
    assert evaluation["metrics"]["hit_at_3"] == 1.0
    assert evaluation["metrics"]["recall_at_1"] == .25
    assert evaluation["metrics"]["mrr"] == .75
    assert evaluation["positive_cases"][1]["first_relevant_rank"] == 2
    assert evaluation["positive_cases"][1]["score_gap_top_to_first_relevant"] == pytest.approx(.3)
    assert evaluation["per_category"]["lexical"]["count"] == 1
    assert evaluation["per_ticker"]["TEST"]["count"] == 2
    assert evaluation["negative_cases"][0]["candidate_count"] == 0
    assert retriever.calls[0][1] == 10
    assert json.dumps(evaluation)
    path = write_artifact(evaluation, tmp_path)
    assert json.loads(path.read_text())["benchmark_version"] == "phase-3.3-manual-v1"


def test_benchmark_validation_rejects_duplicate_and_unknown_labels():
    chunks = [chunk("a")]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cases((RetrievalJudgmentCase("x", "q", "x", {}, ("a",)), RetrievalJudgmentCase("x", "q", "x", {}, ("a",))), chunks)
    with pytest.raises(ValueError, match="unknown"):
        validate_cases((RetrievalJudgmentCase("x", "q", "x", {}, ("missing",)),), chunks)
