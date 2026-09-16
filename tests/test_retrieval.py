import json
from pathlib import Path

import numpy as np
import pytest

from documents.types import FinancialDocumentChunk
from retrieval.benchmark import RetrievalBenchmarkCase, evaluate_benchmark
from retrieval.embeddings import EmbeddingModel
from retrieval.index import build_index, corpus_fingerprint, load_index
from retrieval.retriever import SemanticRetriever


class FakeEmbeddings(EmbeddingModel):
    def __init__(self, vectors, query_vector=(1.0, 0.0), name="fake-embeddings-v1"):
        self.vectors = vectors
        self.query_vector = query_vector
        self.name = name
        self.calls = []

    @property
    def model_name(self):
        return self.name

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=float)

    def embed_query(self, text):
        return np.asarray(self.query_vector, dtype=float)


def make_chunk(chunk_id, text, ticker="MSTR", document_type="10-K", section="PART I ITEM 1A", filing_date="2026-02-19"):
    return FinancialDocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_" + ticker,
        company=ticker + " Company",
        ticker=ticker,
        cik="0000000001",
        accession_number="0000000001-26-000001",
        document_type=document_type,
        filing_date=filing_date,
        period_end="2025-12-31",
        source="SEC EDGAR",
        source_url="https://www.sec.gov/example/" + chunk_id,
        section=section,
        section_title="Risk Factors",
        chunk_index=int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
        text=text,
    )


def make_index(tmp_path, chunks=None, vectors=None, query_vector=(1.0, 0.0)):
    chunks = chunks or [make_chunk("chunk_b", "beta"), make_chunk("chunk_a", "alpha")]
    vectors = vectors or {"alpha": (1.0, 0.0), "beta": (0.0, 1.0)}
    model = FakeEmbeddings(vectors, query_vector)
    index = build_index(chunks, model, tmp_path / "index", batch_size=1)
    return chunks, model, index


def test_index_build_is_sorted_normalized_and_json_metadata(tmp_path):
    chunks, model, index = make_index(tmp_path)

    assert index.chunk_ids == ("chunk_a", "chunk_b")
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert model.calls == [["alpha"], ["beta"]]
    metadata = json.loads((tmp_path / "index" / "metadata.json").read_text())
    assert metadata["chunk_ids"] == ["chunk_a", "chunk_b"]
    assert metadata["embedding_dimension"] == 2
    assert metadata["chunk_count"] == 2


def test_load_checks_mapping_fingerprint_and_embedding_identity(tmp_path):
    chunks, model, _ = make_index(tmp_path)
    loaded = load_index(chunks, tmp_path / "index", model)
    assert loaded.chunk_ids == ("chunk_a", "chunk_b")

    changed = [make_chunk("chunk_b", "beta"), make_chunk("chunk_a", "changed")]
    with pytest.raises(ValueError, match="does not match"):
        load_index(changed, tmp_path / "index")
    with pytest.raises(ValueError, match="embedding model"):
        load_index(chunks, tmp_path / "index", FakeEmbeddings({"alpha": (1, 0), "beta": (0, 1)}, name="other"))


def test_fingerprint_is_stable_and_changes_with_text():
    chunks = [make_chunk("chunk_a", "alpha")]
    assert corpus_fingerprint(chunks) == corpus_fingerprint(list(reversed(chunks)))
    assert corpus_fingerprint(chunks) != corpus_fingerprint([make_chunk("chunk_a", "changed")])


@pytest.mark.parametrize("vectors, message", [
    ({"alpha": (0.0, 0.0)}, "zero vectors"),
    ({"alpha": (float("nan"), 1.0)}, "finite"),
])
def test_invalid_document_vectors_are_rejected(tmp_path, vectors, message):
    with pytest.raises(ValueError, match=message):
        build_index([make_chunk("chunk_a", "alpha")], FakeEmbeddings(vectors), tmp_path / "index")


def test_duplicate_empty_and_corrupt_indexes_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        build_index([make_chunk("chunk_a", "alpha"), make_chunk("chunk_a", "beta")], FakeEmbeddings({"alpha": (1, 0), "beta": (0, 1)}), tmp_path / "index")
    with pytest.raises(ValueError, match="empty"):
        build_index([make_chunk("chunk_a", " ")], FakeEmbeddings({" ": (1, 0)}), tmp_path / "index")
    with pytest.raises(FileNotFoundError):
        load_index([], tmp_path / "missing")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "metadata.json").write_text("not json")
    (tmp_path / "bad" / "vectors.npy").write_bytes(b"not numpy")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_index([], tmp_path / "bad")


def test_search_ranking_filters_ties_and_json_provenance(tmp_path):
    chunks = [
        make_chunk("chunk_c", "third", ticker="AAPL", document_type="10-Q", section="PART I ITEM 2"),
        make_chunk("chunk_b", "second", ticker="MSTR", section="PART II ITEM 7"),
        make_chunk("chunk_a", "first", ticker="MSTR"),
    ]
    vectors = {"first": (1, 0), "second": (1, 0), "third": (0, 1)}
    _, model, index = make_index(tmp_path, chunks, vectors)
    retriever = SemanticRetriever(index, chunks, model)

    results = retriever.search("risk", top_k=10, ticker="MSTR", document_type="10-K")
    assert [item.chunk_id for item in results] == ["chunk_a", "chunk_b"]
    assert [item.rank for item in results] == [1, 2]
    assert all(item.ticker == "MSTR" and item.document_type == "10-K" for item in results)
    assert results[0].source_url.endswith("chunk_a")
    assert json.dumps([item.to_dict() for item in results])
    assert retriever.search("risk", ticker="NVDA") == []
    assert [item.chunk_id for item in retriever.search("risk", section="PART II ITEM 7")] == ["chunk_b"]
    assert [item.chunk_id for item in retriever.search("risk", filing_date_from="2026-01-01", filing_date_to="2026-12-31")] == ["chunk_a", "chunk_b", "chunk_c"]


@pytest.mark.parametrize("kwargs", [
    {"query": " "}, {"query": "risk", "top_k": 0}, {"query": "risk", "document_type": "8-K"},
    {"query": "risk", "ticker": ""}, {"query": "risk", "filing_date_from": "nope"},
    {"query": "risk", "filing_date_from": "2026-02-01", "filing_date_to": "2026-01-01"},
])
def test_search_validates_inputs(tmp_path, kwargs):
    chunks, model, index = make_index(tmp_path)
    with pytest.raises(ValueError):
        SemanticRetriever(index, chunks, model).search(**kwargs)


def test_query_vector_failures_and_benchmark_metrics(tmp_path):
    chunks, model, index = make_index(tmp_path)
    retriever = SemanticRetriever(index, chunks, model)
    model.query_vector = (0.0, 0.0)
    with pytest.raises(ValueError, match="zero vector"):
        retriever.search("risk")
    model.query_vector = (1.0, 0.0)
    case = RetrievalBenchmarkCase("risk", "risk", "MSTR", "10-K", ("PART I ITEM 1A",))
    evaluation = evaluate_benchmark(retriever, (case,), top_k=1)
    assert evaluation["hit_at_k"] == evaluation["recall_at_k"] == evaluation["mrr"] == 1.0
    assert json.dumps(evaluation)


def test_retrieval_package_stays_provider_and_agent_free():
    root = Path(__file__).resolve().parents[1] / "src" / "retrieval"
    source = "\n".join(path.read_text().lower() for path in root.glob("*.py"))
    for forbidden in ("openai", "gemini", "financialanalysisagent", "toolregistry", "from agent", "import agent"):
        assert forbidden not in source
