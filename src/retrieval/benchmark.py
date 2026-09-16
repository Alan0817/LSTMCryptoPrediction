"""Small manually specified metadata-grounded SEC retrieval benchmark."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from documents.storage import CorpusStorage

from .embeddings import SentenceTransformerEmbeddingModel
from .index import load_index
from .retriever import SemanticRetriever


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    case_id: str
    query: str
    ticker: str
    expected_document_type: str | None = None
    acceptable_sections: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["acceptable_sections"] = list(self.acceptable_sections)
        return value


# Manually reviewed metadata expectations, not automatically inferred relevance labels.
BASELINE_CASES = (
    RetrievalBenchmarkCase("mstr_bitcoin_risks", "What risks does the company disclose about Bitcoin?", "MSTR", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("mstr_bitcoin_volatility", "How does Bitcoin volatility affect the company?", "MSTR", "10-K", ("PART I ITEM 1A", "PART II ITEM 7")),
    RetrievalBenchmarkCase("mstr_bitcoin_custody", "What custody risks exist for the company's Bitcoin holdings?", "MSTR", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("mstr_bitcoin_financing", "How does the company finance Bitcoin acquisitions?", "MSTR", "10-K", ("PART II ITEM 7",)),
    RetrievalBenchmarkCase("nvda_competition", "What risks does NVIDIA describe related to competition?", "NVDA", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("nvda_supply", "What supply constraints could affect NVIDIA?", "NVDA", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("nvda_data_center", "What does NVIDIA say about data center demand?", "NVDA", "10-K", ("PART I ITEM 1", "PART II ITEM 7")),
    RetrievalBenchmarkCase("aapl_supply_chain", "What supply chain risks does Apple disclose?", "AAPL", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("aapl_china", "What risks does Apple face in China?", "AAPL", "10-K", ("PART I ITEM 1A",)),
    RetrievalBenchmarkCase("aapl_services", "What does Apple say about services?", "AAPL", "10-K", ("PART I ITEM 1", "PART II ITEM 7")),
    RetrievalBenchmarkCase("nvda_market_risk", "What market risks does NVIDIA disclose?", "NVDA", "10-K", ("PART II ITEM 7A",)),
    RetrievalBenchmarkCase("aapl_market_risk", "What market risks does Apple disclose?", "AAPL", "10-K", ("PART II ITEM 7A",)),
)


def _is_relevant(result, case: RetrievalBenchmarkCase) -> bool:
    return (
        result.ticker == case.ticker
        and (case.expected_document_type is None or result.document_type == case.expected_document_type)
        and (not case.acceptable_sections or result.section in case.acceptable_sections)
    )


def evaluate_benchmark(retriever: SemanticRetriever, cases=BASELINE_CASES, top_k: int = 5) -> dict:
    """Evaluate metadata-grounded placement without judging prose or answer quality."""
    records = []
    reciprocal_ranks = []
    hits = []
    for case in cases:
        results = retriever.search(case.query, top_k=top_k, ticker=case.ticker, document_type=case.expected_document_type)
        ranks = [result.rank for result in results if _is_relevant(result, case)]
        hit = bool(ranks)
        hits.append(hit)
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        records.append({
            "case": case.to_dict(),
            "results": [result.to_dict() for result in results],
            "hit_at_k": hit,
            "recall_at_k": float(hit),  # Labels define metadata targets, not an exhaustive set of relevant chunks.
            "reciprocal_rank": reciprocal_ranks[-1],
        })
    count = len(records)
    return {
        "case_count": count,
        "top_k": top_k,
        "hit_at_k": sum(hits) / count if count else 0.0,
        "recall_at_k": sum(hits) / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "cases": records,
        "limitations": [
            "Cases use manually reviewed ticker/form/section metadata, not exhaustive relevance labels.",
            "This benchmark measures retrieval placement only; it does not judge answer quality or factual completeness.",
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the local SEC semantic retrieval baseline.")
    parser.add_argument("--corpus-dir", default="data/financial_documents")
    parser.add_argument("--index-dir", default="data/financial_documents/semantic_index")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    chunks = CorpusStorage(args.corpus_dir).load_chunks()
    model = SentenceTransformerEmbeddingModel(args.model)
    index = load_index(chunks, args.index_dir, model)
    started = perf_counter()
    result = evaluate_benchmark(SemanticRetriever(index, chunks, model), top_k=args.top_k)
    result["elapsed_seconds"] = round(perf_counter() - started, 3)
    result["embedding_model"] = model.model_name
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
