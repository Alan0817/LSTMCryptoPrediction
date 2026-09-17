"""Deterministic, manually judged evaluation for the unchanged SEC retriever."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


KS = (1, 3, 5, 10)
BENCHMARK_VERSION = "phase-3.3-manual-v1"


@dataclass(frozen=True)
class RetrievalJudgmentCase:
    case_id: str
    query: str
    category: str
    filters: dict
    relevant_chunk_ids: tuple[str, ...] = ()
    notes: str = ""
    manual_failure_labels: tuple[str, ...] = ()

    @property
    def is_negative(self) -> bool:
        return not self.relevant_chunk_ids

    def to_dict(self):
        value = asdict(self)
        value["relevant_chunk_ids"] = list(self.relevant_chunk_ids)
        value["manual_failure_labels"] = list(self.manual_failure_labels)
        return value


# Each positive ID was manually inspected in the Phase 3.0 filing-derived corpus.
PHASE_3_3_CASES = (
    RetrievalJudgmentCase("mstr_custody_exact", "bitcoin custody risk", "lexical", {"ticker": "MSTR"}, ("chunk_14d86ad6cc4c7805bc417671",)),
    RetrievalJudgmentCase("mstr_private_keys", "How could losing access to cryptographic keys affect the company?", "semantic_paraphrase", {"ticker": "MSTR"}, ("chunk_0782f77e28acdf50873968c6",)),
    RetrievalJudgmentCase("mstr_volatility", "How does Bitcoin price volatility affect MSTR?", "semantic_paraphrase", {"ticker": "MSTR"}, ("chunk_af3b7a4eafc3c4e0ee54172c",)),
    RetrievalJudgmentCase("mstr_financing", "How does MSTR finance Bitcoin acquisitions?", "business_fact", {"ticker": "MSTR", "document_type": "10-K"}, ("chunk_09da482097bf3965dc2b6c7f",)),
    RetrievalJudgmentCase("mstr_concentration", "What concentration risks apply to MSTR Bitcoin holdings?", "lexical", {"ticker": "MSTR", "document_type": "10-Q"}, ("chunk_229348983fdda6b524597d5b",)),
    RetrievalJudgmentCase("mstr_counterparty", "What counterparty risks are associated with MSTR Bitcoin custody?", "semantic_paraphrase", {"ticker": "MSTR", "document_type": "10-Q"}, ("chunk_0294109069a0c2d0a559f629",)),
    RetrievalJudgmentCase("mstr_strategy", "What is MSTR's Bitcoin strategy?", "business_fact", {"ticker": "MSTR", "document_type": "10-K"}, ("chunk_1a2793e49cdb04f9f48c1055",)),
    RetrievalJudgmentCase("mstr_market_risk", "What market risk does MSTR disclose?", "section_specific", {"ticker": "MSTR", "document_type": "10-K", "section": "PART II ITEM 7A"}, ("chunk_5ce9da0b1ce3949be86b3cdc",)),
    RetrievalJudgmentCase("mstr_q_custody", "What does the latest MSTR quarterly filing say about custody?", "temporal", {"ticker": "MSTR", "document_type": "10-Q"}, ("chunk_14d86ad6cc4c7805bc417671",)),
    RetrievalJudgmentCase("mstr_multi_document", "What Bitcoin risks does MSTR disclose?", "multi_document", {"ticker": "MSTR"}, ("chunk_8d15f227174983a4919f47f0", "chunk_48130649392ba8c8323cc8a0")),
    RetrievalJudgmentCase("nvda_competition", "What competitive risks does NVIDIA face?", "semantic_paraphrase", {"ticker": "NVDA"}, ("chunk_034fbe1815145d0264db645b", "chunk_02b9a4d51ffe8162819ad5fd"), "Prior baseline top result was 10-Q MD&A; labels include manually inspected 10-K risk evidence."),
    RetrievalJudgmentCase("nvda_competition_exact", "competition", "lexical", {"ticker": "NVDA", "document_type": "10-K"}, ("chunk_034fbe1815145d0264db645b",)),
    RetrievalJudgmentCase("nvda_supply", "What could prevent NVIDIA from obtaining enough components?", "semantic_paraphrase", {"ticker": "NVDA"}, ("chunk_0dfa38db7c45569c08a62cdd",)),
    RetrievalJudgmentCase("nvda_data_center", "What does NVIDIA say about data center demand?", "business_fact", {"ticker": "NVDA", "document_type": "10-K"}, ("chunk_062b04d49ba4640998d49b0a",)),
    RetrievalJudgmentCase("nvda_export_controls", "What export-control risks does NVIDIA disclose?", "section_specific", {"ticker": "NVDA", "document_type": "10-K", "section": "PART I ITEM 1A"}, ("chunk_02b9a4d51ffe8162819ad5fd",)),
    RetrievalJudgmentCase("nvda_foundry", "What manufacturing or foundry dependency risks does NVIDIA disclose?", "semantic_paraphrase", {"ticker": "NVDA", "document_type": "10-K"}, ("chunk_57cc2dd798752dcc63723804",)),
    RetrievalJudgmentCase("nvda_quarterly_risks", "What risks does NVIDIA's latest quarterly filing identify?", "temporal", {"ticker": "NVDA", "document_type": "10-Q"}, ("chunk_1873fe6d7aafbd86762807ec",)),
    RetrievalJudgmentCase("nvda_market_risk", "What market risk does NVIDIA disclose?", "section_specific", {"ticker": "NVDA", "document_type": "10-K", "section": "PART II ITEM 7A"}, ("chunk_0dfa38db7c45569c08a62cdd",), "Manual label is a supply-chain volatility disclosure; section labels may reveal corpus heading limitations."),
    RetrievalJudgmentCase("nvda_multi_document", "How could supply constraints affect NVIDIA?", "multi_document", {"ticker": "NVDA"}, ("chunk_0dfa38db7c45569c08a62cdd", "chunk_02a48025456df8e613ac9957")),
    RetrievalJudgmentCase("aapl_supply_shortages", "supply chain risk", "lexical", {"ticker": "AAPL"}, ("chunk_6e3583804ab8213883d14a6a",)),
    RetrievalJudgmentCase("aapl_manufacturing", "What manufacturing concentration risks does Apple disclose?", "semantic_paraphrase", {"ticker": "AAPL", "document_type": "10-K"}, ("chunk_2265ebbeb2d07596bef13a58",)),
    RetrievalJudgmentCase("aapl_china", "What risks does Apple face in China?", "business_fact", {"ticker": "AAPL", "document_type": "10-K"}, ("chunk_2265ebbeb2d07596bef13a58",)),
    RetrievalJudgmentCase("aapl_services", "What does Apple say about services?", "business_fact", {"ticker": "AAPL", "document_type": "10-Q"}, ("chunk_012b59e0dcfa451a1d716739",)),
    RetrievalJudgmentCase("aapl_competition", "What competition risks does Apple disclose?", "section_specific", {"ticker": "AAPL", "document_type": "10-K", "section": "PART I ITEM 1A"}, ("chunk_59fa763c5b454500b5647f11",)),
    RetrievalJudgmentCase("aapl_regulation", "What regulatory risks does Apple disclose?", "section_specific", {"ticker": "AAPL", "document_type": "10-K", "section": "PART I ITEM 1A"}, ("chunk_30141099ccfd0b117a322e94",)),
    RetrievalJudgmentCase("aapl_quarterly_supply", "What supply risks appear in Apple's latest quarterly filing?", "temporal", {"ticker": "AAPL", "document_type": "10-Q"}, ("chunk_012b59e0dcfa451a1d716739",)),
    RetrievalJudgmentCase("aapl_multi_document", "How can supply shortages affect Apple?", "multi_document", {"ticker": "AAPL"}, ("chunk_6e3583804ab8213883d14a6a", "chunk_012b59e0dcfa451a1d716739")),
    RetrievalJudgmentCase("negative_tesla", "What Tesla battery risks are in the local SEC corpus?", "negative", {"ticker": "TSLA"}),
    RetrievalJudgmentCase("negative_nvda_dividend", "What dividend policy does NVIDIA disclose for common stock?", "negative", {"ticker": "NVDA"}),
    RetrievalJudgmentCase("negative_aapl_bitcoin", "What Bitcoin custody policy does Apple disclose?", "negative", {"ticker": "AAPL"}),
)


def validate_cases(cases, chunks):
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate retrieval benchmark case IDs.")
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    for case in cases:
        if case.is_negative:
            continue
        if not case.relevant_chunk_ids:
            raise ValueError("Positive case {} has no relevant chunks.".format(case.case_id))
        unknown = set(case.relevant_chunk_ids).difference(chunk_ids)
        if unknown:
            raise ValueError("Case {} references unknown chunks: {}.".format(case.case_id, sorted(unknown)))


def _metrics(records):
    result = {"count": len(records)}
    for k in KS:
        result["hit_at_{}".format(k)] = sum(bool(set(record["retrieved_ids"][:k]).intersection(record["relevant_chunk_ids"])) for record in records) / len(records) if records else 0.0
        result["recall_at_{}".format(k)] = sum(len(set(record["retrieved_ids"][:k]).intersection(record["relevant_chunk_ids"])) / len(record["relevant_chunk_ids"]) for record in records) / len(records) if records else 0.0
        result["ndcg_at_{}".format(k)] = sum(_ndcg(record, k) for record in records) / len(records) if records else 0.0
    result["mrr"] = sum(1 / record["first_relevant_rank"] if record["first_relevant_rank"] else 0.0 for record in records) / len(records) if records else 0.0
    return result


def _ndcg(record, k):
    relevant = set(record["relevant_chunk_ids"])
    dcg = sum((1.0 / __import__("math").log2(rank + 1)) for rank, chunk_id in enumerate(record["retrieved_ids"][:k], 1) if chunk_id in relevant)
    ideal = sum(1.0 / __import__("math").log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(retriever, chunks, cases=PHASE_3_3_CASES):
    validate_cases(cases, chunks)
    positives, negatives = [], []
    for case in cases:
        results = retriever.search(case.query, top_k=10, **case.filters)
        retrieved_ids = [item.chunk_id for item in results]
        relevant = set(case.relevant_chunk_ids)
        ranks = [item.rank for item in results if item.chunk_id in relevant]
        first_rank = min(ranks) if ranks else None
        top = results[0] if results else None
        first = next((item for item in results if item.chunk_id in relevant), None)
        record = {"case": case.to_dict(), "relevant_chunk_ids": list(case.relevant_chunk_ids), "retrieved_ids": retrieved_ids,
                  "first_relevant_rank": first_rank, "relevant_in_top_1": sum(i in relevant for i in retrieved_ids[:1]),
                  "relevant_in_top_3": sum(i in relevant for i in retrieved_ids[:3]), "relevant_in_top_5": sum(i in relevant for i in retrieved_ids[:5]),
                  "relevant_in_top_10": sum(i in relevant for i in retrieved_ids[:10]), "top_result_chunk_id": top.chunk_id if top else None,
                  "top_result_score": top.score if top else None, "first_relevant_score": first.score if first else None,
                  "score_gap_top_to_first_relevant": (top.score - first.score) if top and first else None,
                  "results": [item.to_dict() for item in results]}
        if case.is_negative:
            negatives.append({"case": case.to_dict(), "candidate_count": len(results), "top_score": top.score if top else None, "top_chunk_id": top.chunk_id if top else None, "results": [item.to_dict() for item in results]})
        else:
            positives.append(record)
    def grouped(key):
        values = {}
        for record in positives:
            group = record["case"]["filters"].get(key, "any") if key in {"ticker", "document_type"} else record["case"][key]
            values.setdefault(group, []).append(record)
        return {name: _metrics(records) for name, records in sorted(values.items())}
    return {"benchmark_version": BENCHMARK_VERSION, "positive_case_count": len(positives), "negative_case_count": len(negatives),
            "metrics": _metrics(positives), "per_category": grouped("category"), "per_ticker": grouped("ticker"),
            "per_document_type": grouped("document_type"), "positive_cases": positives, "negative_cases": negatives,
            "limitations": ["Manual relevance labels are intentionally incomplete.", "Negative cases study nearest-neighbor scores; no abstention threshold is applied."]}


def write_artifact(evaluation, output_dir="evaluation_results"):
    payload = {**evaluation, "timestamp": datetime.now(timezone.utc).isoformat()}
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    path = root / "retrieval_phase_3_3_{}.json".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
