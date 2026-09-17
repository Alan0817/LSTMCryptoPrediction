"""Deterministic end-to-end evidence-routing evaluation for the research agent."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from retrieval.planning import document_plan_metrics, web_plan_metrics


BENCHMARK_VERSION = "financial-agent-e2e-v1"
QUANTITATIVE_TOOLS = {"analyze_market", "get_market_data", "get_risk_metrics"}


@dataclass(frozen=True)
class E2ECase:
    case_id: str
    category: str
    prompt: str
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_document_tickers: tuple[str, ...] = ()
    expected_document_types: tuple[str, ...] = ()
    expected_limitation_codes: tuple[str, ...] = ()
    max_document_calls: int = 2
    max_web_calls: int = 2


def _case(case_id, category, prompt, required=(), forbidden=(), **kwargs):
    return E2ECase(case_id, category, prompt, required, forbidden, **kwargs)


FINANCIAL_AGENT_E2E_CASES = (
    _case("concept_rsi", "conceptual", "What is RSI?"),
    _case("concept_10k", "conceptual", "What is a 10-K?"),
    _case("concept_mdd", "conceptual", "What is maximum drawdown?"),
    _case("quant_nvda", "quantitative", "Analyze NVDA from 2026-01-01 to 2026-06-30.", ("analyze_market",), ("search_web", "search_financial_documents")),
    _case("quant_risk", "quantitative", "Calculate risk metrics for [0.01, -0.02, 0.03].", ("get_risk_metrics",)),
    _case("sec_mstr_custody", "sec", "What Bitcoin custody risks does MSTR disclose?", ("search_financial_documents",), ("search_web",), expected_document_tickers=("MSTR",)),
    _case("sec_aapl_supply", "sec", "What supply-chain risks does Apple disclose?", ("search_financial_documents",), ("search_web",), expected_document_tickers=("AAPL",)),
    _case("sec_nvda_competition", "sec", "What competitive risks does NVIDIA disclose?", ("search_financial_documents",), ("search_web",), expected_document_tickers=("NVDA",)),
    _case("web_nvda", "web", "What are the latest developments involving NVIDIA?", ("search_web",), ("search_financial_documents",)),
    _case("web_mstr", "web", "What recent developments affect MSTR's Bitcoin strategy?", ("search_web",), ("search_financial_documents",)),
    _case("quant_sec_mstr", "quant_sec", "Analyze MSTR from 2026-01-01 to 2026-06-30 and explain Bitcoin filing risks.", ("analyze_market", "search_financial_documents"), expected_document_tickers=("MSTR",)),
    _case("quant_web_nvda", "quant_web", "Analyze NVDA's recent market performance and relevant recent events.", ("analyze_market", "search_web")),
    _case("sec_web_mstr", "sec_web", "Compare recent MSTR Bitcoin developments with filing risks.", ("search_financial_documents", "search_web"), expected_document_tickers=("MSTR",)),
    _case("full_mstr", "full", "Research MSTR using market behavior, SEC disclosures, and recent developments.", ("analyze_market", "search_financial_documents", "search_web"), expected_document_tickers=("MSTR",)),
    _case("btc_model", "applicability", "Analyze BTC-USD using the ML capability.", ("analyze_market",)),
    _case("aapl_model", "applicability", "Use the BTC LSTM for AAPL.", ("analyze_market",), expected_limitation_codes=("model_symbol_applicability",)),
    _case("mstr_model", "applicability", "Use the BTC LSTM for MSTR.", ("analyze_market",), expected_limitation_codes=("model_symbol_applicability",)),
    _case("unsupported_sec", "unavailable", "What SEC filing evidence is available for TSLA?", ("search_financial_documents",), expected_limitation_codes=("document_retrieval_no_results",)),
    _case("unavailable_web", "unavailable", "What are today's developments involving an unsupported topic?", expected_limitation_codes=("web_search_unavailable",)),
    _case("unavailable_capability", "unavailable", "What is the current P/E ratio for a private company?"),
)


def evaluate_e2e_case(case, result):
    requests = [event for event in result.trace if event.get("event") == "tool_requested"]
    completed = [event for event in result.trace if event.get("event") == "tool_completed"]
    names = [event.get("name") for event in requests if isinstance(event.get("name"), str)]
    missing = [name for name in case.required_tools if name not in names]
    forbidden = [name for name in names if name in case.forbidden_tools]
    evidence = {
        "quantitative": {"required": bool(set(case.required_tools) & QUANTITATIVE_TOOLS), "used": bool(set(names) & QUANTITATIVE_TOOLS)},
        "document": {"required": "search_financial_documents" in case.required_tools, "used": "search_financial_documents" in names},
        "web": {"required": "search_web" in case.required_tools, "used": "search_web" in names},
    }
    document_results = [event.get("result", {}) for event in completed if event.get("name") == "search_financial_documents"]
    web_results = [event.get("result", {}) for event in completed if event.get("name") == "search_web"]
    document_items = [item for payload in document_results for item in payload.get("results", [])]
    web_items = [item for payload in web_results for item in payload.get("results", [])]
    document_provenance = all(all(key in item for key in ("chunk_id", "document_id", "ticker", "document_type", "filing_date", "section", "source_url")) for item in document_items)
    web_provenance = all("title" in item and "url" in item for item in web_items)
    codes = {item.get("code") for item in result.limitations if isinstance(item, dict)}
    duplicate = len({(event.get("name"), json.dumps(event.get("arguments", {}), sort_keys=True)) for event in requests}) != len(requests)
    return {
        "case_id": case.case_id, "category": case.category, "completed": bool(result.answer.strip()), "answer": result.answer,
        "tools_used": result.tools_used, "trace": result.trace, "routing_metrics": {"missing_required_tools": missing, "forbidden_tool_calls": forbidden, "duplicate_exact_calls": duplicate, "tool_call_count": len(names), "tool_round_count": len({event.get("round") for event in requests if isinstance(event.get("round"), int)})},
        "evidence_coverage": evidence, "document_metrics": {"calls": len(document_results), "statuses": [item.get("status") for item in document_results], "result_count": len(document_items), "provenance_preserved": document_provenance, "expected_tickers_present": not case.expected_document_tickers or set(case.expected_document_tickers).issubset({item.get("ticker") for item in document_items})},
        "web_metrics": {"calls": len(web_results), "statuses": [item.get("status") for item in web_results], "result_count": len(web_items), "provenance_preserved": web_provenance},
        "document_plan_metrics": document_plan_metrics(result.trace), "web_plan_metrics": web_plan_metrics(result.trace),
        "limitation_metrics": {"expected": list(case.expected_limitation_codes), "preserved": set(case.expected_limitation_codes).issubset(codes)},
        "failure_reasons": missing + forbidden,
    }


def run_e2e_benchmark(agent, cases=FINANCIAL_AGENT_E2E_CASES, provider_label="unspecified", model=None, retrieval_backend="hybrid"):
    records = []
    for case in cases:
        try:
            records.append(evaluate_e2e_case(case, agent.run(case.prompt)))
        except Exception as error:
            records.append({"case_id": case.case_id, "category": case.category, "completed": False, "failure_reasons": ["{}: {}".format(type(error).__name__, error)]})
    successful = [record for record in records if record["completed"]]
    categories = {}
    for record in successful:
        categories.setdefault(record["category"], []).append(record)
    aggregate = {"attempted_cases": len(records), "completed_cases": len(successful), "required_tool_recall": sum(1 - bool(record["routing_metrics"]["missing_required_tools"]) for record in successful) / len(successful) if successful else 0.0, "missing_required_calls": sum(len(record["routing_metrics"]["missing_required_tools"]) for record in successful), "forbidden_calls": sum(len(record["routing_metrics"]["forbidden_tool_calls"]) for record in successful), "duplicate_calls": sum(record["routing_metrics"]["duplicate_exact_calls"] for record in successful), "categories": {name: len(items) for name, items in sorted(categories.items())}}
    return {"benchmark_version": BENCHMARK_VERSION, "provider": provider_label, "model": model, "retrieval_backend": retrieval_backend, "timestamp": datetime.now(timezone.utc).isoformat(), "aggregate": aggregate, "records": records}


def write_e2e_artifact(run, output_directory="evaluation_results"):
    root = Path(output_directory); root.mkdir(parents=True, exist_ok=True)
    provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["provider"])
    path = root / "financial_agent_e2e_{}_{}.json".format(provider, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path.write_text(json.dumps(run, indent=2) + "\n")
    return path
