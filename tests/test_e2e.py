import json

from agent.types import FinancialAnalysisResult
from evaluation.e2e import FINANCIAL_AGENT_E2E_CASES, evaluate_e2e_case, run_e2e_benchmark, write_e2e_artifact


def case(case_id):
    return next(item for item in FINANCIAL_AGENT_E2E_CASES if item.case_id == case_id)


def event(name, result=None, round_number=1):
    requested = {"event": "tool_requested", "name": name, "arguments": {}, "round": round_number}
    return [requested, {"event": "tool_completed", "name": name, "result": result or {}, "round": round_number}]


def test_e2e_mixed_evidence_and_provenance_are_trace_derived():
    document = {"status": "ok", "results": [{"chunk_id": "c", "document_id": "d", "ticker": "MSTR", "document_type": "10-K", "filing_date": "2026-01-01", "section": "ITEM 1A", "source_url": "https://sec"}]}
    web = {"status": "ok", "results": [{"title": "News", "url": "https://news"}]}
    trace = event("analyze_market") + event("search_financial_documents", document, 2) + event("search_web", web, 3) + [{"event": "final_response"}]
    result = FinancialAnalysisResult("Answer", ["analyze_market", "search_financial_documents", "search_web"], trace, [])
    record = evaluate_e2e_case(case("full_mstr"), result)
    assert record["routing_metrics"]["missing_required_tools"] == []
    assert record["document_metrics"]["provenance_preserved"]
    assert record["web_metrics"]["provenance_preserved"]
    assert all(value["used"] for value in record["evidence_coverage"].values())
    assert json.dumps(record)


class FakeAgent:
    def __init__(self, results):
        self.results = list(results)

    def run(self, prompt):
        return self.results.pop(0)


def test_e2e_runner_records_failure_and_writes_artifact(tmp_path):
    result = FinancialAnalysisResult("RSI answer", [], [{"event": "final_response"}], [])
    run = run_e2e_benchmark(FakeAgent([result, RuntimeError("failure")]), FINANCIAL_AGENT_E2E_CASES[:2])
    assert run["aggregate"]["attempted_cases"] == 2
    assert run["aggregate"]["completed_cases"] == 1
    assert run["records"][1]["completed"] is False
    path = write_e2e_artifact(run, tmp_path)
    assert json.loads(path.read_text())["benchmark_version"] == "financial-agent-e2e-v1"
