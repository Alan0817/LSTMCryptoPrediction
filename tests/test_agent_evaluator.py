import json
from pathlib import Path

import pytest

from agent.types import FinancialAnalysisResult
from evaluation.agent_cases import AGENT_EVALUATION_CASES
from evaluation.agent_evaluator import _load_pairs, evaluate_case, evaluate_results


def case(case_id):
    return next(item for item in AGENT_EVALUATION_CASES if item.id == case_id)


def result(trace=(), limitations=()):
    return FinancialAnalysisResult(
        answer="A deterministic fake answer.", tools_used=[], trace=list(trace), limitations=list(limitations)
    )


def request(name, arguments=None, round_number=1):
    return {"event": "tool_requested", "name": name, "arguments": arguments or {}, "round": round_number}


def test_conceptual_no_tool_case_has_no_tool_penalties():
    evaluation = evaluate_case(case("conceptual_rsi"), result(trace=[{"event": "final_response"}]))

    assert evaluation.required_tool_recall == 1.0
    assert evaluation.tool_call_count == 0
    assert evaluation.unnecessary_tool_calls == []


@pytest.mark.parametrize(
    ("case_id", "tool_name"),
    [
        ("risk_metrics", "get_risk_metrics"),
        ("market_data", "get_market_data"),
    ],
)
def test_explicit_tool_cases_measure_required_tool_recall(case_id, tool_name):
    evaluation = evaluate_case(case(case_id), result(trace=[request(tool_name)]))

    assert evaluation.required_tool_recall == 1.0
    assert evaluation.missing_required_tools == []


def test_full_analysis_extra_allowed_tool_is_measured_not_marked_unnecessary():
    evaluation = evaluate_case(
        case("btc_market_analysis"),
        result(trace=[request("analyze_market", round_number=1), request("get_market_data", round_number=2)]),
    )

    assert evaluation.required_tool_recall == 1.0
    assert evaluation.extra_tool_calls == ["get_market_data"]
    assert evaluation.unnecessary_tool_calls == []
    assert evaluation.tool_round_count == 2


def test_exact_duplicate_tool_requests_are_detected():
    evaluation = evaluate_case(
        case("risk_metrics"),
        result(
            trace=[
                request("get_risk_metrics", {"returns": [0.01, -0.01]}),
                request("get_risk_metrics", {"returns": [0.01, -0.01]}),
            ]
        ),
    )

    assert evaluation.duplicate_tool_calls == 1
    assert evaluation.tool_call_count == 2


def test_forbidden_and_missing_tools_are_reported():
    forbidden = evaluate_case(
        case("unsupported_lstm_request"), result(trace=[request("get_lstm_prediction")])
    )
    missing = evaluate_case(case("risk_metrics"), result())

    assert forbidden.forbidden_tool_calls == ["get_lstm_prediction"]
    assert forbidden.unnecessary_tool_calls == ["get_lstm_prediction"]
    assert missing.required_tool_recall == 0.0
    assert missing.missing_required_tools == ["get_risk_metrics"]


def test_structured_limitations_are_measured_for_unsupported_symbol_case():
    evaluation = evaluate_case(
        case("unsupported_symbol_analysis"),
        result(
            trace=[request("analyze_market")],
            limitations=[
                {"code": "model_symbol_applicability", "applicable": False},
                {"code": "lstm_prediction_status", "status": "not_applicable"},
            ],
        ),
    )

    assert evaluation.limitation_preservation is True


def test_summary_aggregates_and_is_json_safe():
    summary = evaluate_results(
        [
            (case("conceptual_rsi"), result(trace=[{"event": "final_response"}])),
            (
                case("btc_market_analysis"),
                result(trace=[request("analyze_market"), request("get_market_data", round_number=2)]),
            ),
        ]
    )

    assert summary.cases == 2
    assert summary.required_tool_successes == 2
    assert summary.extra_tool_calls == 1
    assert summary.unnecessary_tool_calls == 0
    assert summary.tool_round_count == 2
    json.dumps(summary.to_dict())


def test_offline_json_runner_loads_captured_results(tmp_path):
    payload = [
        {
            "case_id": "conceptual_rsi",
            "result": result(trace=[{"event": "final_response"}]).to_dict(),
        }
    ]
    path = tmp_path / "captured_results.json"
    path.write_text(json.dumps(payload))

    pairs = _load_pairs(path)

    assert pairs[0][0].id == "conceptual_rsi"
    assert pairs[0][1].answer == "A deterministic fake answer."


def test_architecture_boundaries_remain_provider_neutral_and_offline():
    root = Path(__file__).resolve().parents[1] / "src"
    agent_source = (root / "agent" / "financial_agent.py").read_text()
    evaluator_source = (root / "evaluation" / "agent_evaluator.py").read_text().lower()
    registry_source = (root / "tools" / "registry.py").read_text().lower()
    deterministic_tools = "\n".join(path.read_text().lower() for path in (root / "tools").glob("*.py"))
    provider_sources = "\n".join(path.read_text().lower() for path in (root / "llm" / "providers").glob("*.py"))

    assert "from tools" not in agent_source
    assert "import tools" not in agent_source
    assert "google" not in evaluator_source
    assert "openai" not in evaluator_source
    assert "google" not in registry_source
    assert "openai" not in registry_source
    assert "from llm" not in deterministic_tools
    assert "import llm" not in deterministic_tools
    assert "from tools" not in provider_sources
    assert "import tools" not in provider_sources
