"""Offline deterministic measurements over structured agent results."""

import argparse
import json
from pathlib import Path

from agent.types import FinancialAnalysisResult

from .agent_cases import AGENT_EVALUATION_CASES, AgentEvaluationCase
from .types import AgentCaseEvaluation, EvaluationSummary


def evaluate_case(case: AgentEvaluationCase, result: FinancialAnalysisResult) -> AgentCaseEvaluation:
    """Measure tool and limitation behavior without attempting to score answer prose."""
    requests = [event for event in result.trace if event.get("event") == "tool_requested"]
    tool_names = [event.get("name") for event in requests if isinstance(event.get("name"), str)]
    unique_tool_names = set(tool_names)

    missing = [name for name in case.required_tools if name not in unique_tool_names]
    required_recall = 1.0 if not case.required_tools else (len(case.required_tools) - len(missing)) / len(case.required_tools)
    forbidden = [name for name in tool_names if name in case.forbidden_tools]
    extra = [name for name in tool_names if name not in case.required_tools]
    allowed = set(case.required_tools).union(case.allowed_tools)
    unnecessary = [name for name in tool_names if name not in allowed]
    duplicates = _duplicate_request_count(requests)
    limitation_codes = {
        limitation.get("code")
        for limitation in result.limitations
        if isinstance(limitation, dict) and isinstance(limitation.get("code"), str)
    }
    limitation_preservation = (
        None
        if not case.expected_limitation_codes
        else set(case.expected_limitation_codes).issubset(limitation_codes)
    )

    return AgentCaseEvaluation(
        case_id=case.id,
        required_tool_recall=required_recall,
        missing_required_tools=missing,
        forbidden_tool_calls=forbidden,
        extra_tool_calls=extra,
        unnecessary_tool_calls=unnecessary,
        duplicate_tool_calls=duplicates,
        tool_call_count=len(tool_names),
        tool_round_count=_tool_round_count(requests),
        limitation_preservation=limitation_preservation,
    )


def evaluate_results(pairs: list[tuple[AgentEvaluationCase, FinancialAnalysisResult]]) -> EvaluationSummary:
    """Aggregate supplied offline results for one provider or a later cross-provider run."""
    evaluations = [evaluate_case(case, result) for case, result in pairs]
    limitation_evaluations = [item for item in evaluations if item.limitation_preservation is not None]
    return EvaluationSummary(
        cases=len(evaluations),
        required_tool_successes=sum(not item.missing_required_tools for item in evaluations),
        required_tool_recall=(sum(item.required_tool_recall for item in evaluations) / len(evaluations)) if evaluations else 0.0,
        forbidden_tool_calls=sum(len(item.forbidden_tool_calls) for item in evaluations),
        extra_tool_calls=sum(len(item.extra_tool_calls) for item in evaluations),
        unnecessary_tool_calls=sum(len(item.unnecessary_tool_calls) for item in evaluations),
        duplicate_tool_calls=sum(item.duplicate_tool_calls for item in evaluations),
        tool_call_count=sum(item.tool_call_count for item in evaluations),
        tool_round_count=sum(item.tool_round_count for item in evaluations),
        limitation_cases=len(limitation_evaluations),
        limitation_preservation_successes=sum(item.limitation_preservation for item in limitation_evaluations),
        evaluations=evaluations,
    )


def _duplicate_request_count(requests: list[dict]) -> int:
    seen = set()
    duplicates = 0
    for event in requests:
        name = event.get("name")
        arguments = event.get("arguments")
        try:
            signature = (name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError):
            signature = (name, repr(arguments))
        if signature in seen:
            duplicates += 1
        else:
            seen.add(signature)
    return duplicates


def _tool_round_count(requests: list[dict]) -> int:
    rounds = {event.get("round") for event in requests if isinstance(event.get("round"), int)}
    if rounds:
        return len(rounds)
    return int(bool(requests))


def _load_pairs(path: Path) -> list[tuple[AgentEvaluationCase, FinancialAnalysisResult]]:
    payload = json.loads(path.read_text())
    cases_by_id = {case.id: case for case in AGENT_EVALUATION_CASES}
    pairs = []
    for item in payload:
        case = cases_by_id[item["case_id"]]
        pairs.append((case, FinancialAnalysisResult(**item["result"])))
    return pairs


def main() -> None:
    """Evaluate captured JSON results offline: ``python -m evaluation.agent_evaluator results.json``."""
    parser = argparse.ArgumentParser(description="Evaluate captured financial-agent results without network access.")
    parser.add_argument("results_path", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(evaluate_results(_load_pairs(arguments.results_path)).to_dict(), indent=2))


if __name__ == "__main__":
    main()
