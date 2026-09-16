import json
from pathlib import Path

from agent.types import FinancialAnalysisResult
from evaluation.agent_cases import AGENT_EVALUATION_CASES
from evaluation.agent_evaluator import evaluate_results
from evaluation.provider_benchmark import BENCHMARK_VERSION, run_benchmark, write_benchmark_artifact


class FakeAgent:
    def __init__(self, results, model="fake-model"):
        self.results = list(results)
        self.prompts = []
        self._llm_client = type("Client", (), {"model": model})()

    def run(self, prompt):
        self.prompts.append(prompt)
        outcome = self.results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def result(tools=(), limitations=(), trace=None):
    if trace is None:
        trace = [
            {"event": "tool_requested", "name": name, "arguments": {}, "round": index + 1}
            for index, name in enumerate(tools)
        ]
        trace.append({"event": "final_response"})
    return FinancialAnalysisResult(
        answer="Fake answer.", tools_used=list(dict.fromkeys(tools)), trace=trace, limitations=list(limitations)
    )


def test_benchmark_runs_all_supplied_cases_and_preserves_same_case_definitions_for_labels():
    cases = AGENT_EVALUATION_CASES[:2]
    gemini_agent = FakeAgent([result(), result(("analyze_market",))])
    openai_agent = FakeAgent([result(), result(("analyze_market",))])

    gemini_run = run_benchmark(gemini_agent, cases, provider_label="gemini")
    openai_run = run_benchmark(openai_agent, cases, provider_label="openai")

    assert gemini_run.case_ids == openai_run.case_ids == [case.id for case in cases]
    assert gemini_agent.prompts == openai_agent.prompts == [case.prompt for case in cases]
    assert gemini_run.model == "fake-model"
    assert gemini_run.benchmark_version == BENCHMARK_VERSION
    assert gemini_run.aggregate.cases == 2


def test_benchmark_reuses_evaluator_and_marks_allowed_extra_call_without_unnecessary_penalty():
    cases = (AGENT_EVALUATION_CASES[1],)
    captured = result(("analyze_market", "get_market_data"))

    run = run_benchmark(FakeAgent([captured]), cases, provider_label="any-label", model="model-a")
    expected = evaluate_results([(cases[0], captured)])
    evaluation = run.records[0].evaluation

    assert run.aggregate.to_dict() == expected.to_dict()
    assert evaluation.extra_tool_calls == ["get_market_data"]
    assert evaluation.unnecessary_tool_calls == []
    assert run.provider == "any-label"
    assert run.model == "model-a"


def test_benchmark_records_duplicate_calls_and_limitation_preservation():
    unsupported_case = next(case for case in AGENT_EVALUATION_CASES if case.id == "unsupported_symbol_analysis")
    duplicate_trace = [
        {"event": "tool_requested", "name": "analyze_market", "arguments": {"symbol": "ETH-USD"}, "round": 1},
        {"event": "tool_requested", "name": "analyze_market", "arguments": {"symbol": "ETH-USD"}, "round": 2},
        {"event": "final_response"},
    ]
    captured = result(
        limitations=[
            {"code": "model_symbol_applicability", "applicable": False},
            {"code": "lstm_prediction_status", "status": "not_applicable"},
        ],
        trace=duplicate_trace,
    )

    run = run_benchmark(FakeAgent([captured]), (unsupported_case,), provider_label="gemini")

    assert run.records[0].evaluation.duplicate_tool_calls == 1
    assert run.records[0].evaluation.limitation_preservation is True
    assert run.aggregate.limitation_preservation_successes == 1


def test_benchmark_handles_empty_traces_and_records_failures_without_secrets():
    cases = AGENT_EVALUATION_CASES[:2]
    run = run_benchmark(
        FakeAgent([result(trace=[]), RuntimeError("OpenAI key sk-secret-token failed")]), cases, provider_label="openai"
    )

    assert run.records[0].tools_used == []
    assert run.records[0].evaluation.tool_call_count == 0
    assert run.records[1].error == "RuntimeError: OpenAI key [REDACTED] failed"
    assert run.to_dict()["case_count"] == 2
    assert run.to_dict()["successful_cases"] == 1
    assert "sk-secret-token" not in json.dumps(run.to_dict())


def test_benchmark_json_artifact_contains_metadata_and_no_credentials(tmp_path):
    run = run_benchmark(FakeAgent([result()]), AGENT_EVALUATION_CASES[:1], provider_label="gemini", model="gemini-test")

    path = write_benchmark_artifact(run, tmp_path / "artifacts")
    payload = json.loads(path.read_text())

    assert path.parent == tmp_path / "artifacts"
    assert payload["provider"] == "gemini"
    assert payload["model"] == "gemini-test"
    assert payload["case_ids"] == ["conceptual_rsi"]
    assert payload["max_tool_rounds"] == 5
    assert "OPENAI_API_KEY" not in json.dumps(payload)
    assert "GEMINI_API_KEY" not in json.dumps(payload)


def test_runner_has_no_direct_provider_imports_or_branches():
    source = (Path(__file__).resolve().parents[1] / "src" / "evaluation" / "provider_benchmark.py").read_text()

    assert "GeminiProvider" not in source
    assert "OpenAIProvider" not in source
    assert "if provider ==" not in source
    assert "elif provider ==" not in source
