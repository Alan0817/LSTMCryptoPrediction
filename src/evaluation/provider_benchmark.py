"""Run the provider-neutral financial-agent benchmark and save JSON artifacts."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .agent_cases import AGENT_EVALUATION_CASES, AgentEvaluationCase
from .agent_evaluator import evaluate_case, evaluate_results
from .types import AgentCaseEvaluation, EvaluationSummary


BENCHMARK_VERSION = "phase-2.6-eight-cases"
DEFAULT_MAX_TOOL_ROUNDS = 5


@dataclass(frozen=True)
class BenchmarkCaseRecord:
    """One preserved agent result and its deterministic evaluation, if execution succeeded."""

    case_id: str
    prompt: str
    answer: str | None
    tools_used: list[str]
    trace: list[dict]
    limitations: list[dict]
    evaluation: AgentCaseEvaluation | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "tools_used": self.tools_used,
            "trace": self.trace,
            "limitations": self.limitations,
            "evaluation": self.evaluation.to_dict() if self.evaluation is not None else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenchmarkRun:
    """JSON-safe metadata, records, and deterministic aggregates for one provider run."""

    provider: str
    model: str | None
    benchmark_version: str
    timestamp: str
    max_tool_rounds: int
    case_ids: list[str]
    aggregate: EvaluationSummary
    records: list[BenchmarkCaseRecord]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "benchmark_version": self.benchmark_version,
            "timestamp": self.timestamp,
            "max_tool_rounds": self.max_tool_rounds,
            "case_ids": self.case_ids,
            "case_count": len(self.records),
            "successful_cases": self.aggregate.cases,
            "aggregate": self.aggregate.to_dict(),
            "failed_cases": [record.case_id for record in self.records if record.error is not None],
            "records": [record.to_dict() for record in self.records],
        }


def run_benchmark(
    agent,
    cases: tuple[AgentEvaluationCase, ...] | list[AgentEvaluationCase] = AGENT_EVALUATION_CASES,
    provider_label: str = "unspecified",
    model: str | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    timestamp: datetime | None = None,
) -> BenchmarkRun:
    """Run every supplied case once; provider labels are metadata, not control flow."""
    if not hasattr(agent, "run"):
        raise TypeError("agent must provide run(prompt).")
    if not isinstance(provider_label, str) or not provider_label.strip():
        raise ValueError("provider_label must be a non-empty string.")
    if not isinstance(max_tool_rounds, int) or max_tool_rounds <= 0:
        raise ValueError("max_tool_rounds must be a positive integer.")

    records = []
    successful_pairs = []
    for case in cases:
        try:
            result = agent.run(case.prompt)
        except Exception as error:  # Records a failed live case without concealing it.
            records.append(
                BenchmarkCaseRecord(
                    case_id=case.id,
                    prompt=case.prompt,
                    answer=None,
                    tools_used=[],
                    trace=[],
                    limitations=[],
                    evaluation=None,
                    error=_safe_error(error),
                )
            )
            continue

        evaluation = evaluate_case(case, result)
        successful_pairs.append((case, result))
        records.append(
            BenchmarkCaseRecord(
                case_id=case.id,
                prompt=case.prompt,
                answer=result.answer,
                tools_used=result.tools_used,
                trace=result.trace,
                limitations=result.limitations,
                evaluation=evaluation,
            )
        )

    resolved_model = model if model is not None else getattr(getattr(agent, "_llm_client", None), "model", None)
    timestamp = timestamp or datetime.now(timezone.utc)
    return BenchmarkRun(
        provider=provider_label,
        model=resolved_model,
        benchmark_version=BENCHMARK_VERSION,
        timestamp=timestamp.isoformat(),
        max_tool_rounds=max_tool_rounds,
        case_ids=[case.id for case in cases],
        aggregate=evaluate_results(successful_pairs),
        records=records,
    )


def write_benchmark_artifact(run: BenchmarkRun, output_directory: str | Path = "evaluation_results") -> Path:
    """Write one JSON-safe benchmark artifact without environment or credential data."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = run.timestamp.replace("+00:00", "Z").replace(":", "").replace("-", "")
    provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.provider)
    path = output_directory / "{}_{}.json".format(provider, timestamp)
    path.write_text(json.dumps(run.to_dict(), indent=2) + "\n")
    return path


def _safe_error(error: Exception) -> str:
    """Keep failure diagnostics useful without serializing likely credential-shaped tokens."""
    message = "{}: {}".format(type(error).__name__, error)
    return re.sub(r"\b(?:sk|AIza)[-_A-Za-z0-9]+\b", "[REDACTED]", message)


def main() -> None:
    """Run all eight cases once: ``python -m evaluation.provider_benchmark --provider gemini``."""
    parser = argparse.ArgumentParser(description="Run the financial-agent benchmark once for one configured provider.")
    parser.add_argument("--provider", required=True, choices=("gemini", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--output-directory", default="evaluation_results")
    arguments = parser.parse_args()

    # These application-layer imports belong only in the optional live CLI path.
    from agent.financial_agent import FinancialAnalysisAgent
    from llm.client import LLMClient
    from tools.registry import ToolRegistry

    client = LLMClient(provider=arguments.provider, model=arguments.model)
    run = run_benchmark(
        FinancialAnalysisAgent(client, ToolRegistry()),
        provider_label=arguments.provider,
        model=client.model,
    )
    path = write_benchmark_artifact(run, arguments.output_directory)
    print(json.dumps({"artifact": str(path), "aggregate": run.aggregate.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
