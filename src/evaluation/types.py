"""JSON-safe types for deterministic financial-agent evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentCaseEvaluation:
    """Trace-derived measurements for one benchmark case, not a prose-quality score."""

    case_id: str
    required_tool_recall: float
    missing_required_tools: list[str]
    forbidden_tool_calls: list[str]
    extra_tool_calls: list[str]
    unnecessary_tool_calls: list[str]
    duplicate_tool_calls: int
    tool_call_count: int
    tool_round_count: int
    limitation_preservation: bool | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate trace metrics across a set of benchmark results."""

    cases: int
    required_tool_successes: int
    required_tool_recall: float
    forbidden_tool_calls: int
    extra_tool_calls: int
    unnecessary_tool_calls: int
    duplicate_tool_calls: int
    tool_call_count: int
    tool_round_count: int
    limitation_cases: int
    limitation_preservation_successes: int
    evaluations: list[AgentCaseEvaluation]

    def to_dict(self) -> dict:
        return {
            "cases": self.cases,
            "required_tool_successes": self.required_tool_successes,
            "required_tool_recall": self.required_tool_recall,
            "forbidden_tool_calls": self.forbidden_tool_calls,
            "extra_tool_calls": self.extra_tool_calls,
            "unnecessary_tool_calls": self.unnecessary_tool_calls,
            "duplicate_tool_calls": self.duplicate_tool_calls,
            "tool_call_count": self.tool_call_count,
            "tool_round_count": self.tool_round_count,
            "limitation_cases": self.limitation_cases,
            "limitation_preservation_successes": self.limitation_preservation_successes,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }
