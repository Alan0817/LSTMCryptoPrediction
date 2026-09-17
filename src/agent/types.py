"""JSON-safe result types for financial-analysis requests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FinancialAnalysisResult:
    """The final model response plus inspectable, provider-independent execution data."""

    answer: str
    tools_used: list[str]
    trace: list[dict]
    limitations: list[dict]

    def to_dict(self) -> dict:
        """Return only standard JSON-compatible Python containers and scalars."""
        return {
            "answer": self.answer,
            "tools_used": self.tools_used,
            "trace": self.trace,
            "limitations": self.limitations,
        }
