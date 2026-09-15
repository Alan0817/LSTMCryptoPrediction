"""Application-level coordination of the LLM facade and tool registry."""

from .prompts import FINANCIAL_ANALYSIS_SYSTEM_PROMPT
from .types import FinancialAnalysisResult


class FinancialAnalysisAgent:
    """Run one financial-analysis request through a tool-capable LLM client.

    This class deliberately has no access to market-data, indicator, LSTM, or risk
    implementations. The supplied registry remains the execution allowlist.
    """

    def __init__(self, llm_client, registry):
        if not hasattr(llm_client, "generate_with_tools"):
            raise TypeError("llm_client must provide generate_with_tools().")
        if not hasattr(registry, "list_tools") or not hasattr(registry, "execute"):
            raise TypeError("registry must provide list_tools() and execute().")
        self._llm_client = llm_client
        self._registry = registry

    def run(self, prompt: str) -> FinancialAnalysisResult:
        """Return a final answer with the actual tool trace and structured limits."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        trace: list[dict] = []
        answer = self._llm_client.generate_with_tools(
            prompt,
            self._registry,
            system_prompt=FINANCIAL_ANALYSIS_SYSTEM_PROMPT,
            trace=trace,
        )
        return FinancialAnalysisResult(
            answer=answer,
            tools_used=_tools_used(trace),
            trace=trace,
            limitations=_limitations_from_trace(trace),
        )


def _tools_used(trace: list[dict]) -> list[str]:
    """Return stable unique names from actual provider tool-request events."""
    tools_used = []
    for event in trace:
        if event.get("event") != "tool_requested":
            continue
        name = event.get("name")
        if isinstance(name, str) and name not in tools_used:
            tools_used.append(name)
    return tools_used


def _limitations_from_trace(trace: list[dict]) -> list[dict]:
    """Preserve structured limitations emitted by completed deterministic tools."""
    limitations = []
    for event in trace:
        if event.get("event") != "tool_completed" or not isinstance(event.get("result"), dict):
            continue
        result = event["result"]
        for limitation in result.get("limitations", []):
            if isinstance(limitation, dict):
                limitations.append(limitation)

        prediction = result.get("lstm_prediction")
        if isinstance(prediction, dict) and prediction.get("status") != "available":
            limitations.append(
                {
                    "code": "lstm_prediction_status",
                    "status": prediction.get("status"),
                    "reason": prediction.get("reason"),
                }
            )
    return limitations
