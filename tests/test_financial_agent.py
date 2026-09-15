import json
from pathlib import Path

import pytest

from agent.financial_agent import FinancialAnalysisAgent
from agent.prompts import FINANCIAL_ANALYSIS_SYSTEM_PROMPT


class FakeRegistry:
    def list_tools(self):
        return []

    def execute(self, name, arguments):
        raise AssertionError("The fake LLM owns trace construction in this unit test.")


class FakeLLMClient:
    def __init__(self, answer="Final answer.", events=None, error=None):
        self.answer = answer
        self.events = events or [{"event": "final_response"}]
        self.error = error
        self.calls = []

    def generate_with_tools(self, prompt, registry, system_prompt=None, trace=None):
        self.calls.append(
            {"prompt": prompt, "registry": registry, "system_prompt": system_prompt, "trace": trace}
        )
        if self.error is not None:
            raise self.error
        trace.extend(self.events)
        return self.answer


def test_agent_delegates_prompt_registry_and_system_instructions():
    llm_client = FakeLLMClient(answer="BTC analysis complete.")
    registry = FakeRegistry()
    agent = FinancialAnalysisAgent(llm_client, registry)

    result = agent.run("Analyze BTC-USD from 2024-09-01 to 2024-12-01.")

    assert result.answer == "BTC analysis complete."
    assert result.tools_used == []
    assert llm_client.calls[0]["prompt"] == "Analyze BTC-USD from 2024-09-01 to 2024-12-01."
    assert llm_client.calls[0]["registry"] is registry
    assert llm_client.calls[0]["system_prompt"] == FINANCIAL_ANALYSIS_SYSTEM_PROMPT
    json.dumps(result.to_dict())


def test_tools_used_are_unique_and_derived_from_actual_trace_events():
    llm_client = FakeLLMClient(
        events=[
            {"event": "tool_requested", "name": "analyze_market", "arguments": {}},
            {"event": "tool_completed", "name": "analyze_market", "result": {}},
            {"event": "tool_requested", "name": "analyze_market", "arguments": {}},
            {"event": "tool_completed", "name": "analyze_market", "result": {}},
            {"event": "tool_requested", "name": "get_risk_metrics", "arguments": {}},
            {"event": "final_response"},
        ]
    )

    result = FinancialAnalysisAgent(llm_client, FakeRegistry()).run("Compare two analyses.")

    assert result.tools_used == ["analyze_market", "get_risk_metrics"]


def test_agent_preserves_structured_tool_limitations_without_parsing_answer_text():
    structured_result = {
        "lstm_prediction": {
            "status": "not_applicable",
            "reason": "The available LSTM artifact was trained for BTC-USD.",
        },
        "limitations": [
            {"code": "raw_signal_is_exposure", "value": False},
            {"code": "investment_recommendation", "value": False},
        ],
    }
    llm_client = FakeLLMClient(
        answer="No limitations are mentioned in this answer.",
        events=[
            {"event": "tool_requested", "name": "analyze_market", "arguments": {}},
            {"event": "tool_completed", "name": "analyze_market", "result": structured_result},
            {"event": "final_response"},
        ],
    )

    result = FinancialAnalysisAgent(llm_client, FakeRegistry()).run("Analyze ETH-USD.")

    assert result.limitations == [
        {"code": "raw_signal_is_exposure", "value": False},
        {"code": "investment_recommendation", "value": False},
        {
            "code": "lstm_prediction_status",
            "status": "not_applicable",
            "reason": "The available LSTM artifact was trained for BTC-USD.",
        },
    ]
    json.dumps(result.to_dict())


def test_no_tool_conceptual_response_is_supported():
    result = FinancialAnalysisAgent(
        FakeLLMClient(answer="RSI measures the speed and magnitude of recent price changes."), FakeRegistry()
    ).run("What does RSI generally measure?")

    assert result.tools_used == []
    assert result.trace == [{"event": "final_response"}]


def test_tool_capability_error_propagates_clearly():
    agent = FinancialAnalysisAgent(
        FakeLLMClient(error=NotImplementedError("Tool-enabled generation is not supported for the OpenAI provider.")),
        FakeRegistry(),
    )

    with pytest.raises(NotImplementedError, match="not supported for the OpenAI provider"):
        agent.run("Analyze BTC-USD.")


def test_agent_has_no_direct_financial_implementation_imports():
    source = (Path(__file__).resolve().parents[1] / "src" / "agent" / "financial_agent.py").read_text()

    assert "from tools" not in source
    assert "import tools" not in source
    assert "get_market_data" not in source
    assert "get_technical_indicators" not in source
    assert "get_lstm_prediction" not in source
    assert "get_risk_metrics" not in source
