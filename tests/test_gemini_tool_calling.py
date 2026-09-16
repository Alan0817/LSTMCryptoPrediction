from pathlib import Path
from types import SimpleNamespace

import pytest

from llm.client import LLMClient
from tools.registry import ToolDefinition, ToolRegistry
from tools.schemas import MARKET_ANALYSIS_PARAMETERS, RISK_METRICS_PARAMETERS


class FakeInteractions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class RecordingRegistry:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"observations": 3, "sharpe_ratio": 1.2}
        self.definition = ToolDefinition(
            name="get_risk_metrics",
            description="Calculate risk metrics.",
            parameters_schema=RISK_METRICS_PARAMETERS,
            handler=lambda **kwargs: self.result,
        )

    def list_tools(self):
        return [self.definition]

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if name != self.definition.name:
            raise KeyError("Unknown tool: {!r}.".format(name))
        return self.result


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, id=call_id)


def interaction(identifier, steps=None, output_text=None):
    return SimpleNamespace(id=identifier, steps=steps or [], output_text=output_text)


def gemini_client(responses):
    return SimpleNamespace(interactions=FakeInteractions(responses))


def test_gemini_executes_registry_tool_and_returns_final_text():
    fake_client = gemini_client(
        [
            interaction("interaction-1", [tool_call("get_risk_metrics", {"returns": [0.01, -0.02, 0.03]})]),
            interaction("interaction-2", output_text="The Sharpe ratio is 1.2."),
        ]
    )
    registry = RecordingRegistry()
    trace = []

    result = LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
        "Calculate risk metrics.", registry, system_prompt="Use tools when needed.", trace=trace
    )

    assert result == "The Sharpe ratio is 1.2."
    assert registry.calls == [("get_risk_metrics", {"returns": [0.01, -0.02, 0.03]})]
    assert fake_client.interactions.requests[0]["tools"] == [
        {
            "type": "function",
            "name": "get_risk_metrics",
            "description": "Calculate risk metrics.",
            "parameters": RISK_METRICS_PARAMETERS,
        }
    ]
    assert fake_client.interactions.requests[1]["previous_interaction_id"] == "interaction-1"
    function_result = fake_client.interactions.requests[1]["input"][0]
    assert function_result["name"] == "get_risk_metrics"
    assert '"sharpe_ratio": 1.2' in function_result["result"][0]["text"]
    assert [event["event"] for event in trace] == ["tool_requested", "tool_completed", "final_response"]
    assert trace[1]["result"] == {"observations": 3, "sharpe_ratio": 1.2}


def test_gemini_market_data_request_executes_through_registry():
    calls = []

    def fake_downloader(**kwargs):
        import pandas as pd

        calls.append(kwargs)
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.0, 101.0],
                "Volume": [1000.0, 1100.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

    fake_client = gemini_client(
        [
            interaction(
                "interaction-1",
                [tool_call("get_market_data", {"symbol": "BTC-USD", "start_date": "2024-01-01", "end_date": "2024-02-01"})],
            ),
            interaction("interaction-2", output_text="BTC-USD data is available."),
        ]
    )

    result = LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
        "Get BTC data.", ToolRegistry(market_data_downloader=fake_downloader)
    )

    assert result == "BTC-USD data is available."
    assert calls == [{"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-02-01"}]


def test_gemini_receives_market_analysis_schema_without_special_casing():
    definition = ToolDefinition(
        name="analyze_market",
        description="Create a compact analysis.",
        parameters_schema=MARKET_ANALYSIS_PARAMETERS,
        handler=lambda **kwargs: {"symbol": kwargs["symbol"]},
    )
    registry = RecordingRegistry(result={"symbol": "BTC-USD"})
    registry.definition = definition
    fake_client = gemini_client(
        [
            interaction(
                "interaction-1",
                [tool_call("analyze_market", {"symbol": "BTC-USD", "start_date": "2020-01-01", "end_date": "2025-01-01"})],
            ),
            interaction("interaction-2", output_text="Analysis complete."),
        ]
    )

    result = LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
        "Analyze BTC-USD.", registry
    )

    assert result == "Analysis complete."
    assert registry.calls[0][0] == "analyze_market"
    assert fake_client.interactions.requests[0]["tools"][0]["parameters"] == MARKET_ANALYSIS_PARAMETERS


def test_unknown_gemini_tool_request_is_rejected_by_registry():
    fake_client = gemini_client([interaction("interaction-1", [tool_call("unregistered", {})])])

    with pytest.raises(KeyError, match="Unknown tool"):
        LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
            "Call an unknown tool.", ToolRegistry()
        )


def test_malformed_gemini_tool_arguments_are_rejected_by_registry():
    fake_client = gemini_client(
        [interaction("interaction-1", [tool_call("get_risk_metrics", {"returns": [0.01, "bad"]})])]
    )

    with pytest.raises(TypeError, match="returns\\[1\\].*number"):
        LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
            "Calculate risk metrics.", ToolRegistry()
        )


def test_gemini_supports_multiple_sequential_tool_rounds():
    fake_client = gemini_client(
        [
            interaction("interaction-1", [tool_call("get_risk_metrics", {"returns": [0.01, -0.02]})]),
            interaction("interaction-2", [tool_call("get_risk_metrics", {"returns": [0.03, -0.01]}, "call-2")]),
            interaction("interaction-3", output_text="Both risk calculations are complete."),
        ]
    )
    registry = RecordingRegistry()

    result = LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
        "Calculate two metrics.", registry
    )

    assert result == "Both risk calculations are complete."
    assert len(registry.calls) == 2


def test_gemini_tool_round_limit_is_enforced():
    fake_client = gemini_client(
        [interaction("interaction-1", [tool_call("get_risk_metrics", {"returns": [0.01, -0.02]})])]
    )

    with pytest.raises(RuntimeError, match="maximum of 1 tool rounds"):
        LLMClient(provider="gemini", api_client=fake_client).generate_with_tools(
            "Calculate risk metrics.", RecordingRegistry(), max_tool_rounds=1
        )


def test_registry_modules_remain_provider_independent():
    tools_directory = Path(__file__).resolve().parents[1] / "src" / "tools"
    source = "\n".join(
        (tools_directory / filename).read_text()
        for filename in ("registry.py", "schemas.py")
    ).lower()

    assert "openai" not in source
    assert "gemini" not in source
