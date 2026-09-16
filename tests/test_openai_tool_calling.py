import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.financial_agent import FinancialAnalysisAgent
from llm.client import LLMClient
from tools.registry import ToolDefinition, ToolRegistry
from tools.schemas import MARKET_DATA_PARAMETERS, RISK_METRICS_PARAMETERS


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class RecordingRegistry:
    def __init__(self, definitions=None):
        self.calls = []
        self.definitions = definitions or [
            ToolDefinition(
                name="get_risk_metrics",
                description="Calculate risk metrics.",
                parameters_schema=RISK_METRICS_PARAMETERS,
                handler=lambda **kwargs: {},
            )
        ]

    def list_tools(self):
        return self.definitions

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if name not in {definition.name for definition in self.definitions}:
            raise KeyError("Unknown tool: {!r}.".format(name))
        return {"tool": name, "arguments": arguments}


def function_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def response(identifier, output=None, output_text=None):
    return SimpleNamespace(id=identifier, output=output or [], output_text=output_text)


def openai_client(responses):
    return SimpleNamespace(responses=FakeResponses(responses))


def test_openai_no_tool_response_returns_final_text_and_trace():
    fake_client = openai_client([response("response-1", output_text="RSI measures momentum.")])
    trace = []

    answer = LLMClient(provider="openai", api_client=fake_client).generate_with_tools(
        "What does RSI measure?", RecordingRegistry(), system_prompt="Be concise.", trace=trace
    )

    assert answer == "RSI measures momentum."
    assert trace == [{"event": "final_response"}]
    assert fake_client.responses.requests[0]["instructions"] == "Be concise."
    assert fake_client.responses.requests[0]["tools"][0]["parameters"] == RISK_METRICS_PARAMETERS


def test_openai_executes_single_registry_tool_and_returns_output_with_original_call_id():
    fake_client = openai_client(
        [
            response("response-1", [function_call("get_risk_metrics", '{"returns": [0.01, -0.02]}', "call-abc")]),
            response("response-2", output_text="The risk metrics are ready."),
        ]
    )
    registry = RecordingRegistry()
    trace = []

    answer = LLMClient(provider="openai", api_client=fake_client).generate_with_tools(
        "Calculate risk.", registry, system_prompt="Use tools.", trace=trace
    )

    assert answer == "The risk metrics are ready."
    assert registry.calls == [("get_risk_metrics", {"returns": [0.01, -0.02]})]
    output = fake_client.responses.requests[1]["input"]
    assert output == [
        {
            "type": "function_call_output",
            "call_id": "call-abc",
            "output": '{"tool": "get_risk_metrics", "arguments": {"returns": [0.01, -0.02]}}',
        }
    ]
    assert fake_client.responses.requests[1]["previous_response_id"] == "response-1"
    assert fake_client.responses.requests[1]["instructions"] == "Use tools."
    assert trace == [
        {
            "event": "tool_requested",
            "name": "get_risk_metrics",
            "arguments": {"returns": [0.01, -0.02]},
            "round": 1,
        },
        {
            "event": "tool_completed",
            "name": "get_risk_metrics",
            "result": {"tool": "get_risk_metrics", "arguments": {"returns": [0.01, -0.02]}},
            "round": 1,
        },
        {"event": "final_response"},
    ]
    json.dumps(trace)


def test_openai_supports_multiple_sequential_tool_rounds_with_round_numbers():
    fake_client = openai_client(
        [
            response("response-1", [function_call("get_risk_metrics", '{"returns": [0.01, -0.02]}', "call-1")]),
            response("response-2", [function_call("get_risk_metrics", '{"returns": [0.03, -0.01]}', "call-2")]),
            response("response-3", output_text="Both calculations are complete."),
        ]
    )
    registry = RecordingRegistry()
    trace = []

    answer = LLMClient(provider="openai", api_client=fake_client).generate_with_tools("Calculate twice.", registry, trace=trace)

    assert answer == "Both calculations are complete."
    assert registry.calls == [
        ("get_risk_metrics", {"returns": [0.01, -0.02]}),
        ("get_risk_metrics", {"returns": [0.03, -0.01]}),
    ]
    assert [event["round"] for event in trace if event["event"] == "tool_requested"] == [1, 2]
    assert fake_client.responses.requests[2]["previous_response_id"] == "response-2"


def test_openai_supports_multiple_function_calls_in_one_round():
    definitions = [
        ToolDefinition("get_risk_metrics", "Calculate risk.", RISK_METRICS_PARAMETERS, lambda **kwargs: {}),
        ToolDefinition("get_market_data", "Get data.", MARKET_DATA_PARAMETERS, lambda **kwargs: {}),
    ]
    fake_client = openai_client(
        [
            response(
                "response-1",
                [
                    function_call("get_risk_metrics", '{"returns": [0.01, -0.02]}', "risk-call"),
                    function_call(
                        "get_market_data",
                        '{"symbol": "BTC-USD", "start_date": "2024-01-01", "end_date": "2024-02-01"}',
                        "market-call",
                    ),
                ],
            ),
            response("response-2", output_text="Both tool results are available."),
        ]
    )
    registry = RecordingRegistry(definitions)
    trace = []

    LLMClient(provider="openai", api_client=fake_client).generate_with_tools("Use both tools.", registry, trace=trace)

    assert [name for name, _ in registry.calls] == ["get_risk_metrics", "get_market_data"]
    assert [item["call_id"] for item in fake_client.responses.requests[1]["input"]] == ["risk-call", "market-call"]
    assert {event["round"] for event in trace if event["event"] == "tool_requested"} == {1}


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        ("{not json}", ValueError, "malformed JSON arguments"),
        ("[]", TypeError, "must decode to a JSON object"),
    ],
)
def test_openai_rejects_malformed_or_non_object_function_arguments(arguments, exception, message):
    fake_client = openai_client([response("response-1", [function_call("get_risk_metrics", arguments)])])

    with pytest.raises(exception, match=message):
        LLMClient(provider="openai", api_client=fake_client).generate_with_tools("Calculate risk.", RecordingRegistry())


def test_openai_propagates_unknown_tool_and_registry_validation_errors():
    unknown_client = openai_client([response("response-1", [function_call("unknown", "{}")])])
    validation_client = openai_client(
        [response("response-1", [function_call("get_risk_metrics", '{"returns": [0.01, "bad"]}')])]
    )

    with pytest.raises(KeyError, match="Unknown tool"):
        LLMClient(provider="openai", api_client=unknown_client).generate_with_tools("Unknown.", ToolRegistry())
    with pytest.raises(TypeError, match="returns\\[1\\].*number"):
        LLMClient(provider="openai", api_client=validation_client).generate_with_tools("Bad risk.", ToolRegistry())


def test_openai_tool_round_limit_and_missing_final_text_are_clear():
    limit_client = openai_client(
        [response("response-1", [function_call("get_risk_metrics", '{"returns": [0.01, -0.02]}')])]
    )
    missing_text_client = openai_client([response("response-1")])

    with pytest.raises(RuntimeError, match="maximum of 1 tool rounds"):
        LLMClient(provider="openai", api_client=limit_client).generate_with_tools(
            "Calculate risk.", RecordingRegistry(), max_tool_rounds=1
        )
    with pytest.raises(RuntimeError, match="response without text output"):
        LLMClient(provider="openai", api_client=missing_text_client).generate_with_tools(
            "Explain RSI.", RecordingRegistry()
        )


def test_financial_agent_uses_openai_trace_without_provider_specific_changes():
    fake_client = openai_client(
        [
            response("response-1", [function_call("get_risk_metrics", '{"returns": [0.01, -0.02]}')]),
            response("response-2", output_text="Risk analysis complete."),
        ]
    )
    agent = FinancialAnalysisAgent(LLMClient(provider="openai", api_client=fake_client), RecordingRegistry())

    result = agent.run("Calculate risk metrics.")

    assert result.answer == "Risk analysis complete."
    assert result.tools_used == ["get_risk_metrics"]
    assert result.trace[-1] == {"event": "final_response"}
    json.dumps(result.to_dict())


def test_openai_provider_has_no_direct_financial_implementation_imports():
    source = (Path(__file__).resolve().parents[1] / "src" / "llm" / "providers" / "openai_provider.py").read_text()

    assert "from tools" not in source
    assert "import tools" not in source
    assert "get_market_data" not in source
    assert "get_technical_indicators" not in source
    assert "get_lstm_prediction" not in source
    assert "get_risk_metrics" not in source
