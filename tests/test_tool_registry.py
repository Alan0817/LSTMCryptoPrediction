import json

import numpy as np
import pandas as pd
import pytest

from tools.registry import ToolRegistry


def make_ohlcv(rows=4):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.arange(100.0, 100.0 + rows)
    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 1.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(rows, 1_000.0),
        },
        index=index,
    )


def test_registry_lists_and_looks_up_exposed_tools():
    registry = ToolRegistry()

    assert [tool.name for tool in registry.list_tools()] == [
        "get_market_data",
        "get_risk_metrics",
        "analyze_market",
    ]
    assert registry.get("get_market_data").name == "get_market_data"


def test_registry_rejects_unknown_tool():
    with pytest.raises(KeyError, match="Unknown tool"):
        ToolRegistry().get("get_lstm_prediction")


def test_registry_executes_market_data_with_controlled_downloader():
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        return make_ohlcv()

    result = ToolRegistry(market_data_downloader=fake_downloader).execute(
        "get_market_data",
        {"symbol": "BTC-USD", "start_date": "2024-01-01", "end_date": "2024-02-01"},
    )

    assert calls == [{"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-02-01"}]
    assert result["symbol"] == "BTC-USD"
    json.dumps(result)


def test_registry_executes_risk_metrics_with_json_safe_result():
    result = ToolRegistry().execute("get_risk_metrics", {"returns": [0.01, -0.02, 0.03]})

    assert result["observations"] == 3
    json.dumps(result)


def test_registry_rejects_missing_required_argument():
    with pytest.raises(ValueError, match="Missing required tool arguments: end_date"):
        ToolRegistry().execute(
            "get_market_data",
            {"symbol": "BTC-USD", "start_date": "2024-01-01"},
        )


def test_registry_rejects_unexpected_or_internal_argument():
    with pytest.raises(ValueError, match="Unexpected tool arguments: downloader"):
        ToolRegistry().execute(
            "get_market_data",
            {
                "symbol": "BTC-USD",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "downloader": "untrusted",
            },
        )


def test_registry_rejects_wrong_argument_type_without_coercion():
    with pytest.raises(TypeError, match="returns\\[1\\].*number"):
        ToolRegistry().execute("get_risk_metrics", {"returns": [0.01, "-0.02"]})


def test_schemas_expose_only_agent_facing_arguments_and_are_provider_neutral():
    definitions = [tool.as_dict() for tool in ToolRegistry().list_tools()]
    market_schema = next(tool["parameters"] for tool in definitions if tool["name"] == "get_market_data")
    risk_schema = next(tool["parameters"] for tool in definitions if tool["name"] == "get_risk_metrics")
    analysis_schema = next(tool["parameters"] for tool in definitions if tool["name"] == "analyze_market")
    schema_text = json.dumps(definitions).lower()

    assert set(market_schema["properties"]) == {"symbol", "start_date", "end_date"}
    assert set(risk_schema["properties"]) == {"returns"}
    assert set(analysis_schema["properties"]) == {"symbol", "start_date", "end_date"}
    assert "downloader" not in schema_text
    assert "predictor" not in schema_text
    assert "artifact_path" not in schema_text
    assert "historical_context" not in schema_text
    assert "market_analysis_predictor" not in schema_text
    assert "dataframe" not in schema_text
    assert "openai" not in schema_text
    assert "gemini" not in schema_text


def test_dataframe_tools_are_not_registered():
    names = {tool.name for tool in ToolRegistry().list_tools()}

    assert "get_technical_indicators" not in names
    assert "get_lstm_prediction" not in names
