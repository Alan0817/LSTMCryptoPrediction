import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.registry import ToolRegistry
from tools.financial_documents import search_financial_documents


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


class FakeDocumentResult:
    def to_dict(self):
        return {
            "rank": 1, "score": 0.65, "chunk_id": "chunk_sec", "document_id": "sec_doc",
            "ticker": "MSTR", "company": "Strategy Inc.", "document_type": "10-K",
            "filing_date": "2026-02-19", "period_end": "2025-12-31", "section": "PART I ITEM 1A",
            "section_title": "Risk Factors", "text": "Bitcoin custody risk evidence.",
            "source": "SEC EDGAR", "source_url": "https://www.sec.gov/example",
        }


class FakeDocumentRetriever:
    def __init__(self, results=None, error=None):
        self.results = [] if results is None else results
        self.error = error
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.error:
            raise self.error
        return self.results


def test_document_tool_is_capability_gated_and_exposes_only_json_parameters():
    assert "search_financial_documents" not in {tool.name for tool in ToolRegistry().list_tools()}
    registry = ToolRegistry(document_retriever=FakeDocumentRetriever())
    definition = registry.get("search_financial_documents")

    assert set(definition.parameters_schema["properties"]) == {
        "query", "top_k", "ticker", "document_type", "section", "filing_date_from", "filing_date_to"
    }
    schema = json.dumps(definition.as_dict()).lower()
    assert "retriever" not in schema and "sentence" not in schema and "vector" not in schema
    assert "openai" not in schema and "gemini" not in schema


def test_document_search_forwards_filters_preserves_provenance_and_bounds_results():
    retriever = FakeDocumentRetriever([FakeDocumentResult()])
    result = ToolRegistry(document_retriever=retriever).execute(
        "search_financial_documents",
        {
            "query": "What custody risks exist?", "top_k": 1, "ticker": "MSTR", "document_type": "10-K",
            "section": "PART I ITEM 1A", "filing_date_from": "2026-01-01", "filing_date_to": "2026-12-31",
        },
    )

    assert retriever.calls == [{
        "query": "What custody risks exist?", "top_k": 1, "ticker": "MSTR", "document_type": "10-K",
        "section": "PART I ITEM 1A", "filing_date_from": "2026-01-01", "filing_date_to": "2026-12-31",
    }]
    assert result["status"] == "ok" and result["result_count"] == 1
    assert result["results"][0]["source_url"] == "https://www.sec.gov/example"
    json.dumps(result)


@pytest.mark.parametrize("arguments", [
    {"query": "risk", "top_k": 0}, {"query": "risk", "top_k": 11}, {"query": "risk", "top_k": "5"},
    {"query": "risk", "retriever": "untrusted"},
])
def test_document_search_schema_rejects_invalid_or_internal_arguments(arguments):
    with pytest.raises((TypeError, ValueError)):
        ToolRegistry(document_retriever=FakeDocumentRetriever()).execute("search_financial_documents", arguments)


def test_document_search_no_results_unavailable_and_errors_are_structured():
    empty = search_financial_documents("No match", retriever=FakeDocumentRetriever())
    unavailable = search_financial_documents("No match")
    failed = search_financial_documents("No match", retriever=FakeDocumentRetriever(error=RuntimeError("disk path leaked")))

    assert empty["status"] == "no_results" and empty["results"] == []
    assert unavailable["status"] == "unavailable"
    assert failed["status"] == "retrieval_error"
    assert "disk path" not in json.dumps(failed)


def test_document_tool_does_not_construct_embedding_models():
    source = (Path(__file__).resolve().parents[1] / "src" / "tools" / "financial_documents.py").read_text()
    assert "SentenceTransformer" not in source
