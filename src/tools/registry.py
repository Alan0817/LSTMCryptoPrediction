"""Provider-neutral registry for selected deterministic financial tools."""

from dataclasses import dataclass
import json
import math
from typing import Callable

from .market_data import get_market_data
from .market_analysis import analyze_market
from .financial_documents import search_financial_documents
from .risk_metrics import get_risk_metrics
from .schemas import (
    FINANCIAL_DOCUMENT_SEARCH_PARAMETERS,
    MARKET_ANALYSIS_PARAMETERS,
    MARKET_DATA_PARAMETERS,
    RISK_METRICS_PARAMETERS,
    WEB_SEARCH_PARAMETERS,
)
from web_search import search_web


@dataclass(frozen=True)
class ToolDefinition:
    """A provider-neutral description plus controlled execution handler."""

    name: str
    description: str
    parameters_schema: dict
    handler: Callable[..., dict]

    def as_dict(self) -> dict:
        """Return the provider-neutral, JSON-safe portion of the definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class ToolRegistry:
    """Validate and execute explicitly exposed deterministic tools.

    ``market_data_downloader`` is an application-controlled dependency for
    deterministic testing or alternative data sources; it is never exposed in
    an LLM-facing schema.
    """

    def __init__(
        self,
        market_data_downloader=None,
        market_analysis_predictor=None,
        market_analysis_artifact_path=None,
        document_retriever=None,
        web_search_provider=None,
    ):
        self._market_data_downloader = market_data_downloader
        self._market_analysis_predictor = market_analysis_predictor
        self._market_analysis_artifact_path = market_analysis_artifact_path
        self._document_retriever = document_retriever
        self._web_search_provider = web_search_provider
        self._tools = {
            "get_market_data": ToolDefinition(
                name="get_market_data",
                description="Get a structured daily OHLCV market-data summary for a ticker and date range.",
                parameters_schema=MARKET_DATA_PARAMETERS,
                handler=self._execute_market_data,
            ),
            "get_risk_metrics": ToolDefinition(
                name="get_risk_metrics",
                description="Calculate risk and performance metrics from a concise list of daily returns.",
                parameters_schema=RISK_METRICS_PARAMETERS,
                handler=get_risk_metrics,
            ),
            "analyze_market": ToolDefinition(
                name="analyze_market",
                description="Create a compact market, technical, LSTM, and risk analysis for one ticker and date range.",
                parameters_schema=MARKET_ANALYSIS_PARAMETERS,
                handler=self._execute_market_analysis,
            ),
        }
        if self._document_retriever is not None:
            self._tools["search_financial_documents"] = ToolDefinition(
                name="search_financial_documents",
                description=(
                    "Search configured local SEC filing chunks and return compact, cited documentary evidence. "
                    "Filing-date filters apply to SEC filing dates, not financial reporting periods."
                ),
                parameters_schema=FINANCIAL_DOCUMENT_SEARCH_PARAMETERS,
                handler=self._execute_financial_documents,
            )
        if self._web_search_provider is not None:
            self._tools["search_web"] = ToolDefinition(
                name="search_web",
                description="Search current public web information and return compact source provenance.",
                parameters_schema=WEB_SEARCH_PARAMETERS,
                handler=self._execute_web_search,
            )

    def list_tools(self) -> list[ToolDefinition]:
        """Return the explicitly exposed provider-neutral tool definitions."""
        return list(self._tools.values())

    def get(self, name: str) -> ToolDefinition:
        """Look up one tool by stable name."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError("Unknown tool: {!r}.".format(name)) from error

    def execute(self, name: str, arguments: dict) -> dict:
        """Validate, execute, and confirm JSON serialization for one tool call."""
        tool = self.get(name)
        _validate_arguments(tool.parameters_schema, arguments)
        result = tool.handler(**arguments)
        try:
            json.dumps(result)
        except (TypeError, ValueError) as error:
            raise TypeError("Tool {!r} returned a non-JSON-serializable result.".format(name)) from error
        return result

    def _execute_market_data(self, **arguments) -> dict:
        if self._market_data_downloader is None:
            return get_market_data(**arguments)
        return get_market_data(**arguments, downloader=self._market_data_downloader)

    def _execute_market_analysis(self, **arguments) -> dict:
        dependencies = {}
        if self._market_data_downloader is not None:
            dependencies["downloader"] = self._market_data_downloader
        if self._market_analysis_predictor is not None:
            dependencies["predictor"] = self._market_analysis_predictor
        if self._market_analysis_artifact_path is not None:
            dependencies["artifact_path"] = self._market_analysis_artifact_path
        return analyze_market(**arguments, **dependencies)

    def _execute_financial_documents(self, **arguments) -> dict:
        return search_financial_documents(**arguments, retriever=self._document_retriever)

    def _execute_web_search(self, **arguments):
        return search_web(**arguments, provider=self._web_search_provider)


def _validate_arguments(schema: dict, arguments: dict) -> None:
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object.")

    required = set(schema.get("required", []))
    missing = sorted(required.difference(arguments))
    if missing:
        raise ValueError("Missing required tool arguments: {}.".format(", ".join(missing)))

    properties = schema.get("properties", {})
    unexpected = sorted(set(arguments).difference(properties))
    if unexpected:
        raise ValueError("Unexpected tool arguments: {}.".format(", ".join(unexpected)))

    for name, value in arguments.items():
        _validate_value(value, properties[name], name)


def _validate_value(value, schema: dict, path: str) -> None:
    value_type = schema["type"]
    if value_type == "string":
        if not isinstance(value, str):
            raise TypeError("Tool argument {!r} must be a string.".format(path))
        if len(value) < schema.get("minLength", 0):
            raise ValueError("Tool argument {!r} is too short.".format(path))
        return

    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Tool argument {!r} must be a number.".format(path))
        if not math.isfinite(value):
            raise ValueError("Tool argument {!r} must be finite.".format(path))
        return

    if value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Tool argument {!r} must be an integer.".format(path))
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError("Tool argument {!r} is outside the allowed range.".format(path))
        return

    if value_type == "array":
        if not isinstance(value, list):
            raise TypeError("Tool argument {!r} must be an array.".format(path))
        if len(value) < schema.get("minItems", 0):
            raise ValueError("Tool argument {!r} has too few items.".format(path))
        if len(value) > schema.get("maxItems", float("inf")):
            raise ValueError("Tool argument {!r} has too many items.".format(path))
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], "{}[{}]".format(path, index))
        return

    raise ValueError("Unsupported schema type: {!r}.".format(value_type))
