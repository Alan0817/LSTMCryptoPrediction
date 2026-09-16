"""Deterministic, JSON-safe adapters for the project's financial capabilities."""

from .lstm_prediction import get_lstm_prediction
from .market_data import get_market_data
from .market_analysis import analyze_market
from .financial_documents import search_financial_documents
from .registry import ToolDefinition, ToolRegistry
from .risk_metrics import get_risk_metrics
from .technical_analysis import get_technical_indicators

__all__ = [
    "get_lstm_prediction",
    "get_market_data",
    "analyze_market",
    "search_financial_documents",
    "get_risk_metrics",
    "get_technical_indicators",
    "ToolDefinition",
    "ToolRegistry",
]
