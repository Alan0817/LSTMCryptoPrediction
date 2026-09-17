"""Deterministic composite analysis over the project's market and LSTM tools."""

import json

import pandas as pd

from config import SEQUENCE_LENGTH
from data_processing import download_market_data, engineer_features, validate_ohlcv_data
from prediction import predict_features

from .lstm_prediction import TARGET_DEFINITION, get_lstm_prediction
from .market_data import get_market_data, validate_date_range
from .risk_metrics import get_risk_metrics
from .technical_analysis import get_technical_indicators_from_features


SUPPORTED_MODEL_SYMBOLS = frozenset({"BTC-USD"})


def analyze_market(
    symbol: str,
    start_date: str,
    end_date: str,
    downloader=download_market_data,
    predictor=predict_features,
    artifact_path=None,
    feature_engineer=engineer_features,
    risk_metrics_calculator=get_risk_metrics,
    model_sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """Return a compact, JSON-safe market analysis without exposing tabular data.

    Dependency parameters are application-controlled testing and deployment hooks.
    Only ``symbol``, ``start_date``, and ``end_date`` are suitable for an LLM-facing
    schema.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string.")
    if not isinstance(model_sequence_length, int) or model_sequence_length <= 0:
        raise ValueError("model_sequence_length must be a positive integer.")

    start, end = validate_date_range(start_date, end_date)
    raw_data = downloader(
        symbol=symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
    )
    validate_ohlcv_data(raw_data)

    # Reuse the established summary adapter while retaining the one downloaded frame.
    market = get_market_data(
        symbol,
        start.date().isoformat(),
        end.date().isoformat(),
        downloader=lambda **_kwargs: raw_data,
    )
    feature_data = feature_engineer(raw_data)
    technical_analysis = get_technical_indicators_from_features(feature_data)
    market_returns = raw_data["Close"].pct_change().dropna().to_list()
    risk = risk_metrics_calculator(market_returns)

    lstm_prediction = _lstm_result(
        symbol=symbol,
        feature_data=feature_data,
        predictor=predictor,
        artifact_path=artifact_path,
        model_sequence_length=model_sequence_length,
    )
    result = {
        "symbol": symbol,
        "period": {
            "requested": market["requested_date_range"],
            "available": market["available_date_range"],
        },
        "market": {
            "observations": market["observations"],
            "latest_ohlcv": market["latest_ohlcv"],
            "recent_returns": market["recent_returns"],
            "period_return": float(raw_data["Close"].iloc[-1] / raw_data["Close"].iloc[0] - 1),
        },
        "technical_analysis": technical_analysis,
        "lstm_prediction": lstm_prediction,
        "risk": risk,
        "limitations": _limitations(symbol),
    }
    # Keep the public boundary honest even when an injected dependency is careless.
    try:
        json.dumps(result)
    except (TypeError, ValueError) as error:
        raise TypeError("Market analysis returned a non-JSON-serializable result.") from error
    return result


def _lstm_result(
    symbol: str,
    feature_data: pd.DataFrame,
    predictor,
    artifact_path,
    model_sequence_length: int,
) -> dict:
    if symbol.upper() not in SUPPORTED_MODEL_SYMBOLS:
        return {
            "status": "not_applicable",
            "reason": "The available LSTM artifact was trained for BTC-USD.",
        }
    if len(feature_data) <= model_sequence_length:
        return {
            "status": "insufficient_history",
            "reason": "The requested period has fewer than {} engineered observations required by the LSTM.".format(
                model_sequence_length + 1
            ),
        }
    try:
        prediction = get_lstm_prediction(
            feature_data,
            artifact_path=artifact_path,
            predictor=predictor,
        )
    except FileNotFoundError as error:
        return {"status": "artifact_unavailable", "reason": str(error)}
    except ValueError as error:
        if "Insufficient feature history" in str(error):
            return {"status": "insufficient_history", "reason": str(error)}
        raise

    return {
        "status": "available",
        **prediction,
        "probability_up_definition": "P(Target = 1)",
        "raw_signal_is_exposure": False,
    }


def _limitations(symbol: str) -> list[dict]:
    return [
        {
            "code": "model_symbol_applicability",
            "supported_symbols": sorted(SUPPORTED_MODEL_SYMBOLS),
            "applicable": symbol.upper() in SUPPORTED_MODEL_SYMBOLS,
        },
        {"code": "target_definition", "value": TARGET_DEFINITION},
        {"code": "raw_signal_is_exposure", "value": False},
        {"code": "investment_recommendation", "value": False},
    ]
