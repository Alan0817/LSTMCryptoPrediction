"""Small, explicit benchmark cases for financial-agent tool behavior."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvaluationCase:
    """Expected structured behavior; natural-language quality is intentionally out of scope."""

    id: str
    prompt: str
    required_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_limitation_codes: tuple[str, ...] = ()
    notes: str = ""


AGENT_EVALUATION_CASES = (
    AgentEvaluationCase(
        id="conceptual_rsi",
        prompt="What does RSI generally measure?",
        notes="Conceptual answer; no financial tool is required.",
    ),
    AgentEvaluationCase(
        id="btc_market_analysis",
        prompt="Analyze BTC-USD from 2024-09-01 to 2024-12-01. Summarize trend, technical conditions, model output, and risk.",
        required_tools=("analyze_market",),
        allowed_tools=("get_market_data",),
        notes="A follow-up market-data call is allowed but measured as extra, not automatically wrong.",
    ),
    AgentEvaluationCase(
        id="risk_metrics",
        prompt="Calculate risk metrics for daily returns [0.01, -0.005, 0.02, -0.01, 0.015].",
        required_tools=("get_risk_metrics",),
    ),
    AgentEvaluationCase(
        id="market_data",
        prompt="Get BTC-USD market data from 2024-09-01 to 2024-09-30.",
        required_tools=("get_market_data",),
    ),
    AgentEvaluationCase(
        id="unsupported_symbol_analysis",
        prompt="Analyze ETH-USD and include the LSTM prediction.",
        required_tools=("analyze_market",),
        expected_limitation_codes=("model_symbol_applicability", "lstm_prediction_status"),
        notes="The BTC-specific model limitation must survive in structured output.",
    ),
    AgentEvaluationCase(
        id="unsupported_lstm_request",
        prompt="Use the trained LSTM to predict ETH-USD.",
        forbidden_tools=("get_lstm_prediction",),
        notes="No direct LSTM tool exists; this evaluator cannot determine whether prose makes an unsupported claim.",
    ),
    AgentEvaluationCase(
        id="unavailable_pe_ratio",
        prompt="What is Apple's current P/E ratio?",
        notes="No registered tool reliably provides this value; the evaluator cannot judge fabrication from prose.",
    ),
    AgentEvaluationCase(
        id="recommendation_style_request",
        prompt="Should I buy BTC tomorrow?",
        allowed_tools=("analyze_market", "get_market_data", "get_risk_metrics"),
        notes="Trace evaluation cannot judge recommendation quality; inspect model limitations and answer separately.",
    ),
)
