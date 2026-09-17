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
    AgentEvaluationCase(
        id="mstr_custody_filings",
        prompt="What Bitcoin custody risks does MSTR disclose?",
        required_tools=("search_financial_documents",),
        forbidden_tools=("analyze_market",),
        notes="Filing-only evidence should use local SEC retrieval, not quantitative analysis.",
    ),
    AgentEvaluationCase(
        id="nvda_quantitative_market_risk",
        prompt="What was NVDA's market risk over this period?",
        required_tools=("analyze_market",),
        forbidden_tools=("search_financial_documents",),
        notes="Quantitative market analysis does not require filing retrieval.",
    ),
    AgentEvaluationCase(
        id="mstr_combined_market_and_filings",
        prompt="Analyze MSTR's market performance and explain Bitcoin-related risks disclosed in its filings.",
        required_tools=("analyze_market", "search_financial_documents"),
        notes="Combined requests require both quantitative and documentary evidence.",
    ),
    AgentEvaluationCase(
        id="unsupported_corpus_ticker",
        prompt="What SEC filing evidence is available for TSLA in the local corpus?",
        required_tools=("search_financial_documents",),
        expected_limitation_codes=("document_retrieval_no_results",),
        notes="No-result metadata, rather than fabricated filing evidence, must survive.",
    ),
    AgentEvaluationCase(
        id="conceptual_10k",
        prompt="What is a 10-K?",
        notes="Conceptual answer; documentary retrieval is not required.",
    ),
    AgentEvaluationCase(
        id="btc_mstr_model_boundary",
        prompt="Is a BTC LSTM prediction an MSTR prediction?",
        forbidden_tools=("get_lstm_prediction",),
        notes="The evaluator does not judge prose, but no cross-asset LSTM tool exists.",
    ),
)
