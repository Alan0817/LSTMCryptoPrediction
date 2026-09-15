# LSTM-based Bitcoin Trading Strategy

## Overview
This project explores whether an LSTM can extract predictive information from historical Bitcoin market data and evaluates the resulting threshold-based backtest.

### Project Workflow
```
Data Collection
    ↓
Feature Engineering
    ↓
LSTM Prediction
    ↓
Signal Generation
    ↓
Backtesting
    ↓
Performance Evaluation
```

## Dataset
- Asset: BTC-USD
- Source: Yahoo Finance
- Period: 2020–2025
- Frequency: Daily

## Features
### Technical Indicators
- RSI
- MACD
- MACD Signal
- Moving Averages
- EMA
- Volatility
### Market Features
- Close Price
- Volume
- Returns

## Model Architecture
```
Input Sequence (30 Days)
          ↓
      LSTM
          ↓
 Fully Connected
          ↓
     Sigmoid
          ↓
 Upward Probability
 ```

## Trading Semantics
```
Model output: probability of Target = 1

Probability > 0.52 → raw threshold label +1
Probability < 0.48 → raw threshold label -1
Otherwise → raw threshold label 0

Frozen backtest exposure = -raw threshold label
Raw +1 → exposure -1
Raw -1 → exposure +1
Raw 0 → exposure 0
```

The threshold label is not itself a long or short position. The current
backtest intentionally applies contrarian exposure and calculates returns as
`exposure * next-day market return`.

## Result
| Metric       | Strategy | Buy & Hold |
| ------------ | -------- | ---------- |
| Sharpe Ratio | 1.672    | 1.547      |
| Max Drawdown | 0.306    | 0.262      |


## Findings
- Target engineering matters more than model complexity
- Financial prediction is extremely noisy
- Trading evaluation is more important than prediction accuracy

## Known Limitations
- Evaluation uses one fixed holdout period rather than walk-forward validation.
- Backtest returns exclude transaction costs and slippage.
- Reported performance does not establish that the strategy is profitable or deployable.
- Sharpe annualization uses `sqrt(252)` even though BTC trades daily.

## LLM Client
The isolated LLM client reads `LLM_PROVIDER`, `LLM_MODEL`, and provider API
keys from the environment or local `.env` file. It supports `openai` and
`gemini`, and is not called by the LSTM, trading, or backtest pipeline.

```bash
python -c "from src.llm.client import LLMClient; print(LLMClient(provider='gemini').generate('Explain RSI in one sentence.'))"
```

## Deterministic Financial Tools
The `tools` package exposes JSON-safe, directly callable adapters for market
data, project-defined technical indicators, LSTM inference, and risk metrics.
They do not call an LLM; each remains directly testable in Python.

```python
from tools.risk_metrics import get_risk_metrics

print(get_risk_metrics([0.01, -0.005, 0.02]))
```

Run direct examples with `PYTHONPATH=src` in this script-style project.

## Provider-Neutral Tool Registry
A deterministic tool is a Python adapter called directly by application code.
An LLM-callable tool definition is a deliberately selected JSON schema and
handler registered in `ToolRegistry`; it does not invoke an LLM in this phase.

The registry exposes `get_market_data(symbol, start_date, end_date)`,
`get_risk_metrics(returns)`, and high-level
`analyze_market(symbol, start_date, end_date)`. The market downloader remains
application controlled, while risk-return lists are capped at 1,000 values to
keep a future tool request concise. Technical-analysis and LSTM tools remain
direct Python adapters because they require DataFrames and controlled model
dependencies that should not be supplied by an LLM.

## Gemini Tool Calling
Gemini tool calling uses the official stateful `interactions.create` manual
function-call loop. Gemini selects from provider-neutral registry schemas, but
the application validates and executes every request through `ToolRegistry`.
OpenAI tool calling is not implemented yet.

```text
Gemini interaction -> ToolRegistry.execute -> JSON function result -> Gemini final text
```

```python
from llm.client import LLMClient
from tools.registry import ToolRegistry

trace = []
answer = LLMClient(provider="gemini").generate_with_tools(
    "Calculate risk metrics for [0.01, -0.005, 0.02]. Use a tool.",
    ToolRegistry(),
    trace=trace,
)
```

## Deterministic Market Analysis
`analyze_market(symbol, start_date, end_date)` is the high-level deterministic
tool for a compact market analysis. Its DataFrames remain inside Python: a
downloader feeds the project's feature engineering, technical-indicator summary,
BTC-specific LSTM inference when applicable, and market-return risk metrics.
The LLM-facing registry exposes only its ticker and date arguments.

```text
User / future LLM
        |
   analyze_market
        |
   Python orchestration
   /       |       \\
market     TA      LSTM
   \\       |       /
      risk/summary
```

The available LSTM artifact applies only to `BTC-USD`. Other symbols still
receive market, technical, and risk results, with an LSTM `not_applicable`
status. `probability_up` remains `P(Target = 1)`, where Target is 1 only when
`Future_Return > 0.005`; its raw signal is not trading exposure. Results are
model outputs, not investment recommendations.

## FinancialAnalysisAgent
`FinancialAnalysisAgent` is a thin application layer for natural-language
financial-analysis requests. It delegates reasoning and tool selection to a
tool-capable LLM client, while all quantitative work remains in deterministic
tools behind `ToolRegistry`.

```text
User
  |
FinancialAnalysisAgent
  |
LLMClient
  |
Gemini
  |
ToolRegistry
  |
analyze_market
  |
deterministic financial pipeline
```

```python
from agent.financial_agent import FinancialAnalysisAgent
from llm.client import LLMClient
from tools.registry import ToolRegistry

agent = FinancialAnalysisAgent(LLMClient(provider="gemini"), ToolRegistry())
result = agent.run("Analyze BTC-USD from 2024-09-01 to 2024-12-01.")
print(result.to_dict())
```

`FinancialAnalysisResult` contains the final answer, unique tool names derived
from actual trace events, the trace itself, and structured limitations reported
by completed tools. OpenAI tool-enabled generation is not implemented yet.

# Results
![Alt Text](src/plots/cumulative_comparison.png)
