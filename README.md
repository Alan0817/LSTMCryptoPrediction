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
`gemini`, and is not called by the LSTM, trading, or backtest pipeline. Select
`LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini`; model selection remains
environment-driven through `LLM_MODEL` or a provider-specific model variable.

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
Gemini and OpenAI provider adapters both translate the same provider-neutral
registry definitions into their SDK-specific function-tool format. Every actual
function execution still passes through `ToolRegistry`.

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
by completed tools.

```text
FinancialAnalysisAgent
        |
    LLMClient
    /       \\
Gemini    OpenAI
    \\       /
   ToolRegistry
        |
   analyze_market
        |
deterministic pipeline
```

## Agent Evaluation And Architecture Audit
The offline `evaluation` package contains eight representative agent cases for
conceptual questions, market analysis, risk metrics, market data, unsupported
LSTM symbols, unavailable data capabilities, and recommendation-style requests.
It measures trace-derived required-tool recall, forbidden and extra calls,
unnecessary calls, exact duplicate requests, call/round counts, and structured
limitation preservation. It deliberately does not score natural-language
quality, factual prose, or investment advice with an LLM.

The observed `analyze_market -> get_market_data` pattern is represented as an
allowed but extra call in the full-analysis case. This records possible
redundancy without preventing a provider from requesting additional evidence.
Captured results can be evaluated offline later for Gemini or OpenAI:

```bash
PYTHONPATH=src python -m evaluation.agent_evaluator captured_results.json
```

Current traces include requested tool names and arguments, completed tool
results, and round numbers. Results are JSON-safe, but full tool results can
grow with payload size; a future trace design may split execution metadata,
structured evidence, and debug payloads. The agent needs the current structured
results to preserve deterministic limitations.

### Provider Tool-Calling Boundary
`FinancialAnalysisAgent` depends only on the provider-neutral `LLMClient`
interface and `ToolRegistry`; no Gemini types leak into the agent or registry.
The Gemini and OpenAI adapters each translate the existing tool definitions,
execute requests through `ToolRegistry`, and emit the same trace event contract.
This keeps future provider additions below the application and deterministic
financial layers.

## Cross-Provider Benchmark
`evaluation.provider_benchmark` runs the same eight financial-agent cases once
for a configured provider, saves final answers and provider-neutral traces, and
passes successful results to the deterministic evaluator. Artifacts are written
under the ignored `evaluation_results/` directory and include provider, model,
benchmark version, UTC timestamp, case IDs, default maximum tool rounds,
aggregate metrics, per-case records, and recorded failures. They never include
environment values or credentials.

```bash
PYTHONPATH=src python -m evaluation.provider_benchmark --provider gemini
PYTHONPATH=src python -m evaluation.provider_benchmark --provider openai --model gpt-5.6-luna
```

The runner measures required-tool recall, missing/forbidden/extra/unnecessary
calls, exact duplicate calls, tool-call and round counts, and structured
limitation preservation. It does not measure prose quality, unrestricted factual
knowledge, semantic redundancy of allowed extra calls, investment quality, or
provider intelligence. Live model outputs can vary across runs despite identical
case prompts and system instructions.

The first one-run comparison used `gemini-3.6-flash` and `gpt-5.6-luna`:

| Metric | Gemini | OpenAI |
| --- | ---: | ---: |
| Attempted cases | 8 | 8 |
| Completed cases | 6 | 8 |
| Required-tool recall (completed cases) | 1.000 | 0.875 |
| Missing required tools | 0 | 1 |
| Forbidden calls | 0 | 0 |
| Extra calls | 1 | 0 |
| Unnecessary calls | 1 | 0 |
| Duplicate calls | 0 | 0 |
| Tool calls / rounds | 5 / 5 | 3 / 3 |
| Limitations preserved | 1 / 1 | 0 / 1 |

Gemini completed the full BTC analysis with `analyze_market`; OpenAI did the
same in this run. Gemini additionally called `analyze_market` for the direct
ETH-LSTM request, which was measured as extra and unnecessary. For the
unsupported-symbol analysis, Gemini called `analyze_market` and preserved its
structured limitation; OpenAI made no tool call, producing the one missing
required-tool and limitation-preservation failure. Gemini's Apple P/E and
recommendation cases were recorded as free-tier rate-limit failures, not treated
as successful no-tool responses. These are observations from one run, not a
provider ranking.

## SEC Financial-Document Corpus
Phase 3.0 adds deterministic SEC EDGAR ingestion only; embeddings, retrieval,
and RAG generation are not implemented yet.

```text
Financial Research Agent
          |
    future RAG tool
          |
   retrieval layer
          |
   chunked corpus
          |
    SEC ingestion
          |
      SEC EDGAR
```

The initial controlled corpus configuration contains `NVDA`, `AAPL`, and
`MSTR`, with official `10-K` and `10-Q` primary HTML filings. NVDA offers an
AI/semiconductor research case, AAPL a large diversified technology-company
case, and MSTR a bridge between corporate research and Bitcoin exposure. These
are configuration entries, not parsing branches: additional SEC-reporting
companies can be added through `CompanyIdentity` data. The existing LSTM remains
`BTC-USD` only and is unrelated to SEC-document applicability.

`SECClient` discovers filings from SEC submissions metadata using CIK as the
stable identifier, requires an identifiable `SEC_USER_AGENT`, and downloads
official primary filing HTML. The pipeline removes non-content markup, preserves
paragraphs and basic pipe-delimited table rows, detects Part/Item headings
best-effort, chunks within sections, and writes duplicate-safe JSONL documents
and chunks under ignored `data/financial_documents/`. Raw HTML is cached under
ignored `data/sec_filings/raw/`.

Section detection is intentionally heuristic: formatting and table-of-contents
patterns vary across filings, so unresolved text is retained as `UNKNOWN` rather
than dropped. Table cells are preserved as text but are not given advanced
financial-table interpretation. Corpus validation checks IDs, metadata, source
URLs, chunk references, contiguous indices, document types, and empty text.

# Results
![Alt Text](src/plots/cumulative_comparison.png)
