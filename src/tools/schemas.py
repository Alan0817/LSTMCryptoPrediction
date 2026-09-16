"""Provider-neutral JSON-schema-compatible definitions for exposed tools."""


MARKET_DATA_PARAMETERS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Market ticker symbol, for example BTC-USD.",
            "minLength": 1,
        },
        "start_date": {
            "type": "string",
            "format": "date",
            "description": "Inclusive requested start date in YYYY-MM-DD format.",
        },
        "end_date": {
            "type": "string",
            "format": "date",
            "description": "Exclusive requested end date in YYYY-MM-DD format.",
        },
    },
    "required": ["symbol", "start_date", "end_date"],
    "additionalProperties": False,
}


RISK_METRICS_PARAMETERS = {
    "type": "object",
    "properties": {
        "returns": {
            "type": "array",
            "description": "Daily decimal returns, limited to 1,000 values for a concise tool request.",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 1000,
        },
    },
    "required": ["returns"],
    "additionalProperties": False,
}


MARKET_ANALYSIS_PARAMETERS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Market ticker symbol, for example BTC-USD.",
            "minLength": 1,
        },
        "start_date": {
            "type": "string",
            "format": "date",
            "description": "Inclusive requested start date in YYYY-MM-DD format.",
        },
        "end_date": {
            "type": "string",
            "format": "date",
            "description": "Exclusive requested end date in YYYY-MM-DD format.",
        },
    },
    "required": ["symbol", "start_date", "end_date"],
    "additionalProperties": False,
}


FINANCIAL_DOCUMENT_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Question to search in the configured local SEC filing corpus.", "minLength": 1},
        "top_k": {"type": "integer", "description": "Maximum evidence chunks to return, from 1 to 10.", "minimum": 1, "maximum": 10},
        "ticker": {"type": "string", "description": "Optional ticker filter in the configured SEC corpus.", "minLength": 1},
        "document_type": {"type": "string", "description": "Optional SEC filing form filter: 10-K or 10-Q.", "minLength": 1},
        "section": {"type": "string", "description": "Optional normalized filing section filter.", "minLength": 1},
        "filing_date_from": {"type": "string", "format": "date", "description": "Optional inclusive filing date in YYYY-MM-DD."},
        "filing_date_to": {"type": "string", "format": "date", "description": "Optional inclusive filing date in YYYY-MM-DD."},
    },
    "required": ["query"],
    "additionalProperties": False,
}
WEB_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Current-information web search query.",
            "minLength": 1,
        },
        "max_results": {
            "type": "integer",
            "description": "Results to return, from 1 to 10.",
            "minimum": 1,
            "maximum": 10,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}
