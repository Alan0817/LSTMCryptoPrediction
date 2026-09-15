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
