"""Trace-derived document-retrieval planning diagnostics; no provider logic."""


def document_plan_metrics(trace):
    calls = [event for event in trace if event.get("event") == "tool_requested" and event.get("name") == "search_financial_documents"]
    signatures = [
        (event.get("arguments", {}).get("query"), tuple(sorted((key, value) for key, value in event.get("arguments", {}).items() if key != "query")))
        for event in calls
    ]
    return {
        "document_call_count": len(calls),
        "one_shot": len(calls) == 1,
        "refinement_attempted": len(calls) > 1,
        "identical_repeated_retrieval": len(signatures) != len(set(signatures)),
        "exceeded_document_budget": len(calls) > 2,
        "materially_changed_query": len(calls) > 1 and signatures[0][0] != signatures[1][0],
        "materially_changed_filters": len(calls) > 1 and signatures[0][1] != signatures[1][1],
    }


def web_plan_metrics(trace):
    calls = [event for event in trace if event.get("event") == "tool_requested" and event.get("name") == "search_web"]
    queries = [event.get("arguments", {}).get("query") for event in calls]
    return {
        "web_call_count": len(calls),
        "one_shot_web_search": len(calls) == 1,
        "refinement_attempted": len(calls) > 1,
        "materially_changed_query": len(calls) > 1 and queries[0] != queries[1],
        "identical_repeated_search": len(queries) != len(set(queries)),
        "exceeded_web_search_budget": len(calls) > 2,
    }
