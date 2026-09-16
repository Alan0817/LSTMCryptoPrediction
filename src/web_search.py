"""Provider-neutral current web-search abstraction with one Tavily adapter."""

from dataclasses import asdict, dataclass
import os

import requests


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str | None = None
    published_at: str | None = None

    def to_dict(self):
        return asdict(self)


class WebSearchProvider:
    def search(self, query, max_results=5):
        raise NotImplementedError


class TavilyWebSearchProvider(WebSearchProvider):
    def __init__(self, api_key=None, session=requests, timeout=15):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.session = session
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY must be set to initialize web search.")

    def search(self, query, max_results=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10.")

        response = self.session.post(
            "https://api.tavily.com/search",
            json={"api_key": self.api_key, "query": query, "max_results": max_results},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("results")
        if not isinstance(rows, list):
            raise ValueError("Web search provider returned malformed results.")

        results = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("title"), str) or not isinstance(row.get("url"), str):
                raise ValueError("Web search provider returned malformed result.")
            results.append(WebSearchResult(row["title"], row["url"], str(row.get("content", "")), row.get("source"), row.get("published_date")))
        return results


def search_web(query, max_results=5, provider=None):
    if provider is None:
        return {"status": "unavailable", "query": query, "result_count": 0, "results": [], "limitations": [{"code": "web_search_unavailable", "reason": "No web-search provider is configured."}]}
    try:
        rows = provider.search(query, max_results)
    except (TypeError, ValueError) as error:
        return {"status": "invalid_request", "query": query, "result_count": 0, "results": [], "limitations": [{"code": "web_search_invalid_request", "reason": str(error)}]}
    except Exception:
        return {"status": "search_error", "query": query, "result_count": 0, "results": [], "limitations": [{"code": "web_search_error", "reason": "Current web search could not complete."}]}

    results = [{"rank": rank, **row.to_dict()} for rank, row in enumerate(rows, 1)]
    return {"status": "ok" if results else "no_results", "query": query, "result_count": len(results), "results": results, "limitations": [] if results else [{"code": "web_search_no_results", "reason": "No current web results were returned."}]}
