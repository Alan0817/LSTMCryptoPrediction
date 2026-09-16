"""JSON-safe adapter for controlled SEC filing retrieval."""

from typing import Protocol


class DocumentRetriever(Protocol):
    """The small retriever surface required by the application-facing tool."""

    def search(self, query: str, top_k: int = 5, ticker: str | None = None, document_type: str | None = None,
               section: str | None = None, filing_date_from: str | None = None,
               filing_date_to: str | None = None) -> list: ...


def search_financial_documents(
    query: str,
    top_k: int = 5,
    ticker: str | None = None,
    document_type: str | None = None,
    section: str | None = None,
    filing_date_from: str | None = None,
    filing_date_to: str | None = None,
    retriever: DocumentRetriever | None = None,
) -> dict:
    """Return compact SEC evidence, while keeping retrieval dependencies controlled."""
    filters = {
        "ticker": ticker,
        "document_type": document_type,
        "section": section,
        "filing_date_from": filing_date_from,
        "filing_date_to": filing_date_to,
    }
    if retriever is None:
        return {
            "status": "unavailable",
            "query": query,
            "filters": filters,
            "result_count": 0,
            "results": [],
            "limitations": [{"code": "document_retrieval_unavailable", "reason": "No local SEC retriever is configured."}],
        }
    try:
        results = retriever.search(query, top_k=top_k, **filters)
    except (TypeError, ValueError) as error:
        return {
            "status": "invalid_request",
            "query": query,
            "filters": filters,
            "result_count": 0,
            "results": [],
            "limitations": [{"code": "document_retrieval_invalid_request", "reason": str(error)}],
        }
    except Exception:
        return {
            "status": "retrieval_error",
            "query": query,
            "filters": filters,
            "result_count": 0,
            "results": [],
            "limitations": [{"code": "document_retrieval_error", "reason": "Local SEC retrieval could not complete."}],
        }
    serialized = [result.to_dict() if hasattr(result, "to_dict") else result for result in results]
    status = "ok" if serialized else "no_results"
    limitations = []
    if status == "no_results":
        limitations.append({"code": "document_retrieval_no_results", "reason": "No configured corpus chunks matched this query and filters."})
    return {
        "status": status,
        "query": query,
        "filters": filters,
        "result_count": len(serialized),
        "results": serialized,
        "limitations": limitations,
    }
