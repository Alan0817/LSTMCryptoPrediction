"""JSON-safe values returned by semantic retrieval."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    ticker: str
    company: str
    document_type: str
    filing_date: str
    period_end: str | None
    section: str
    section_title: str
    text: str
    source: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)
