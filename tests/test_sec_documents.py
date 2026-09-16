import json
from pathlib import Path

import pytest

from documents.chunking import chunk_document
from documents.ingestion import INITIAL_COMPANIES, ingest_company_filings
from documents.parser import parse_filing_html
from documents.sec_client import SECClient
from documents.sections import FilingSection, detect_sections
from documents.storage import CorpusStorage, RawFilingCache, validate_corpus
from documents.types import (
    CompanyIdentity,
    FilingReference,
    FinancialDocument,
    make_chunk_id,
    make_document_id,
    normalize_accession,
)


SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "10-K", "10-Q", "10-K"],
            "filingDate": ["2025-05-20", "2025-05-01", "2025-02-20", "2024-11-20", "2024-02-21"],
            "reportDate": ["2025-04-27", "", "2025-01-26", "2024-10-27", "2024-01-28"],
            "accessionNumber": [
                "0001045810-25-000050",
                "0001045810-25-000040",
                "0001045810-25-000020",
                "0001045810-24-000090",
                "0001045810-24-000010",
            ],
            "primaryDocument": ["nvda-20250427.htm", "event.htm", "nvda-20250126.htm", "nvda-q3.htm", "nvda-20240128.htm"],
        }
    }
}

SEC_HTML = """
<html><head><style>.hidden { display: none; }</style><script>ignore()</script></head>
<body><nav>navigation</nav><h1>FORM 10-K</h1>
<ix:header><ix:hidden>xbrli:context hidden taxonomy metadata</ix:hidden></ix:header>
<p>Table of Contents</p>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 1B. Unresolved Staff Comments</p>
<p>Item 2. Properties</p><p>Item 3. Legal Proceedings</p>
<h2>Item 1. Business</h2><p>We design accelerated computing platforms.</p>
<table><tr><th>Revenue</th><th>2025</th><th>2024</th></tr><tr><td>Data Center</td><td>100</td><td>50</td></tr></table>
<h2>Item 1A. Risk Factors</h2><p>Supply constraints and competition could affect results.</p>
<h2>Item 7. Management's Discussion and Analysis</h2><p>Management discusses liquidity and operations.</p>
</body></html>
"""


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self.payload = payload
        self.text = text

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeTransport:
    def __init__(self):
        self.requests = []

    def get(self, url, headers, timeout):
        self.requests.append({"url": url, "headers": headers, "timeout": timeout})
        if "submissions" in url:
            return FakeResponse(payload=SUBMISSIONS)
        return FakeResponse(text=SEC_HTML)


def make_client():
    transport = FakeTransport()
    return SECClient(user_agent="Research Team research@example.com", transport=transport), transport


def make_document():
    return FinancialDocument(
        document_id=make_document_id("320193", "0000320193-25-000010", "aapl-10k.htm"),
        company="Apple Inc.",
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000010",
        primary_document="aapl-10k.htm",
        document_type="10-K",
        filing_date="2025-11-01",
        period_end="2025-09-28",
        source="SEC EDGAR",
        source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000010/aapl-10k.htm",
        text="Example document text.",
    )


def test_sec_client_requires_identifiable_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        SECClient(transport=FakeTransport())


def test_submissions_discovery_selects_latest_10k_and_10q_and_builds_official_urls():
    client, transport = make_client()

    filings = client.find_filings("1045810", forms=("10-K", "10-Q"))

    assert {(filing.form, filing.accession_number) for filing in filings} == {
        ("10-K", "0001045810-25-000020"),
        ("10-Q", "0001045810-25-000050"),
    }
    ten_k = next(filing for filing in filings if filing.form == "10-K")
    assert client.filing_url(ten_k) == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000020/nvda-20250126.htm"
    )
    assert transport.requests[0]["headers"]["User-Agent"] == "Research Team research@example.com"


def test_discovery_supports_multiple_latest_filings_and_missing_forms_without_substitution():
    client, _ = make_client()

    filings = client.find_filings("0001045810", forms=("10-Q",), latest_per_form=2)
    absent = client.find_filings("0001045810", forms=("10-K", "10-Q"), latest_per_form=1)

    assert [filing.accession_number for filing in filings] == ["0001045810-25-000050", "0001045810-24-000090"]
    assert {filing.form for filing in absent} == {"10-K", "10-Q"}
    with pytest.raises(ValueError, match="Unsupported SEC forms"):
        client.find_filings("1045810", forms=("8-K",))


def test_accession_normalization_and_stable_document_chunk_ids():
    accession = normalize_accession("000032019325000010")
    document_id = make_document_id("320193", accession, "aapl-10k.htm")

    assert accession == "0000320193-25-000010"
    assert document_id == make_document_id("0000320193", "0000320193-25-000010", "aapl-10k.htm")
    assert make_chunk_id(document_id, "ITEM 1A", 2) == make_chunk_id(document_id, "item 1a", 2)


def test_html_parser_removes_noncontent_preserves_paragraphs_and_tables():
    text = parse_filing_html(SEC_HTML)

    assert "ignore()" not in text
    assert "navigation" not in text
    assert "xbrli:context" not in text
    assert "We design accelerated computing platforms." in text
    assert "Revenue | 2025 | 2024" in text
    assert "Data Center | 100 | 50" in text
    assert "\n\n" in text


def test_section_detection_handles_10k_toc_repetition_and_unknown_fallback():
    sections = detect_sections(parse_filing_html(SEC_HTML), "10-K")
    by_section = {section.section: section for section in sections}

    assert "ITEM 1" in by_section
    assert "ITEM 1A" in by_section
    assert "ITEM 7" in by_section
    assert by_section["ITEM 1"].text.startswith("We design")
    assert "Supply constraints" in by_section["ITEM 1A"].text
    assert detect_sections("Unstructured filing text.", "10-Q") == [
        FilingSection("UNKNOWN", "Unknown", "Unstructured filing text.")
    ]


def test_section_detection_skips_pipe_delimited_table_of_contents_rows():
    text = (
        "Part I. | Financial Information | 1\n"
        "Item 9C. | Disclosure Regarding Foreign Jurisdictions | 46\n"
        "Item 14. | Principal Accountant Fees and Services | 47\n"
        "PART I\n\nItem 1. Business\n\nActual business discussion."
    )

    sections = detect_sections(text, "10-K")

    assert [section.section for section in sections] == ["UNKNOWN", "PART I ITEM 1"]
    assert all("|" not in section.title for section in sections)


def test_10q_part_and_item_section_detection():
    text = "PART I\n\nItem 1. Financial Statements\n\nStatements here.\n\nItem 2. Management's Discussion and Analysis\n\nMD&A here.\n\nPART II\n\nItem 1A. Risk Factors\n\nRisk text."
    sections = detect_sections(text, "10-Q")

    assert any(section.section == "PART I ITEM 1" and "Statements here" in section.text for section in sections)
    assert any(section.section == "PART I ITEM 2" and "MD&A here" in section.text for section in sections)
    assert any(section.section == "PART II ITEM 1A" and "Risk text" in section.text for section in sections)


def test_section_aware_chunking_preserves_metadata_and_overlap():
    document = make_document()
    sections = [
        FilingSection("ITEM 1A", "Risk Factors", "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three."),
        FilingSection("ITEM 7", "MD&A", "Delta paragraph four."),
    ]

    chunks = chunk_document(document, sections, target_size=25, max_size=45, overlap=8)

    assert len(chunks) >= 3
    assert all(chunk.document_id == document.document_id and chunk.source_url == document.source_url for chunk in chunks)
    assert {chunk.section for chunk in chunks}.issubset({"ITEM 1A", "ITEM 7"})
    assert all(chunk.text.strip() for chunk in chunks)
    assert chunks[0].section == chunks[1].section == "ITEM 1A"
    assert chunks[1].text.startswith(chunks[0].text[-8:].lstrip())
    json.dumps([chunk.to_dict() for chunk in chunks])


def test_jsonl_round_trip_validation_and_duplicate_safe_ingestion(tmp_path):
    client, transport = make_client()
    company = CompanyIdentity("NVIDIA Corporation", "NVDA", "1045810")
    storage = CorpusStorage(tmp_path / "corpus")
    cache = RawFilingCache(tmp_path / "raw")

    first = ingest_company_filings(
        company, client, storage, forms=("10-K",), raw_cache=cache, target_chunk_size=80, max_chunk_size=120, chunk_overlap=20
    )
    request_count = len(transport.requests)
    second = ingest_company_filings(
        company, client, storage, forms=("10-K",), raw_cache=cache, target_chunk_size=80, max_chunk_size=120, chunk_overlap=20
    )

    documents = storage.load_documents()
    chunks = storage.load_chunks()
    assert len(first.documents) == len(second.documents) == 1
    assert len(documents) == 1
    assert len(chunks) == len(first.chunks)
    assert len(transport.requests) == request_count + 1  # Second run discovers again but uses raw HTML cache.
    assert validate_corpus(documents, chunks) == []
    json.dumps([document.to_dict() for document in documents] + [chunk.to_dict() for chunk in chunks])


def test_reprocessing_document_replaces_stale_chunks(tmp_path):
    storage = CorpusStorage(tmp_path / "corpus")
    document = make_document()
    first = chunk_document(document, [FilingSection("ITEM 1", "Business", "First text.")])
    second = chunk_document(document, [FilingSection("ITEM 1A", "Risk Factors", "Replacement text.")])

    storage.write_chunks(first, replace_document_ids=[document.document_id])
    storage.write_chunks(second, replace_document_ids=[document.document_id])

    assert [chunk.section for chunk in storage.load_chunks()] == ["ITEM 1A"]


def test_ingestion_reports_missing_requested_form_and_validation_reports_errors(tmp_path):
    class OnlyTenKClient:
        def find_filings(self, cik, forms, latest_per_form):
            return [
                FilingReference(cik, "10-K", "2025-02-20", "2025-01-26", "0001045810-25-000020", "nvda.htm")
            ]

        def download_filing(self, reference):
            return "<p>Plain filing content.</p>"

        def filing_url(self, reference):
            return "https://www.sec.gov/example"

    result = ingest_company_filings(
        CompanyIdentity("NVIDIA Corporation", "NVDA", "1045810"),
        OnlyTenKClient(),
        CorpusStorage(tmp_path / "corpus"),
    )
    broken = result.chunks[0].to_dict()
    broken["document_id"] = "missing"

    assert result.missing_forms == ["10-Q"]
    assert any("unknown document" in error for error in validate_corpus(result.documents, [type(result.chunks[0]).from_dict(broken)]))


def test_initial_company_configuration_is_data_not_parser_logic_and_documents_are_llm_free():
    assert {(company.ticker, company.cik) for company in INITIAL_COMPANIES} == {
        ("NVDA", "0001045810"),
        ("AAPL", "0000320193"),
        ("MSTR", "0001050446"),
    }
    source = "\n".join(path.read_text().lower() for path in (Path(__file__).resolve().parents[1] / "src" / "documents").glob("*.py"))
    assert "from llm" not in source
    assert "import llm" not in source
    assert "openai" not in source
    assert "gemini" not in source
