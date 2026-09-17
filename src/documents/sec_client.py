"""Small official SEC EDGAR submissions and primary-document client."""

import os
from pathlib import PurePosixPath

import requests

from .types import FilingReference, normalize_accession, normalize_cik


SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SUPPORTED_FORMS = frozenset({"10-K", "10-Q"})


class SECClient:
    """Use official SEC endpoints with an identifiable, configurable User-Agent."""

    def __init__(self, user_agent: str | None = None, transport=None, timeout: float = 30.0):
        self._user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self._user_agent:
            raise RuntimeError("SEC_USER_AGENT must be configured for SEC EDGAR access.")
        self._transport = transport or requests.Session()
        self._timeout = timeout

    def get_company_submissions(self, cik: str | int) -> dict:
        cik = normalize_cik(cik)
        return self._get_json("{}/submissions/CIK{}.json".format(SEC_DATA_BASE_URL, cik))

    def find_filings(
        self,
        cik: str | int,
        forms: tuple[str, ...] | list[str] | set[str] = ("10-K", "10-Q"),
        latest_per_form: int = 1,
    ) -> list[FilingReference]:
        """Select newest official filing metadata per requested form without substitutions."""
        if latest_per_form <= 0:
            raise ValueError("latest_per_form must be positive.")
        requested_forms = set(forms)
        unsupported = requested_forms.difference(SUPPORTED_FORMS)
        if unsupported:
            raise ValueError("Unsupported SEC forms: {}.".format(sorted(unsupported)))
        submissions = self.get_company_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        records = _recent_records(recent)
        references = []
        for form in sorted(requested_forms):
            matches = [record for record in records if record.get("form") == form and record.get("primaryDocument")]
            matches.sort(key=lambda record: record.get("filingDate", ""), reverse=True)
            for record in matches[:latest_per_form]:
                references.append(
                    FilingReference(
                        cik=cik,
                        form=form,
                        filing_date=record["filingDate"],
                        period_end=record.get("reportDate") or None,
                        accession_number=record["accessionNumber"],
                        primary_document=record["primaryDocument"],
                    )
                )
        return sorted(references, key=lambda reference: (reference.form, reference.filing_date), reverse=True)

    def filing_url(self, reference: FilingReference) -> str:
        accession_compact = reference.accession_number.replace("-", "")
        cik_number = str(int(reference.cik))
        primary_document = str(PurePosixPath(reference.primary_document))
        return "{}/{}/{}/{}".format(SEC_ARCHIVES_BASE_URL, cik_number, accession_compact, primary_document)

    def download_filing(self, reference: FilingReference) -> str:
        """Download an official primary filing HTML document and preserve source traceability."""
        response = self._get(self.filing_url(reference))
        return response.text

    def _get_json(self, url: str) -> dict:
        return self._get(url).json()

    def _get(self, url: str):
        response = self._transport.get(
            url,
            headers={"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response


def _recent_records(recent: dict | list[dict]) -> list[dict]:
    if isinstance(recent, list):
        return list(recent)
    if not isinstance(recent, dict):
        raise ValueError("SEC submissions recent filings must be a list or column mapping.")
    lengths = {len(value) for value in recent.values() if isinstance(value, list)}
    if not lengths:
        return []
    if len(lengths) != 1:
        raise ValueError("SEC submissions filing columns have inconsistent lengths.")
    length = lengths.pop()
    return [{key: value[index] for key, value in recent.items() if isinstance(value, list)} for index in range(length)]
