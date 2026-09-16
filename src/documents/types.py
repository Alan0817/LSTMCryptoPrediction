"""JSON-safe SEC filing and chunk metadata types with deterministic identities."""

from dataclasses import asdict, dataclass
from hashlib import sha256


def normalize_cik(cik: str | int) -> str:
    """Return the SEC's stable ten-digit CIK representation."""
    value = str(cik).strip()
    if not value.isdigit():
        raise ValueError("CIK must contain only digits.")
    return value.zfill(10)


def normalize_accession(accession_number: str) -> str:
    """Validate the common SEC accession format without changing its stable value."""
    value = accession_number.strip()
    compact = value.replace("-", "")
    if not compact.isdigit() or len(compact) != 18:
        raise ValueError("accession_number must contain 18 digits, optionally separated by hyphens.")
    return "{}-{}-{}".format(compact[:10], compact[10:12], compact[12:])


def make_document_id(cik: str | int, accession_number: str, primary_document: str) -> str:
    """Derive a stable corpus ID from the filing's SEC identity."""
    identity = "|".join((normalize_cik(cik), normalize_accession(accession_number), primary_document.strip()))
    return "sec_" + sha256(identity.encode("utf-8")).hexdigest()[:24]


def make_chunk_id(document_id: str, section: str, chunk_index: int) -> str:
    """Derive a stable chunk ID from the document, section, and deterministic index."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative.")
    identity = "|".join((document_id, section.strip().upper(), str(chunk_index)))
    return "chunk_" + sha256(identity.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class CompanyIdentity:
    """Configuration metadata; CIK is the stable SEC identity, not the ticker."""

    company: str
    ticker: str
    cik: str

    def __post_init__(self):
        object.__setattr__(self, "cik", normalize_cik(self.cik))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FilingReference:
    """A selected primary SEC filing document discovered from submissions metadata."""

    cik: str
    form: str
    filing_date: str
    period_end: str | None
    accession_number: str
    primary_document: str

    def __post_init__(self):
        object.__setattr__(self, "cik", normalize_cik(self.cik))
        object.__setattr__(self, "accession_number", normalize_accession(self.accession_number))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FinancialDocument:
    document_id: str
    company: str
    ticker: str
    cik: str
    accession_number: str
    primary_document: str
    document_type: str
    filing_date: str
    period_end: str | None
    source: str
    source_url: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "FinancialDocument":
        return cls(**value)


@dataclass(frozen=True)
class FinancialDocumentChunk:
    chunk_id: str
    document_id: str
    company: str
    ticker: str
    cik: str
    accession_number: str
    document_type: str
    filing_date: str
    period_end: str | None
    source: str
    source_url: str
    section: str
    section_title: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "FinancialDocumentChunk":
        return cls(**value)
