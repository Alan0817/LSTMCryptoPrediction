"""Deterministic JSONL corpus persistence and validation for SEC documents."""

import json
from pathlib import Path

from .types import FinancialDocument, FinancialDocumentChunk


class CorpusStorage:
    """Upsert JSON-safe corpus records by stable IDs rather than appending duplicates."""

    def __init__(self, root: str | Path = "data/financial_documents"):
        self.root = Path(root)
        self.documents_path = self.root / "documents.jsonl"
        self.chunks_path = self.root / "chunks.jsonl"

    def write_documents(self, documents: list[FinancialDocument]) -> None:
        existing = {document.document_id: document for document in self.load_documents()}
        existing.update({document.document_id: document for document in documents})
        self._write_jsonl(self.documents_path, [document.to_dict() for document in sorted(existing.values(), key=lambda item: item.document_id)])

    def write_chunks(
        self,
        chunks: list[FinancialDocumentChunk],
        replace_document_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Upsert chunks, replacing a reprocessed document's previous chunk set."""
        existing = {chunk.chunk_id: chunk for chunk in self.load_chunks()}
        for chunk_id, chunk in list(existing.items()):
            if chunk.document_id in replace_document_ids:
                del existing[chunk_id]
        existing.update({chunk.chunk_id: chunk for chunk in chunks})
        self._write_jsonl(self.chunks_path, [chunk.to_dict() for chunk in sorted(existing.values(), key=lambda item: item.chunk_id)])

    def load_documents(self) -> list[FinancialDocument]:
        return [FinancialDocument.from_dict(value) for value in self._read_jsonl(self.documents_path)]

    def load_chunks(self) -> list[FinancialDocumentChunk]:
        return [FinancialDocumentChunk.from_dict(value) for value in self._read_jsonl(self.chunks_path)]

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


class RawFilingCache:
    """File cache for official raw HTML so parser experiments do not redownload filings."""

    def __init__(self, root: str | Path = "data/sec_filings/raw"):
        self.root = Path(root)

    def load(self, document_id: str) -> str | None:
        path = self.root / "{}.html".format(document_id)
        return path.read_text() if path.exists() else None

    def write(self, document_id: str, html: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "{}.html".format(document_id)
        path.write_text(html)
        return path


def validate_corpus(documents: list[FinancialDocument], chunks: list[FinancialDocumentChunk]) -> list[str]:
    """Return human-readable invariant violations without changing persisted records."""
    errors = []
    document_ids = [document.document_id for document in documents]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(document_ids) != len(set(document_ids)):
        errors.append("Duplicate document IDs found.")
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("Duplicate chunk IDs found.")
    documents_by_id = {document.document_id: document for document in documents}
    indices_by_document = {}
    for document in documents:
        if document.document_type not in {"10-K", "10-Q"}:
            errors.append("Document {} has invalid document_type.".format(document.document_id))
        if not document.source_url:
            errors.append("Document {} is missing source_url.".format(document.document_id))
        if not document.text.strip():
            errors.append("Document {} has empty text.".format(document.document_id))
    for chunk in chunks:
        if chunk.document_id not in documents_by_id:
            errors.append("Chunk {} references an unknown document.".format(chunk.chunk_id))
        if not chunk.text.strip():
            errors.append("Chunk {} has empty text.".format(chunk.chunk_id))
        if not chunk.source_url:
            errors.append("Chunk {} is missing source_url.".format(chunk.chunk_id))
        if chunk.document_type not in {"10-K", "10-Q"}:
            errors.append("Chunk {} has invalid document_type.".format(chunk.chunk_id))
        if chunk.chunk_index < 0:
            errors.append("Chunk {} has a negative chunk_index.".format(chunk.chunk_id))
        indices_by_document.setdefault(chunk.document_id, []).append(chunk.chunk_index)
    for document_id, indices in indices_by_document.items():
        if sorted(indices) != list(range(len(indices))):
            errors.append("Chunk indices are not contiguous for document {}.".format(document_id))
    return errors
