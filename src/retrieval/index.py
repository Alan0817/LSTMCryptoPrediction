"""Persistent exact-cosine vector index for FinancialDocumentChunk records."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from documents.types import FinancialDocumentChunk

from .embeddings import EmbeddingModel


INDEX_VERSION = "1"
VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class SemanticIndex:
    vectors: np.ndarray
    chunk_ids: tuple[str, ...]
    metadata: dict

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])


def _ordered_chunks(chunks: Sequence[FinancialDocumentChunk]) -> list[FinancialDocumentChunk]:
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    ids = [chunk.chunk_id for chunk in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot index duplicate chunk IDs.")
    if any(not chunk.text.strip() for chunk in ordered):
        raise ValueError("Cannot index chunks with empty text.")
    return ordered


def corpus_fingerprint(chunks: Sequence[FinancialDocumentChunk]) -> str:
    """Hash stable corpus identity and text so stale indexes fail closed."""
    digest = sha256()
    for chunk in _ordered_chunks(chunks):
        text_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
        digest.update("|".join((chunk.chunk_id, chunk.document_id, text_hash)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_document_vectors(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Document embeddings must be a two-dimensional array.")
    if values.shape[0] != expected_rows:
        raise ValueError("Document embedding row count does not match chunk count.")
    if values.shape[1] <= 0 and expected_rows:
        raise ValueError("Document embeddings must have a positive dimension.")
    if not np.isfinite(values).all():
        raise ValueError("Document embeddings must contain only finite values.")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("Document embeddings must not contain zero vectors.")
    return values / norms[:, np.newaxis]


def build_index(
    chunks: Sequence[FinancialDocumentChunk],
    embedding_model: EmbeddingModel,
    output_dir: str | Path,
    batch_size: int = 64,
) -> SemanticIndex:
    """Embed sorted SEC chunks and atomically persist an inspectable index."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    ordered = _ordered_chunks(chunks)
    if not ordered:
        raise ValueError("Cannot build an index from an empty chunk corpus.")

    batches = []
    for start in range(0, len(ordered), batch_size):
        embedded = embedding_model.embed_documents([chunk.text for chunk in ordered[start : start + batch_size]])
        batches.append(np.asarray(embedded, dtype=np.float64))
    try:
        vectors = np.vstack(batches)
    except ValueError as error:
        raise ValueError("Document embedding batches have incompatible dimensions.") from error
    vectors = _normalize_document_vectors(vectors, len(ordered))

    chunk_ids = tuple(chunk.chunk_id for chunk in ordered)
    metadata = {
        "index_version": INDEX_VERSION,
        "embedding_model": embedding_model.model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "document_count": len({chunk.document_id for chunk in ordered}),
        "chunk_count": len(ordered),
        "chunk_ids": list(chunk_ids),
        "corpus_fingerprint": corpus_fingerprint(ordered),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / VECTORS_FILE, vectors, allow_pickle=False)
    (root / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return SemanticIndex(vectors=vectors, chunk_ids=chunk_ids, metadata=metadata)


def load_index(
    chunks: Sequence[FinancialDocumentChunk],
    index_dir: str | Path,
    embedding_model: EmbeddingModel | None = None,
) -> SemanticIndex:
    """Load an index only when it exactly matches the current chunk corpus."""
    root = Path(index_dir)
    vectors_path = root / VECTORS_FILE
    metadata_path = root / METADATA_FILE
    if not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Semantic index requires both vectors.npy and metadata.json.")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("Semantic index metadata.json is invalid JSON.") from error
    required = {"index_version", "embedding_model", "embedding_dimension", "document_count", "chunk_count", "chunk_ids", "corpus_fingerprint", "created_at"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Semantic index metadata is incomplete.")
    if metadata["index_version"] != INDEX_VERSION:
        raise ValueError("Semantic index version is unsupported.")
    ordered = _ordered_chunks(chunks)
    expected_ids = [chunk.chunk_id for chunk in ordered]
    if metadata["chunk_ids"] != expected_ids or metadata["corpus_fingerprint"] != corpus_fingerprint(ordered):
        raise ValueError("Semantic index does not match the current chunk corpus; rebuild it.")
    if metadata["chunk_count"] != len(ordered) or metadata["document_count"] != len({chunk.document_id for chunk in ordered}):
        raise ValueError("Semantic index metadata counts do not match the current chunk corpus.")
    if embedding_model is not None and metadata["embedding_model"] != embedding_model.model_name:
        raise ValueError("Semantic index embedding model does not match the requested model.")
    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Semantic index vectors.npy cannot be loaded.") from error
    vectors = _normalize_document_vectors(vectors, len(ordered))
    if vectors.shape[1] != metadata["embedding_dimension"]:
        raise ValueError("Semantic index vector dimension does not match metadata.")
    return SemanticIndex(vectors=vectors, chunk_ids=tuple(expected_ids), metadata=metadata)
