"""Local, provider-neutral semantic retrieval for SEC filing chunks."""

from .embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel
from .index import SemanticIndex, build_index, corpus_fingerprint, load_index
from .retriever import SemanticRetriever
from .types import RetrievalResult

__all__ = [
    "EmbeddingModel",
    "RetrievalResult",
    "SemanticIndex",
    "SemanticRetriever",
    "SentenceTransformerEmbeddingModel",
    "build_index",
    "corpus_fingerprint",
    "load_index",
]
