"""Local, provider-neutral semantic retrieval for SEC filing chunks."""

from .embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel
from .index import SemanticIndex, build_index, corpus_fingerprint, load_index
from .retriever import SemanticRetriever
from .types import RetrievalResult
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever
from .reranker import CrossEncoderReranker, Reranker

__all__ = [
    "EmbeddingModel",
    "RetrievalResult",
    "SemanticIndex",
    "SemanticRetriever",
    "SentenceTransformerEmbeddingModel",
    "build_index",
    "corpus_fingerprint",
    "load_index",
    "BM25Retriever", "HybridRetriever", "CrossEncoderReranker", "Reranker",
]
