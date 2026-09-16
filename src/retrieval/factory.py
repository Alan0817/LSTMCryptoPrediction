"""Application-level construction for configured local document retrieval."""
import os
from documents.storage import CorpusStorage
from .bm25 import BM25Retriever
from .embeddings import SentenceTransformerEmbeddingModel
from .hybrid import HybridRetriever
from .index import load_index
from .reranker import CrossEncoderReranker
from .retriever import SemanticRetriever

VALID_BACKENDS={'dense','bm25','hybrid','hybrid_reranked'}
DEFAULT_BACKEND='hybrid'
def build_document_retriever(backend=None, chunks=None, embedding_model=None, index_dir='data/financial_documents/semantic_index', reranker=None):
    backend=backend or os.getenv('RETRIEVAL_BACKEND') or DEFAULT_BACKEND
    if backend not in VALID_BACKENDS: raise ValueError('Unsupported retrieval backend: {!r}.'.format(backend))
    chunks=list(chunks) if chunks is not None else CorpusStorage().load_chunks()
    if not chunks: raise RuntimeError('Document corpus is unavailable or empty.')
    if backend=='bm25': return BM25Retriever(chunks)
    model=embedding_model or SentenceTransformerEmbeddingModel()
    dense=SemanticRetriever(load_index(chunks,index_dir,model),chunks,model)
    if backend=='dense': return dense
    hybrid=HybridRetriever(dense,BM25Retriever(chunks),candidate_k=20,rrf_k=60)
    if backend=='hybrid': return hybrid
    return HybridRetriever(dense,BM25Retriever(chunks),reranker=reranker or CrossEncoderReranker(device='cpu'),candidate_k=20,rrf_k=60)
