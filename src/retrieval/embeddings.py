"""Provider-neutral local embedding adapters."""

from abc import ABC, abstractmethod
import os
from typing import Sequence

import numpy as np


class EmbeddingModel(ABC):
    """Minimal embedding contract used by the local semantic index."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier persisted with an index."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row per input text."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return one embedding vector for a query."""


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Lazy adapter around the local SentenceTransformers package."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None, model=None):
        self._model_name = model_name or os.getenv("SEC_EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for local SEC retrieval. "
                    "Install project dependencies before creating the embedding model."
                ) from error
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(self._get_model().encode(values, show_progress_bar=False), dtype=np.float64)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._get_model().encode(text, show_progress_bar=False), dtype=np.float64)
