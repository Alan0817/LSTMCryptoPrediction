"""Lazy local cross-encoder adapter."""

import os

import numpy as np


class Reranker:
    def rerank(self, query, candidates, top_k):
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name=None, device="cpu", model=None):
        self.model_name = model_name or os.getenv("SEC_RERANKER_MODEL") or self.DEFAULT_MODEL
        self.device = device
        self._model = model

    def _get(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query, candidates, top_k):
        pairs = [(query, candidate.text) for candidate, _ in candidates]
        scores = np.asarray(self._get().predict(pairs), dtype=float)
        ranked = zip(candidates, scores)
        return sorted(
            ranked,
            key=lambda item: (-float(item[1]), item[0][0].chunk_id),
        )[:top_k]
