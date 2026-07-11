from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")


class HashEmbeddingModel:
    """Small deterministic embedding fallback.

    The production path should call EmbeddingGemma. This class gives the app a
    fully local, dependency-light vector path for demos and tests.
    """

    def __init__(self, dims: int = 256) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self.dims = dims

    def embed(self, text: str, *, truncate_dims: int | None = None) -> np.ndarray:
        dims = truncate_dims or self.dims
        if dims <= 0 or dims > self.dims:
            raise ValueError("truncate_dims must be between 1 and the model dimensions")

        vector = np.zeros(self.dims, dtype=np.float32)
        tokens = list(self._tokens(text))
        if not tokens:
            return vector[:dims]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log1p(len(token)))

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector[:dims]

    def embed_many(self, texts: Iterable[str], *, truncate_dims: int | None = None) -> np.ndarray:
        return np.vstack([self.embed(text, truncate_dims=truncate_dims) for text in texts])

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        return TOKEN_RE.findall(text.lower())


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denom = query_norm * candidate_norms
    denom[denom == 0] = 1.0
    return (candidates @ query) / denom

