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


class LiteRTEmbeddingModel:
    """Uses ManagedEmbeddingClient to hit local litert-lm serve for real embeddings."""
    def __init__(self, client: 'ManagedEmbeddingClient', dims: int = 256):
        self.client = client
        self.dims = dims

    def embed(self, text: str, *, truncate_dims: int | None = None) -> np.ndarray:
        # Get embeddings from the client
        vector_list = self.client.embed(text)
        if not vector_list:
            # Fallback to zeros if model fails to return embedding
            vector = np.zeros(self.dims, dtype=np.float32)
        else:
            vector = np.array(vector_list, dtype=np.float32)
            
        dims = truncate_dims or self.dims
        if vector.size < dims:
            # Pad if needed, though real model should match or exceed dims
            padded = np.zeros(dims, dtype=np.float32)
            padded[:vector.size] = vector
            vector = padded
        return vector[:dims]

    def embed_many(self, texts: Iterable[str], *, truncate_dims: int | None = None) -> np.ndarray:
        return np.vstack([self.embed(text, truncate_dims=truncate_dims) for text in texts])


class SentenceTransformerEmbeddingModel:
    """Uses sentence-transformers library for high-quality local embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dims: int = 384) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise RuntimeError("sentence-transformers is not installed. Run `pip install sentence-transformers`")
            
        self.dims = dims
        
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Verify the model dims match our expected dims
        try:
            actual_dims = self.model.get_embedding_dimension()
        except AttributeError:
            actual_dims = self.model.get_sentence_embedding_dimension()
        
        if actual_dims != self.dims:
            import logging
            logging.getLogger(__name__).warning(
                f"Model dimension mismatch. Expected {self.dims}, but {model_name} has {actual_dims}. "
                "Embeddings will be truncated or padded."
            )

    def embed(self, text: str, *, truncate_dims: int | None = None) -> np.ndarray:
        vector = self.model.encode(text, convert_to_numpy=True)
        dims = truncate_dims or self.dims
        
        if vector.size < dims:
            padded = np.zeros(dims, dtype=np.float32)
            padded[:vector.size] = vector
            vector = padded
            
        return vector[:dims]

    def embed_many(self, texts: Iterable[str], *, truncate_dims: int | None = None) -> np.ndarray:
        vectors = self.model.encode(list(texts), convert_to_numpy=True)
        dims = truncate_dims or self.dims
        
        if vectors.shape[1] < dims:
            padded = np.zeros((vectors.shape[0], dims), dtype=np.float32)
            padded[:, :vectors.shape[1]] = vectors
            vectors = padded
            
        return vectors[:, :dims]


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denom = query_norm * candidate_norms
    denom[denom == 0] = 1.0
    return (candidates @ query) / denom

