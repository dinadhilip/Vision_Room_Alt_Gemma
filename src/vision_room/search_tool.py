from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import HashEmbeddingModel, cosine_similarity
from .index_store import FrameRecord, IndexStore


@dataclass(frozen=True)
class SearchHit:
    record: FrameRecord
    score: float

    def to_public_dict(self) -> dict:
        return self.record.to_public_dict(score=self.score)


class VideoSearchTool:
    def __init__(self, store: IndexStore, embedding_model: HashEmbeddingModel) -> None:
        self.store = store
        self.embedding_model = embedding_model

    def search_video_library(self, query: str, top_k: int = 3) -> list[SearchHit]:
        frames = self.store.all_frames()
        if not frames:
            return []

        query_vector = self.embedding_model.embed(query, truncate_dims=len(frames[0].embedding))
        matrix = np.vstack([frame.embedding for frame in frames])
        scores = cosine_similarity(query_vector, matrix)
        ranked_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        return [SearchHit(frames[index], float(scores[index])) for index in ranked_indices]

