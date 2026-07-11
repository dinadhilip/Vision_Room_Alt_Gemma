from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .embedding import HashEmbeddingModel
from .index_store import FrameRecord, IndexStore
from .scene_detect import ingest_frame


@dataclass(frozen=True)
class FrameSummary:
    video_id: str
    timestamp_s: float
    frame_path: Path
    description: str
    summary: str


class VideoIndexBackend:
    """On-demand video indexing pipeline skeleton.

    This is intentionally not wired into the running demo yet. The agreed flow is:
    frame description -> concise summary -> embedding -> vector/index upsert.
    """

    def __init__(self, store: IndexStore, embedding_model: HashEmbeddingModel) -> None:
        self.store = store
        self.embedding_model = embedding_model

    def index_summary(self, frame: FrameSummary) -> FrameRecord:
        caption = f"{frame.summary}. {frame.description}".strip()
        return ingest_frame(
            frame.frame_path,
            caption=caption,
            video_id=frame.video_id,
            timestamp_s=frame.timestamp_s,
            store=self.store,
            embedding_model=self.embedding_model,
        )


# Later: run queued video indexing only when the machine is comfortably idle.
# Kept commented for now so the sprint stays focused on skill sync and the demo path.
#
# import psutil
# import time
#
# def run_when_cpu_is_low(queue, backend: VideoIndexBackend, threshold_percent: float = 35.0):
#     while True:
#         if psutil.cpu_percent(interval=1.0) < threshold_percent:
#             job = queue.get_nowait()
#             backend.index_video(job.video_path)
#         time.sleep(5.0)

