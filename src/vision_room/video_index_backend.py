from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .embedding import HashEmbeddingModel, LiteRTEmbeddingModel, SentenceTransformerEmbeddingModel
from .index_store import FrameRecord, IndexStore
from .scene_detect import ingest_frame, extract_keyframes
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

# To type hint either model
AnyEmbeddingModel = Union[HashEmbeddingModel, LiteRTEmbeddingModel, SentenceTransformerEmbeddingModel]

@dataclass(frozen=True)
class FrameSummary:
    video_id: str
    timestamp_s: float
    frame_path: Path
    description: str
    summary: str


class VideoIndexBackend:
    """On-demand video indexing pipeline using Gemma models."""

    def __init__(self, store: IndexStore, embedding_model: AnyEmbeddingModel, model_manager: ModelManager | None = None) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.model_manager = model_manager

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

    def index_folder(self, folder_path: Path, output_dir: Path) -> list[FrameRecord]:
        print(f"[INDEXING: SCANNING] Scanning folder {folder_path} for video files...")
        video_files = list(folder_path.rglob("*.mp4")) + list(folder_path.rglob("*.mov"))
        records = []
        
        vision_client = self.model_manager.get_vision_client() if self.model_manager else None

        for video_path in video_files:
            print(f"[INDEXING: FFMPEG] Extracting keyframes for {video_path.name}...")
            keyframes = extract_keyframes(video_path, output_dir)
            
            for offset, frame_path in enumerate(keyframes):
                timestamp_s = float(offset)
                
                print(f"[INDEXING: GEMMA_VISION] Analyzing frame {frame_path.name}...")
                if vision_client and self.model_manager.is_healthy:
                    import base64
                    b64_frame = base64.b64encode(frame_path.read_bytes()).decode("ascii")
                    description = vision_client.analyze_frame(
                        b64_frame, 
                        prompt="Describe objects, activity, colour, humans, things in this scene."
                    )
                    summary = description.split('.')[0] if description else "Local scene keyframe"
                else:
                    description = "Local scene keyframe"
                    summary = "Local scene keyframe"
                
                print(f"[INDEXING: SQLITE] Saving vector index for {frame_path.name}...")
                records.append(self.index_summary(FrameSummary(
                    video_id=video_path.stem,
                    timestamp_s=timestamp_s,
                    frame_path=frame_path,
                    description=description,
                    summary=summary
                )))
                
        return records

