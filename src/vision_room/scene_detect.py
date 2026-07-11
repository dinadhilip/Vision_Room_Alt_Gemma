from __future__ import annotations

import argparse
import subprocess
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from .config import get_settings
from .embedding import HashEmbeddingModel
from .index_store import FrameRecord, IndexStore


def ingest_frame(
    frame_path: Path,
    *,
    caption: str,
    video_id: str,
    timestamp_s: float,
    store: IndexStore,
    embedding_model: HashEmbeddingModel,
    frame_id: str | None = None,
) -> FrameRecord:
    resolved_frame_id = frame_id or f"{video_id}_{int(timestamp_s * 1000):09d}_{uuid.uuid4().hex[:6]}"
    record = FrameRecord(
        id=resolved_frame_id,
        video_id=video_id,
        timestamp_s=timestamp_s,
        frame_path=str(frame_path),
        caption=caption,
        embedding=embedding_model.embed(caption),
    )
    store.upsert_frame(record)
    return record


def create_demo_index(store: IndexStore, embedding_model: HashEmbeddingModel, output_dir: Path) -> list[FrameRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_frames = [
        ("pipe_leak", 12.4, "a utility room pipe starts leaking near a blue valve"),
        ("product_table", 27.0, "a clean tabletop product shot with space for a featured bottle"),
        ("workshop_person", 43.8, "a person in a workshop looking at tools under warm light"),
    ]
    records: list[FrameRecord] = []
    for index, (video_id, timestamp, caption) in enumerate(demo_frames):
        frame_path = output_dir / f"{video_id}.png"
        _make_demo_frame(frame_path, caption, index)
        records.append(
            ingest_frame(
                frame_path,
                caption=caption,
                video_id=video_id,
                timestamp_s=timestamp,
                store=store,
                embedding_model=embedding_model,
                frame_id=f"demo_{video_id}",
            )
        )
    return records


def extract_keyframes(video_path: Path, output_dir: Path, scene_threshold: float = 0.32) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(c if c.isalnum() else "_" for c in video_path.stem)
    pattern = output_dir / f"{safe_stem}_%04d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{scene_threshold})',showinfo",
        "-fps_mode",
        "vfr",
        "-strict",
        "unofficial",
        str(pattern),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return sorted(output_dir.glob(f"{safe_stem}_*.jpg"))


def _make_demo_frame(path: Path, caption: str, index: int) -> None:
    palette = [(30, 64, 110), (55, 86, 69), (83, 57, 92)]
    image = Image.new("RGB", (960, 540), palette[index % len(palette)])
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((42, 42, 918, 498), outline=(255, 255, 255, 130), width=3)
    draw.rectangle((70, 350, 890, 470), fill=(0, 0, 0, 130))
    draw.text((92, 388), caption, fill=(245, 247, 250, 255))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest demo frames or extract video keyframes.")
    parser.add_argument("--demo", action="store_true", help="Create a three-frame demo index.")
    parser.add_argument("--video", type=Path, help="Video file to extract keyframes from.")
    parser.add_argument("--caption", default="local scene keyframe", help="Caption for extracted frames.")
    args = parser.parse_args()

    settings = get_settings()
    store = IndexStore(settings.resolved_index_db_path)
    model = HashEmbeddingModel(settings.embedding_dims)

    if args.demo:
        records = create_demo_index(store, model, settings.resolved_uploads_dir)
        print(f"Indexed {len(records)} demo frames in {settings.resolved_index_db_path}")
        return

    if not args.video:
        parser.error("--demo or --video is required")

    keyframes = extract_keyframes(args.video, settings.resolved_uploads_dir)
    for offset, frame_path in enumerate(keyframes):
        ingest_frame(
            frame_path,
            caption=f"{args.caption} #{offset + 1}",
            video_id=args.video.stem,
            timestamp_s=float(offset),
            store=store,
            embedding_model=model,
        )
    print(f"Indexed {len(keyframes)} keyframes in {settings.resolved_index_db_path}")
