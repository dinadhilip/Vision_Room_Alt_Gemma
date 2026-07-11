from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    id: str
    video_id: str
    timestamp_s: float
    frame_path: str
    caption: str
    embedding: np.ndarray

    def to_public_dict(self, score: float | None = None) -> dict:
        payload = {
            "frame_id": self.id,
            "video_id": self.video_id,
            "timestamp_s": self.timestamp_s,
            "frame_path": self.frame_path,
            "caption": self.caption,
        }
        if score is not None:
            payload["score"] = round(float(score), 4)
        return payload


class IndexStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_frame(self, record: FrameRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO frames (id, video_id, timestamp_s, frame_path, caption, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    video_id=excluded.video_id,
                    timestamp_s=excluded.timestamp_s,
                    frame_path=excluded.frame_path,
                    caption=excluded.caption,
                    embedding=excluded.embedding
                """,
                (
                    record.id,
                    record.video_id,
                    record.timestamp_s,
                    record.frame_path,
                    record.caption,
                    record.embedding.astype(np.float32).tobytes(),
                ),
            )

    def get_frame(self, frame_id: str) -> FrameRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, video_id, timestamp_s, frame_path, caption, embedding FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def all_frames(self) -> list[FrameRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, video_id, timestamp_s, frame_path, caption, embedding FROM frames"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        return int(row[0])

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    timestamp_s REAL NOT NULL,
                    frame_path TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames(video_id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _row_to_record(row: sqlite3.Row | tuple) -> FrameRecord:
        return FrameRecord(
            id=row[0],
            video_id=row[1],
            timestamp_s=float(row[2]),
            frame_path=row[3],
            caption=row[4],
            embedding=np.frombuffer(row[5], dtype=np.float32),
        )

