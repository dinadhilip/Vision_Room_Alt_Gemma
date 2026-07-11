from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent_orchestrator import AgentOrchestrator, SessionRegistry
from .config import get_settings
from .embedding import HashEmbeddingModel
from .index_store import IndexStore
from .local_agent import LocalGemmaPlanner
from .providers import build_cast_provider, build_video_provider
from .scene_detect import create_demo_index, ingest_frame
from .search_tool import VideoSearchTool


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


def create_app() -> FastAPI:
    settings = get_settings()
    store = IndexStore(settings.resolved_index_db_path)
    embedding_model = HashEmbeddingModel(settings.embedding_dims)
    if store.count() == 0:
        create_demo_index(store, embedding_model, settings.resolved_uploads_dir)

    search_tool = VideoSearchTool(store, embedding_model)
    local_planner = (
        LocalGemmaPlanner(
            base_url=settings.litert_base_url,
            model=settings.litert_model,
            api_key=settings.litert_api_key,
            timeout_s=settings.litert_timeout_s,
        )
        if settings.litert_base_url
        else None
    )
    orchestrator = AgentOrchestrator(
        search_tool,
        build_cast_provider(settings),
        build_video_provider(settings),
        search_confidence_threshold=settings.search_confidence_threshold,
        local_planner=local_planner,
    )
    sessions = SessionRegistry()

    app = FastAPI(title="Vision Room Alt Gemma Bridge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/assets/uploads", StaticFiles(directory=settings.resolved_uploads_dir), name="uploads")
    app.mount("/assets/generated", StaticFiles(directory=settings.resolved_generated_dir), name="generated")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "indexed_frames": store.count()}

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict:
        session = sessions.get(request.session_id)
        return orchestrator.handle_turn(session, request.message)

    @app.post("/ingest/frame")
    async def ingest_uploaded_frame(
        frame: UploadFile = File(...),
        caption: str = Form(...),
        video_id: str = Form("uploaded"),
        timestamp_s: float = Form(0.0),
    ) -> dict:
        suffix = Path(frame.filename or "frame.png").suffix or ".png"
        safe_video_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in video_id)
        target = settings.resolved_uploads_dir / f"{safe_video_id}_{uuid.uuid4().hex[:10]}{suffix}"
        target.write_bytes(await frame.read())
        record = ingest_frame(
            target,
            caption=caption,
            video_id=safe_video_id,
            timestamp_s=timestamp_s,
            store=store,
            embedding_model=embedding_model,
        )
        return {"indexed_frames": store.count(), "frame": record.to_public_dict(score=1.0)}

    static_dir = settings.resolved_static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("vision_room.bridge_api:app", host="127.0.0.1", port=8000, reload=True)
