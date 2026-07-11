from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent_orchestrator import AgentOrchestrator, SessionRegistry
from .config import get_settings
from .embedding import HashEmbeddingModel
from .index_store import IndexStore
from .providers import build_cast_provider, build_video_provider
from .scene_detect import create_demo_index
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
    orchestrator = AgentOrchestrator(
        search_tool,
        build_cast_provider(settings),
        build_video_provider(settings),
        search_confidence_threshold=settings.search_confidence_threshold,
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

    static_dir = settings.resolved_static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("vision_room.bridge_api:app", host="127.0.0.1", port=8000, reload=True)
