from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from vision_room.agent_orchestrator import AgentOrchestrator, SessionRegistry
from vision_room.embedding import HashEmbeddingModel
from vision_room.index_store import IndexStore
from vision_room.local_agent import PlannedToolCall
from vision_room.providers import DemoCastProvider, DemoVideoProvider
from vision_room.scene_detect import create_demo_index
from vision_room.search_tool import VideoSearchTool


class FakePlanner:
    def __init__(self, decision: PlannedToolCall) -> None:
        self.decision = decision

    def plan(self, **_kwargs) -> PlannedToolCall:
        return self.decision


def build_orchestrator(tmp_path: Path, local_planner=None) -> AgentOrchestrator:
    store = IndexStore(tmp_path / "frames.sqlite")
    model = HashEmbeddingModel(128)
    create_demo_index(store, model, tmp_path / "uploads")
    return AgentOrchestrator(
        VideoSearchTool(store, model),
        DemoCastProvider(tmp_path / "generated"),
        DemoVideoProvider(tmp_path / "generated"),
        search_confidence_threshold=-1.0,
        local_planner=local_planner,
    )


def test_search_cast_synthesize_flow(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    session = SessionRegistry().get("demo")

    search = orchestrator.handle_turn(session, "find the pipe leaking near a valve")
    assert search["ui_action"]["type"] == "show_frame_gallery"
    assert "pipe" in search["ui_action"]["payload"]["primary"]["caption"]

    cast = orchestrator.handle_turn(session, "cast a repair technician in a yellow jacket")
    assert cast["ui_action"]["type"] == "show_frame_gallery"
    assert Path(cast["ui_action"]["payload"]["primary"]["frame_path"]).exists()

    video = orchestrator.handle_turn(session, "approved, make video")
    assert video["ui_action"]["type"] == "show_generated_video"
    assert video["ui_action"]["payload"]["video_id"].startswith("video_")


def test_local_gemma_planner_tool_decision_executes(tmp_path: Path) -> None:
    planner = FakePlanner(
        PlannedToolCall("search_video_library", {"query": "blue valve pipe leak", "top_k": 2})
    )
    orchestrator = build_orchestrator(tmp_path, local_planner=planner)
    session = SessionRegistry().get("gemma")

    response = orchestrator.handle_turn(session, "please find the relevant moment")

    assert response["ui_action"]["type"] == "show_frame_gallery"
    assert len(response["ui_action"]["payload"]["frames"]) == 2
    assert "pipe" in response["ui_action"]["payload"]["primary"]["caption"]


def test_bridge_health_and_chat(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_ROOM_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("VISION_ROOM_INDEX_DB_PATH", "data/index/test.sqlite")
    monkeypatch.setenv("VISION_ROOM_STATIC_DIR", str(Path.cwd() / "frontend"))

    from vision_room import config
    from vision_room import bridge_api

    config.get_settings.cache_clear()
    app = bridge_api.create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["indexed_frames"] == 3

    chat = client.post("/chat", json={"session_id": "x", "message": "find a product shot"})
    assert chat.status_code == 200
    assert chat.json()["ui_action"]["type"] == "show_frame_gallery"


def test_ingest_frame_endpoint_indexes_upload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_ROOM_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("VISION_ROOM_INDEX_DB_PATH", "data/index/test.sqlite")
    monkeypatch.setenv("VISION_ROOM_STATIC_DIR", str(Path.cwd() / "frontend"))

    from vision_room import bridge_api
    from vision_room import config

    config.get_settings.cache_clear()
    app = bridge_api.create_app()
    client = TestClient(app)

    image = Image.new("RGB", (64, 36), (20, 90, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    ingest = client.post(
        "/ingest/frame",
        data={
            "caption": "custom hallway scene with a red emergency light",
            "video_id": "hallway-demo",
            "timestamp_s": "9.5",
        },
        files={"frame": ("hallway.png", buffer, "image/png")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["indexed_frames"] == 4

    chat = client.post(
        "/chat",
        json={"session_id": "uploaded", "message": "find the red emergency hallway light"},
    )
    assert chat.status_code == 200
    assert "red emergency light" in chat.json()["ui_action"]["payload"]["primary"]["caption"]
