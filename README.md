# Vision Room Alt Gemma

An edge-to-cloud conversational video studio: local Gemma-style orchestration searches the user's footage, NB2 Lite casts a subject into the matched frame, and Omni Flash turns approved anchors into a short conversational video.

The current repo includes a runnable demo spine. It is intentionally local-first: search, indexing, session state, and the chat bridge run offline; NB2 Lite and Omni Flash are behind provider interfaces with deterministic demo fallbacks until real endpoints are configured.

## What Works Now

- `POST /chat` conversational bridge with one chat surface and tool-like state transitions.
- Optional OpenAI-compatible local Gemma planner via `litert-lm serve`.
- `search_video_library` over a local SQLite frame index using deterministic embeddings.
- `POST /ingest/frame` for adding captioned local frames to the searchable index.
- `cast_into_frame` provider interface with an NB2 Lite HTTP hook and an offline visual fallback.
- `synthesize_video` provider interface with an Omni Flash HTTP hook and an offline preview manifest fallback.
- Automatic demo index creation on first server start.
- Static frontend that updates frame galleries and generated video previews inline.
- Tests covering search → cast → synthesize plus bridge health/chat behavior.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
python -m vision_room.scene_detect --demo
uvicorn vision_room.bridge_api:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), then try:

```text
find where the pipe started leaking
cast a repair technician in a yellow jacket
approved, make video
```

## Cloud Provider Hooks

Set these environment variables when local Gemma or real cloud services are available:

```bash
export VISION_ROOM_LITERT_BASE_URL="http://127.0.0.1:8080"
export VISION_ROOM_LITERT_MODEL="gemma-4-local"
export VISION_ROOM_NB2_LITE_ENDPOINT="https://..."
export VISION_ROOM_NB2_LITE_API_KEY="..."
export VISION_ROOM_OMNI_FLASH_ENDPOINT="https://..."
export VISION_ROOM_OMNI_FLASH_API_KEY="..."
```

If a provider call fails, the bridge falls back to the local demo provider so the conversation can continue during a live demo.

## Add Local Frames

```bash
curl -X POST http://127.0.0.1:8000/ingest/frame \
  -F frame=@/path/to/frame.png \
  -F caption="a pipe starts leaking near a blue valve" \
  -F video_id="workshop_clip" \
  -F timestamp_s=12.4
```

## Project Map

```text
src/vision_room/bridge_api.py          FastAPI app and static frontend mount
src/vision_room/agent_orchestrator.py  Sessionful search/cast/synthesize routing
src/vision_room/local_agent.py         OpenAI-compatible Gemma tool planner
src/vision_room/index_store.py         SQLite frame index
src/vision_room/embedding.py           Local deterministic embedding fallback
src/vision_room/search_tool.py         search_video_library implementation
src/vision_room/providers.py           NB2 Lite / Omni Flash interfaces and fallbacks
src/vision_room/scene_detect.py        Demo ingest and ffmpeg keyframe extraction
frontend/                             Single chat surface UI
tests/                                Pipeline and bridge tests
```
