# Vision Room Architecture

## Five Components

1. Frontend chat Gemma agent skill
   - Files: `frontend/`, `src/vision_room/bridge_api.py`, `src/vision_room/agent_orchestrator.py`
   - Role: one chat surface, no user-facing modes, reactive visual updates from `ui_action`.

2. Semantic search skill
   - Files: `src/vision_room/search_tool.py`, `src/vision_room/index_store.py`, `src/vision_room/embedding.py`
   - Role: local/offline text query to vector match over indexed frame captions.

3. Nano Banana Lite skill
   - File: `src/vision_room/providers.py`
   - Role: `cast_into_frame` cloud hook, with deterministic offline fallback.

4. Omni Flash skill
   - File: `src/vision_room/providers.py`
   - Role: `synthesize_video` cloud hook, with deterministic offline preview fallback.

5. On-demand video indexing backend
   - Files: `src/vision_room/scene_detect.py`, `src/vision_room/video_index_backend.py`
   - Role: description -> summary -> embedding -> vector/index upsert. The CPU-idle scheduler is intentionally commented for later.

## Sync Contract

The bridge owns session state:

```text
search results -> confirmed frame -> cast anchor frame -> synthesized video
```

- Search resets downstream cast/video state.
- Frame confirmation can happen through chat text such as `use the second frame` or through `POST /session/confirm-frame`.
- Casting always uses `confirmed_frame` when it exists.
- Video synthesis requires at least one anchor frame unless a future speculative fallback is explicitly added.
- Every tool response returns a `ui_action` so the frontend updates the visual surface without switching modes.

## Runtime Hooks

- `VISION_ROOM_LITERT_BASE_URL`: optional OpenAI-compatible local Gemma planner.
- `VISION_ROOM_NB2_LITE_ENDPOINT`: optional Nano Banana Lite provider.
- `VISION_ROOM_OMNI_FLASH_ENDPOINT`: optional Omni Flash provider.
- Missing or failing providers fall back to local deterministic demo outputs.

