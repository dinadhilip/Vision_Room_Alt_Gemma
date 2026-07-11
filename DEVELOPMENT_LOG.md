# Development Log

## 2026-07-11

- Cloned and initialized work against `dinadhilip/Vision_Room_Alt_Gemma` on `codex/local-agent-bridge`.
- Added a FastAPI bridge with `/health` and `/chat` endpoints.
- Added local SQLite frame index, deterministic hash embeddings, and cosine search for `search_video_library`.
- Added demo ingestion that creates three verifiable local keyframes when no index exists.
- Added NB2 Lite and Omni Flash provider wrappers with deterministic demo fallbacks for offline judging.
- Added a sessionful rule-based orchestrator matching the planned search → cast → synthesize flow.
- Added optional OpenAI-compatible local Gemma planner support through `VISION_ROOM_LITERT_BASE_URL`.
- Added `POST /ingest/frame` so agents can add captioned frames to the local index over HTTP.
- Added confirmed-frame sync through natural-language selection and `/session/confirm-frame`.
- Added frontend thumbnail selection and frame indexing controls.
- Added `/skills`, session inspect/reset endpoints, and frontend status chips for runtime coordination.
- Added retry/fallback metadata for NB Lite and Omni provider calls, surfaced in visual captions.
- Added deferred `video_index_backend.py` skeleton for description → summary → embedding → vector upsert, with low-CPU scheduling left commented for later.
- Added `docs/ARCHITECTURE.md` to document the five components and their sync contract.
- Added a static single-surface chat UI that reacts to tool outcomes without user-visible mode switches.
- Added pytest coverage for the orchestrated flow and bridge endpoint behavior.
- Verified with `ruff check`, `pytest`, `/health`, frontend `HEAD /`, and sequential `/chat` smoke turns for search, cast, and synthesize.
