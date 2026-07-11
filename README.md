Update : New Loom Video is out (Code is in a branch)!

# Vision Room Alt Gemma

Vision Room is an edge-to-cloud conversational video studio. It uses a local Gemma model to orchestrate searches over your local video footage, Nano Banana 2 Lite (NB2 Lite) to cast subjects into matching frames, and Omni Flash to synthesize a final video.

## Prerequisites & Environment Variables

Before running the application, make sure to set the required API keys and endpoint URLs in your `.env` file at the root of the project:

```env
VISION_ROOM_LITERT_BASE_URL="http://127.0.0.1:8000"
VISION_ROOM_LITERT_MODEL="gemma-4-local"
VISION_ROOM_NB2_LITE_API_KEY="your_actual_gemini_api_key_here"
# Add your Omni Flash API key when ready for video synthesis
# VISION_ROOM_OMNI_FLASH_API_KEY="..."
```

*Note: If an API key is missing or a cloud provider call fails, the bridge API will automatically fall back to local deterministic demo stubs so your workflow isn't blocked.*

## How to Run the Application

The application requires two servers running simultaneously. Open two separate terminal windows inside the project root (`/Users/vyshnav/Documents/Vision room`).

### 1. Start the Local Model Server (Terminal 1)
Run the local `litert-lm` server to provide the Gemma-4-local OpenAI-compatible endpoint:
```bash
./.venv/bin/litert-lm serve --port 8000
```

### 2. Start the Bridge API & UI (Terminal 2)
Run the main FastAPI bridge which hosts the UI, handles chat routing, and orchestrates the backend skills:
```bash
PYTHONPATH=src ./.venv/bin/python -m vision_room.bridge_api
```

## Open the Application

Once both servers are running, open your web browser and navigate to the bridge API's frontend:

**[http://localhost:8080](http://localhost:8080)**

From the UI, you can chat with the agent to:
1. Search your indexed local video frames (e.g., *"find max verstappen"*).
2. Cast a new subject into the matched frame (e.g., *"cast a repair technician in a yellow jacket"*).
3. Synthesize a video from the approved cast frame (e.g., *"approved, make video"*).

## Adding New Local Videos
To index new video footage, drop your `.mp4` files into the project folder. You can trigger the indexing process (which extracts keyframes, generates captions via Gemma Vision, creates embeddings, and upserts into the local SQLite database) by clicking the **"Load Folder"** button directly in the web UI.
