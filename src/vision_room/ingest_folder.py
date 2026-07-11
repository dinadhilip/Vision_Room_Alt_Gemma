import argparse
import time
from pathlib import Path

from vision_room.config import get_settings
from vision_room.embedding import SentenceTransformerEmbeddingModel
from vision_room.index_store import IndexStore
from vision_room.model_manager import ModelManager
from vision_room.video_index_backend import VideoIndexBackend

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest an entire folder of videos using local Edge AI.")
    parser.add_argument("folder", type=Path, help="Path to the folder containing video files (.mp4, .mov)")
    args = parser.parse_args()

    settings = get_settings()
    store = IndexStore(settings.resolved_index_db_path)
    
    model_manager = ModelManager(
        base_url=settings.litert_base_url,
        api_key=settings.litert_api_key,
        timeout_s=settings.litert_timeout_s,
    ) if settings.litert_base_url else None
    
    if model_manager:
        print("[SETUP] Checking Model Manager health...")
        
        # Check local disk first
        if not model_manager.check_local_model_disk_path():
            print("[ERROR] Gemma 4 E2B model is not found on the local disk at expected path. Please download it via the Edge Gallery.")
            import sys
            sys.exit(1)
            
        while not model_manager.check_health():
            print("[ERROR] litert-lm serve is offline or unreachable. Retrying in 5 seconds...")
            time.sleep(5)
            
        print("[SETUP] Local Edge AI API (litert-lm) is online.")
        
        expected_models = ["gemma-4-local"]
        while True:
            loaded_models = model_manager.get_loaded_models()
            missing_models = [m for m in expected_models if m not in loaded_models]
            if not missing_models:
                print("[SETUP] Gemma Models are preloaded. Proceeding.")
                break
            print(f"[ERROR] Required models {missing_models} are not preloaded. Loaded models: {loaded_models}. Retrying in 5 seconds...")
            time.sleep(5)
            
    else:
        print("[ERROR] No litert_base_url configured. Cannot proceed without local models.")
        import sys
        sys.exit(1)
        
    print("[SETUP] Initializing local SentenceTransformer for embeddings...")
    embedding_model = SentenceTransformerEmbeddingModel(dims=settings.embedding_dims)
    
    backend = VideoIndexBackend(store, embedding_model, model_manager)
    
    folder_path = args.folder
    if not folder_path.is_dir():
        parser.error(f"Provided path is not a directory: {folder_path}")
        
    records = backend.index_folder(folder_path, settings.resolved_uploads_dir)
    print(f"\n[DONE] Successfully indexed {len(records)} total frames from the folder into {settings.resolved_index_db_path}.")

if __name__ == "__main__":
    main()
