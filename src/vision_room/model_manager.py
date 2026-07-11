from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger(__name__)

class ModelManager:
    """Manager that acts as an interface to litert-lm serve (Google Edge AI Gallery).
    It tracks API health, handles retries, and can provide status/error logs."""

    def __init__(self, base_url: str | None, api_key: str | None = None, timeout_s: float = 120.0, retry_count: int = 1):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.retry_count = retry_count
        self.is_healthy = True

    def check_health(self) -> bool:
        if not self.base_url:
            self.is_healthy = False
            return False
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            response.raise_for_status()
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Model manager health check failed: {e}")
            self.is_healthy = False
            return False

    def get_loaded_models(self) -> list[str]:
        if not self.base_url:
            return []
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [model.get("id") for model in data.get("data", []) if "id" in model]
        except Exception as e:
            logger.warning(f"Failed to fetch loaded models: {e}")
            return []

    def check_local_model_disk_path(self) -> bool:
        """Verifies if the Gemma 4 E2B model is present on the local disk."""
        import os
        model_path = os.path.expanduser("~/Library/Application Support/com.google.AIEdgeGallery/Documents/Gemma_4_E2B_it/v2")
        return os.path.isdir(model_path)

    def get_vision_client(self) -> ManagedVisionClient:
        return ManagedVisionClient(self)

    def get_embedding_client(self) -> ManagedEmbeddingClient:
        return ManagedEmbeddingClient(self)


class ManagedVisionClient:
    """Hits litert-lm serve for Gemma E4B (vision tasks)."""

    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.model = "gemma-4-local"  # Target Gemma E4B

    def analyze_frame(self, frame_b64: str, prompt: str) -> str:
        """Extracts descriptive metadata from a frame."""
        if not self.manager.base_url:
            return "Local demo: No litert server configured for vision analysis."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                    ]
                }
            ],
            "temperature": 0.2,
        }
        
        headers = {"Authorization": f"Bearer {self.manager.api_key}"} if self.manager.api_key else {}
        attempts = self.manager.retry_count + 1

        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    f"{self.manager.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.manager.timeout_s,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Vision API error (attempt {attempt}): {e}")
                self.manager.is_healthy = False
                if attempt == attempts:
                    raise RuntimeError(f"Vision analysis failed: {e}")
        return ""


class ManagedEmbeddingClient:
    """Hits litert-lm serve for EmbeddedGemma 300M (embedding tasks)."""

    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.model = "embedded-gemma-300m"  # Target EmbeddedGemma 300M

    def embed(self, text: str) -> list[float]:
        """Gets vector embedding for the given text."""
        if not self.manager.base_url:
            return []

        payload = {
            "model": self.model,
            "input": text
        }
        
        headers = {"Authorization": f"Bearer {self.manager.api_key}"} if self.manager.api_key else {}
        attempts = self.manager.retry_count + 1

        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    f"{self.manager.base_url}/v1/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=self.manager.timeout_s,
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
            except Exception as e:
                logger.warning(f"Embedding API error (attempt {attempt}): {e}")
                self.manager.is_healthy = False
                if attempt == attempts:
                    raise RuntimeError(f"Embedding failed: {e}")
        return []
