from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the bridge and local tools."""

    model_config = SettingsConfigDict(env_prefix="VISION_ROOM_", env_file=".env", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("data")
    index_db_path: Path = Path("data/index/frames.sqlite")
    uploads_dir: Path = Path("data/uploads")
    generated_dir: Path = Path("data/generated")
    static_dir: Path = Path("frontend")
    search_confidence_threshold: float = 0.18
    embedding_dims: int = 256

    litert_base_url: str | None = None
    litert_model: str = "gemma-4-local"
    nb2_lite_endpoint: str | None = None
    nb2_lite_api_key: str | None = None
    omni_flash_endpoint: str | None = None
    omni_flash_api_key: str | None = None

    @property
    def resolved_index_db_path(self) -> Path:
        return self._resolve(self.index_db_path)

    @property
    def resolved_uploads_dir(self) -> Path:
        return self._resolve(self.uploads_dir)

    @property
    def resolved_generated_dir(self) -> Path:
        return self._resolve(self.generated_dir)

    @property
    def resolved_static_dir(self) -> Path:
        return self._resolve(self.static_dir)

    def ensure_dirs(self) -> None:
        self._resolve(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.resolved_index_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_uploads_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_generated_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings

