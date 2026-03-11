from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseModel):
    app_name: str = "Autonomous Market Intelligence AI Agent"
    llm_provider: str = "openai"
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    chat_model: str = Field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    apify_api_token: str | None = Field(default_factory=lambda: os.getenv("APIFY_API_TOKEN"))
    apify_actor_id: str = Field(default_factory=lambda: os.getenv("APIFY_ACTOR_ID", "apify/google-search-scraper"))
    apify_country_code: str = Field(default_factory=lambda: os.getenv("APIFY_COUNTRY_CODE", "us"))
    apify_language_code: str = Field(default_factory=lambda: os.getenv("APIFY_LANGUAGE_CODE", "en"))

    qdrant_url: str | None = Field(default_factory=lambda: os.getenv("QDRANT_URL"))
    qdrant_api_key: str | None = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_local_path: str = Field(
        default_factory=lambda: os.getenv("QDRANT_LOCAL_PATH", str(ROOT_DIR / "data" / "qdrant"))
    )

    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "900")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    fallback_embedding_dimension: int = Field(default_factory=lambda: int(os.getenv("FALLBACK_EMBED_DIM", "256")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
