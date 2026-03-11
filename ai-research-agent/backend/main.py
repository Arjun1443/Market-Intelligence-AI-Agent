from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR / "backend", ROOT_DIR / "config"):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from rag_pipeline import ResearchPipeline  # noqa: E402
from settings import get_settings  # noqa: E402


settings = get_settings()
pipeline = ResearchPipeline(settings=settings)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Autonomous Market Intelligence AI Agent API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research question to investigate")
    max_results: int = Field(default=10, ge=3, le=25)


class ReportRequest(BaseModel):
    query: str = Field(..., min_length=3)
    collection_name: str | None = None
    top_k: int = Field(default=8, ge=3, le=20)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "apify_configured": bool(settings.apify_api_token),
        "openai_configured": bool(settings.openai_api_key),
    }


@app.post("/research")
def run_research(request: ResearchRequest) -> dict[str, Any]:
    try:
        return pipeline.run_research(query=request.query, max_results=request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {exc}") from exc


@app.post("/report")
def generate_report(request: ReportRequest) -> dict[str, Any]:
    try:
        return pipeline.generate_report(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc
