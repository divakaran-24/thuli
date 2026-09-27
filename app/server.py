"""FastAPI web server providing REST endpoints and serving the modern web UI."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.memory.database import DatabaseManager
from app.memory.research_memory import ResearchMemoryManager
from app.models.schemas import AnalystAnswer, AuditResult, Claim, Evidence, ResearchPlan, RunMetrics
from app.orchestration.workflow import ResearchWorkflow

logger = logging.getLogger("server")

app = FastAPI(
    title="Analyst + Auditor Research Agent",
    description="Live AI intelligence platform with strict verification, source-grounded claim gating, and independent auditing.",
    version="2.0.0",
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

# Single workflow and memory instance
db_manager = DatabaseManager(db_path=settings.database_path)
memory_manager = ResearchMemoryManager(db_manager)
workflow = ResearchWorkflow(memory_manager=memory_manager)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The research question to analyze.")
    mode: str = Field(default="NORMAL_ANALYST", description="Mode: NORMAL_ANALYST or ADVERSARIAL_ANALYST")
    question_id: Optional[str] = Field(default=None, description="Optional custom question ID")


def _serialize_dataclass(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_dataclass(v) for v in obj]
    return obj


def serialize_research_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert state with Pydantic and dataclass objects into clean JSON-serializable dict."""
    answer = state.get("answer")
    plan = state.get("plan")
    metrics = state.get("metrics")
    audits = state.get("audit_results", [])
    evidences = state.get("evidences", [])
    verified_claims = state.get("verified_claims", [])
    search_results = state.get("search_results", [])
    fetched_pages = state.get("fetched_pages", {})

    serialized_fetched = {}
    for url, page in fetched_pages.items():
        if dataclasses.is_dataclass(page):
            d = dataclasses.asdict(page)
            # Shorten very long text for network payload efficiency
            if "text" in d and len(d["text"]) > 1000:
                d["text_preview"] = d["text"][:1000] + "..."
            serialized_fetched[url] = d
        elif isinstance(page, dict):
            serialized_fetched[url] = page

    return {
        "question_id": state.get("question_id"),
        "question": state.get("question"),
        "mode": state.get("mode", "NORMAL_ANALYST"),
        "error": state.get("error"),
        "failures": state.get("failures", []),
        "answer": answer.model_dump() if isinstance(answer, AnalystAnswer) else answer,
        "plan": plan.model_dump() if isinstance(plan, ResearchPlan) else plan,
        "metrics": metrics.model_dump() if isinstance(metrics, RunMetrics) else metrics,
        "audit_results": [a.model_dump() if isinstance(a, AuditResult) else a for a in audits],
        "evidences": [e.model_dump() if isinstance(e, Evidence) else e for e in evidences],
        "verified_claims": [c.model_dump() if isinstance(c, Claim) else c for c in verified_claims],
        "search_results": [s.model_dump() if hasattr(s, "model_dump") else s for s in search_results],
        "fetched_pages": serialized_fetched,
        "learned_policies": state.get("learned_policies", []),
    }


# =====================================================================
# REST Endpoints
# =====================================================================

@app.get("/api/status")
async def get_system_status() -> Dict[str, Any]:
    """Return system readiness, API key availability, and memory stats."""
    return {
        "status": "ready",
        "has_gemini_key": settings.has_valid_gemini_key,
        "has_tavily_key": settings.has_valid_tavily_key,
        "database_path": settings.database_path,
        "gemini_model": settings.gemini_model,
        "adversarial_model": settings.gemini_adversarial_model,
        "usd_to_inr_rate": settings.usd_to_inr_rate,
        "usd_to_inr_date": settings.usd_to_inr_date,
    }


@app.post("/api/research")
async def execute_research(payload: ResearchRequest) -> Dict[str, Any]:
    """Execute end-to-end research graph and return structured verified results."""
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    qid = payload.question_id or f"q_{uuid.uuid4().hex[:8]}"
    mode = payload.mode.upper().strip()
    if mode not in {"NORMAL_ANALYST", "ADVERSARIAL_ANALYST"}:
        mode = "NORMAL_ANALYST"

    logger.info(f"Received research request [{qid}]: '{q}' (mode: {mode})")

    try:
        final_state = await workflow.execute_question(
            question=q,
            question_id=qid,
            mode=mode,
        )
        return serialize_research_state(final_state)
    except Exception as exc:
        logger.error(f"Error executing research workflow for [{qid}]: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history")
async def get_recent_history(limit: int = Query(default=15, ge=1, le=50)) -> Dict[str, Any]:
    """Retrieve recent research runs and questions from SQLite memory."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, question_id, question, mode, latency_ms, total_tokens,
                       estimated_cost_usd, estimated_cost_inr, llm_calls, search_calls,
                       fetch_calls, status, created_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            history = [
                {
                    "run_id": r[0],
                    "question_id": r[1],
                    "question": r[2],
                    "mode": r[3],
                    "latency_ms": r[4],
                    "total_tokens": r[5],
                    "estimated_cost_usd": r[6],
                    "estimated_cost_inr": r[7],
                    "llm_calls": r[8],
                    "search_calls": r[9],
                    "fetch_calls": r[10],
                    "status": r[11],
                    "created_at": r[12],
                }
                for r in rows
            ]
            return {"history": history, "count": len(history)}
    except Exception as exc:
        logger.warning(f"Error fetching run history: {exc}")
        return {"history": [], "count": 0, "error": str(exc)}


@app.get("/api/policies")
async def get_learned_policies() -> Dict[str, Any]:
    """Retrieve active operational feedback rules derived from auditor loops."""
    policies = memory_manager.get_feedback_policies(limit=10)
    return {"policies": policies, "count": len(policies)}


# =====================================================================
# Static File & UI Serving
# =====================================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page application UI."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI index.html not found.")
    return FileResponse(index_file)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start uvicorn server programmatically."""
    settings.ensure_directories()
    uvicorn.run("app.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
