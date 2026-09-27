"""State definition for LangGraph orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

from app.models.schemas import (
    AnalystAnswer,
    AuditPolicy,
    AuditResult,
    Budget,
    Claim,
    ConflictRecord,
    Evidence,
    ResearchPlan,
    RunMetrics,
    SearchResult,
)
from app.tools.fetch import FetchedPage


class ResearchState(TypedDict, total=False):
    """LangGraph state representing the complete Analyst + Auditor workflow."""

    # Run Identifiers & Inputs
    question_id: str
    question: str
    mode: str  # NORMAL_ANALYST or ADVERSARIAL_ANALYST

    # Budgets & Controls
    budget: Budget
    start_time_monotonic: float

    # Memory Lookup Phase
    known_entities: List[Dict[str, Any]]
    audit_policies: List[str]
    memory_hits: int
    memory_misses: int

    # Planning Phase
    plan: Optional[ResearchPlan]

    # Search & Fetch Phase
    search_queries: List[str]
    search_results: List[SearchResult]
    fetched_pages: Dict[str, FetchedPage]

    # Evidence & Verification Phase
    evidences: List[Evidence]
    verified_claims: List[Claim]
    conflicts: List[ConflictRecord]

    # Analyst Synthesis Phase
    answer: Optional[AnalystAnswer]

    # Auditor Phase
    audit_results: List[AuditResult]

    # Feedback & Learning Loop Phase
    learned_policies: List[str]

    # Logging
    trace_logger: Any

    # Final Metrics & Failures
    metrics: Optional[RunMetrics]
    failures: List[str]
    error: Optional[str]
