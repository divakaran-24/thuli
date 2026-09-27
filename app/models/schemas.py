"""Strict Pydantic schemas for all Analyst + Auditor data structures."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class AuditStatus(str, Enum):
    """Allowed verification statuses for claims assessed by the Auditor."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MISSING_CITATION = "MISSING_CITATION"


class SourceStatus(str, Enum):
    """Allowed lifecycle statuses for web sources."""

    FOUND = "FOUND"
    FETCHED = "FETCHED"
    FETCH_FAILED = "FETCH_FAILED"
    EXTRACTED = "EXTRACTED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class ResearchPlan(BaseModel):
    """Structured plan produced by the Planner Agent before searching."""

    question: str = Field(description="The original user research question.")
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities, companies, or subjects identified in the question.",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Logical breakdown of sub-questions required to answer the main question.",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Specific, targeted live search queries to execute on web search.",
    )
    required_evidence: List[str] = Field(
        default_factory=list,
        description="Types of factual evidence required (e.g. store counts, dates, official filings).",
    )
    verification_requirements: List[str] = Field(
        default_factory=list,
        description="Specific verification rules (e.g. cross-check single sources, verify numbers).",
    )
    parallel_tasks: List[str] = Field(
        default_factory=list,
        description="Queries or tasks that can execute concurrently in parallel.",
    )
    cross_check_targets: List[str] = Field(
        default_factory=list,
        description="High-priority claims or attributes that need multi-source confirmation.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Research plan question cannot be empty.")
        return v.strip()

    @field_validator("search_queries")
    @classmethod
    def queries_must_not_be_empty(cls, v: List[str]) -> List[str]:
        cleaned = [q.strip() for q in v if q.strip()]
        if not cleaned:
            raise ValueError("Research plan must contain at least one valid search query.")
        return cleaned


class SearchResult(BaseModel):
    """Structured result returned by the Web Search Tool."""

    title: str = Field(description="Title of the search result.")
    url: str = Field(description="Live URL of the search result.")
    snippet: str = Field(description="Summary snippet returned by search engine.")
    source_domain: str = Field(description="Extracted domain name of the source.")
    published_date: Optional[str] = Field(
        default=None, description="Published date string if available."
    )
    score: Optional[float] = Field(
        default=None, description="Search relevance score if provided."
    )
    source_status: SourceStatus = Field(
        default=SourceStatus.FOUND, description="Lifecycle status of this source."
    )
    source_type: str = Field(
        default="LIVE", description="'LIVE' web source or 'DEMO' test source."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if search/fetch failed."
    )

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"Invalid URL scheme: {v}")
        return v.strip()


class Evidence(BaseModel):
    """Structured piece of evidence extracted from a live fetched web page."""

    evidence_id: str = Field(description="Unique identifier for this evidence item.")
    source_id: str = Field(description="Identifier for the source document/page.")
    url: str = Field(description="Live URL the evidence was extracted from.")
    title: str = Field(description="Title of the source webpage.")
    publisher: str = Field(description="Publisher or organization domain name.")
    published_date: Optional[str] = Field(
        default=None, description="Publication date of the passage if available."
    )
    passage: str = Field(description="Extracted factual passage verbatim or near-verbatim.")
    retrieved_at: str = Field(description="ISO timestamp of when the page was retrieved.")
    source_type: str = Field(
        default="LIVE", description="'LIVE' or 'DEMO'."
    )
    source_status: SourceStatus = Field(
        default=SourceStatus.EXTRACTED, description="Status of the supporting source."
    )
    period_type: Optional[str] = Field(
        default=None, description="Timeframe granularity: 'FY', 'Q4', 'CY', 'H1', 'UNKNOWN'."
    )
    reported_period: Optional[str] = Field(
        default=None, description="Reported period, e.g. 'FY24', 'Q4 FY24', 'CY2024'."
    )
    period_start: Optional[str] = Field(
        default=None, description="Start date of period if known."
    )
    period_end: Optional[str] = Field(
        default=None, description="End date of period if known."
    )
    metric: Optional[str] = Field(
        default=None, description="Metric definition: 'net_store_additions', 'gross_store_openings', 'total_store_count', 'store_closures'."
    )
    business_segment: Optional[str] = Field(
        default=None, description="Business segment: 'Jewellery', 'Company-Wide', 'Watches'."
    )
    geography: Optional[str] = Field(
        default=None, description="Geographic scope: 'India', 'Global', 'Middle East'."
    )
    entity: Optional[str] = Field(
        default=None, description="Target company/entity name."
    )

    @field_validator("passage")
    @classmethod
    def passage_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Evidence passage cannot be empty.")
        return v.strip()


class Claim(BaseModel):
    """Factual claim synthesized from research evidence."""

    claim_id: str = Field(description="Unique identifier for the claim (e.g. 'c1').")
    text: str = Field(description="The precise factual statement.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of Evidence objects supporting this claim.",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Direct URLs or source references backing this claim.",
    )
    entity: Optional[str] = Field(
        default=None, description="Primary entity this claim pertains to."
    )
    metric: Optional[str] = Field(
        default=None, description="Metric asserted, e.g. 'net_store_additions', 'gross_store_openings'."
    )
    business_segment: Optional[str] = Field(
        default=None, description="Segment: 'Jewellery', 'Company-Wide'."
    )
    geography: Optional[str] = Field(
        default=None, description="Geography: 'India', 'Global'."
    )
    period: Optional[str] = Field(
        default=None, description="Reporting period: 'FY24', 'Q4 FY24', 'CY2024'."
    )
    source_type: str = Field(
        default="LIVE", description="'LIVE' or 'DEMO'."
    )
    verification_status: str = Field(
        default="PENDING", description="'VERIFIED', 'REJECTED', 'PENDING'."
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Reason if rejected by verification gate."
    )
    is_verified: bool = Field(
        default=False, description="Whether claim passed the deterministic verification gate."
    )
    confidence: Optional[float] = Field(
        default=1.0, description="Verification confidence score (0.0 to 1.0)."
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text cannot be empty.")
        return v.strip()



class AuditResult(BaseModel):
    """Independent assessment of a single claim by the Auditor Agent."""

    claim_id: str = Field(description="ID of the claim being audited.")
    claim: str = Field(description="Text of the claim evaluated.")
    status: AuditStatus = Field(description="Auditor decision status.")
    citation_present: bool = Field(
        description="Whether a valid URL/citation was provided for the claim."
    )
    source_url: Optional[str] = Field(
        default=None, description="URL of the cited source fetched for verification."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Relevant passage found in the cited source (or None if missing).",
    )
    reason: str = Field(
        description="Detailed rationale for the audit status (e.g. why supported or contradicted)."
    )


class ConflictRecord(BaseModel):
    """Record of disagreement detected between credible sources and how it was handled."""

    conflict_id: str = Field(description="Unique identifier for this conflict.")
    entity: str = Field(description="Entity or metric involved in the disagreement.")
    attribute: str = Field(description="Property or metric under dispute (e.g. 'store_openings').")
    source_a: str = Field(description="First source URL or title.")
    source_b: str = Field(description="Second source URL or title.")
    claim_a: str = Field(description="Figure/claim asserted by Source A.")
    claim_b: str = Field(description="Figure/claim asserted by Source B.")
    discrepancy_type: str = Field(
        description="Cause of discrepancy: reporting_date, scope, definition, FY_vs_CY, or unresolved."
    )
    resolution: str = Field(description="Explanation of resolution or note that it remains unresolved.")
    resolved: bool = Field(description="Whether the conflict was resolved with evidence.")


class AnalystAnswer(BaseModel):
    """Final synthesized research answer produced by the Analyst."""

    answer: str = Field(description="Concise, factual research answer with inline citations.")
    claims: List[Claim] = Field(
        default_factory=list, description="All individual factual claims extracted from the answer."
    )
    unanswered_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the question that could not be established from available sources.",
    )
    mode: str = Field(
        default="NORMAL_ANALYST", description="Analyst mode: NORMAL_ANALYST or ADVERSARIAL_ANALYST."
    )


class AuditPolicy(BaseModel):
    """Learned policy rule derived from audit failures to improve future analyst runs."""

    policy_id: str = Field(description="Unique ID of the policy rule.")
    rule: str = Field(description="Concise operational rule (e.g. 'Numerical claims require direct evidence').")
    source_audit_id: str = Field(description="Audit result that triggered this policy.")
    created_at: str = Field(description="ISO timestamp of when the policy was recorded.")


class RunMetrics(BaseModel):
    """Comprehensive performance and cost metrics for a single research question run."""

    question_id: str = Field(description="Unique question identifier (e.g. 'q01').")
    latency_ms: float = Field(description="Total wall-clock duration in milliseconds.")
    input_tokens: int = Field(default=0, description="Total input tokens across all LLM calls.")
    output_tokens: int = Field(default=0, description="Total output tokens across all LLM calls.")
    total_tokens: int = Field(default=0, description="Total tokens consumed.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD.")
    estimated_cost_inr: float = Field(default=0.0, description="Estimated cost in INR.")
    llm_calls: int = Field(default=0, description="Count of LLM completions made.")
    search_calls: int = Field(default=0, description="Count of search queries executed.")
    fetch_calls: int = Field(default=0, description="Count of web pages fetched.")
    memory_hits: int = Field(default=0, description="Entities/claims reused from memory.")
    memory_misses: int = Field(default=0, description="Entities/claims researched fresh.")
    failures: List[str] = Field(default_factory=list, description="Any recorded non-fatal failures.")


class Budget(BaseModel):
    """Operational limits and accounting for a research execution."""

    max_llm_calls: int = 12
    max_search_calls: int = 10
    max_fetch_calls: int = 10
    max_input_tokens: int = 50000
    max_output_tokens: int = 10000
    max_time_seconds: float = 120.0

    # Current usage
    llm_calls: int = 0
    search_calls: int = 0
    fetch_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = Field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        """Return elapsed wall-clock seconds since budget was created."""
        return time.time() - self.start_time

    def remaining_seconds(self) -> float:
        """Return remaining wall-clock seconds before hard timeout."""
        remaining = self.max_time_seconds - self.elapsed_seconds()
        return max(0.0, remaining)

    def is_time_exceeded(self) -> bool:
        """Return True if the 120s wall-clock limit has been exceeded."""
        return self.elapsed_seconds() >= self.max_time_seconds

    def can_call_llm(self) -> bool:
        """Check if an additional LLM call is within budget."""
        return (
            self.llm_calls < self.max_llm_calls
            and self.input_tokens < self.max_input_tokens
            and self.output_tokens < self.max_output_tokens
            and not self.is_time_exceeded()
        )

    def can_search(self) -> bool:
        """Check if an additional search query is within budget."""
        return self.search_calls < self.max_search_calls and not self.is_time_exceeded()

    def can_fetch(self) -> bool:
        """Check if an additional page fetch is within budget."""
        return self.fetch_calls < self.max_fetch_calls and not self.is_time_exceeded()

    def consume_llm(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM call and token consumption."""
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def consume_search(self, count: int = 1) -> None:
        """Record search call consumption."""
        self.search_calls += count

    def consume_fetch(self, count: int = 1) -> None:
        """Record fetch call consumption."""
        self.fetch_calls += count

    def calculate_cost(
        self,
        cost_per_m_in: float = 0.15,
        cost_per_m_out: float = 0.60,
        search_cost: float = 0.005,
        fetch_cost: float = 0.005,
        usd_to_inr: float = 86.50,
    ) -> tuple[float, float]:
        """Compute estimated cost in USD and INR deterministically."""
        token_cost_usd = (self.input_tokens / 1_000_000.0) * cost_per_m_in + (
            self.output_tokens / 1_000_000.0
        ) * cost_per_m_out
        tool_cost_usd = (self.search_calls * search_cost) + (self.fetch_calls * fetch_cost)
        total_usd = token_cost_usd + tool_cost_usd
        total_inr = total_usd * usd_to_inr
        return round(total_usd, 6), round(total_inr, 4)
