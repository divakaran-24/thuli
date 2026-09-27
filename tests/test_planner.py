"""Unit tests for ResearchPlan schema and Planner Agent (Milestone 1)."""

import json
import pytest
from pydantic import ValidationError

from app.agents.planner import PlannerAgent, PLANNER_SYSTEM_PROMPT
from app.models.schemas import (
    AuditResult,
    AuditStatus,
    Budget,
    Claim,
    Evidence,
    ResearchPlan,
    SearchResult,
)


# =====================================================================
# 1. Pydantic Schema Validation Tests
# =====================================================================

def test_research_plan_schema_valid():
    """Test creating a valid ResearchPlan with all required fields."""
    plan_data = {
        "question": "Which Indian jewellery retailers opened the most new stores in the last two years?",
        "entities": ["Titan Company (Tanishq)", "Kalyan Jewellers", "Malabar Gold & Diamonds"],
        "sub_questions": [
            "What were Titan/Tanishq net new store additions in FY23-FY25?",
            "What were Kalyan Jewellers net new store additions in FY23-FY25?",
            "Compare store openings across all three retailers.",
        ],
        "search_queries": [
            "Titan Company Tanishq annual report new store openings FY24 FY25",
            "Kalyan Jewellers showroom additions store count 2023 2024",
            "Malabar Gold store count expansion retail India 2024",
        ],
        "required_evidence": [
            "Official investor presentations with showroom counts",
            "Annual report retail footprint disclosures",
        ],
        "verification_requirements": [
            "All store counts must cite official investor filings",
            "Clarify net vs gross additions",
            "Cross-check news reports against company disclosures",
        ],
        "parallel_tasks": [
            "Search Titan store openings",
            "Search Kalyan store openings",
            "Search Malabar store openings",
        ],
        "cross_check_targets": ["Store count numbers", "Reporting period (FY vs CY)"],
    }
    plan = ResearchPlan.model_validate(plan_data)
    assert plan.question == plan_data["question"]
    assert len(plan.entities) == 3
    assert len(plan.search_queries) == 3
    assert len(plan.parallel_tasks) == 3


def test_research_plan_empty_question_fails():
    """Test that an empty question fails validation."""
    with pytest.raises(ValidationError):
        ResearchPlan(
            question="   ",
            entities=["Kalyan Jewellers"],
            sub_questions=["Subq 1"],
            search_queries=["Kalyan store openings"],
            required_evidence=["Annual report"],
            verification_requirements=["Check numbers"],
        )


def test_research_plan_empty_search_queries_fails():
    """Test that missing or empty search queries fails validation."""
    with pytest.raises(ValidationError):
        ResearchPlan(
            question="Valid question",
            entities=["Entity 1"],
            sub_questions=["Subq 1"],
            search_queries=[],
            required_evidence=["Evidence 1"],
            verification_requirements=["Verification 1"],
        )


def test_search_result_invalid_url_fails():
    """Test that a SearchResult with an invalid URL scheme fails validation."""
    with pytest.raises(ValidationError):
        SearchResult(
            title="Invalid Source",
            url="ftp://invalid-url.com",
            snippet="Some snippet",
            source_domain="invalid-url.com",
        )


def test_audit_result_allowed_statuses():
    """Test that AuditResult strictly enforces allowed audit statuses."""
    res = AuditResult(
        claim_id="c1",
        claim="Kalyan opened 35 stores",
        status=AuditStatus.SUPPORTED,
        citation_present=True,
        source_url="https://example.com/filing",
        evidence="Kalyan opened 35 showrooms in FY24.",
        reason="Direct match in annual report.",
    )
    assert res.status == AuditStatus.SUPPORTED

    with pytest.raises(ValidationError):
        AuditResult(
            claim_id="c2",
            claim="Kalyan opened 50 stores",
            status="MAYBE_CORRECT",  # Invalid status
            citation_present=True,
            reason="Uncertain",
        )


def test_budget_accounting_and_cost():
    """Test that Budget tracks calls, tokens, timeouts, and cost calculation."""
    budget = Budget(
        max_llm_calls=2,
        max_search_calls=3,
        max_fetch_calls=3,
        max_time_seconds=120.0,
    )
    assert budget.can_call_llm() is True
    assert budget.can_search() is True
    assert budget.can_fetch() is True

    budget.consume_llm(input_tokens=1000, output_tokens=200)
    assert budget.llm_calls == 1
    assert budget.input_tokens == 1000
    assert budget.output_tokens == 200

    budget.consume_search(1)
    budget.consume_fetch(1)

    cost_usd, cost_inr = budget.calculate_cost(usd_to_inr=86.50)
    assert cost_usd > 0
    assert cost_inr > 0
    assert round(cost_usd * 86.50, 4) == cost_inr

    # Exhaust llm calls
    budget.consume_llm(input_tokens=500, output_tokens=100)
    assert budget.can_call_llm() is False


# =====================================================================
# 2. Planner Agent Unit Tests
# =====================================================================

def test_planner_prompt_construction():
    """Test that the planner builds a prompt with entities and audit policies."""
    planner = PlannerAgent(api_key="mock_key")
    prompt = planner.build_user_prompt(
        question="What is the store count of Kalyan Jewellers?",
        known_entities=[
            {"name": "Kalyan Jewellers", "verified_claims": [{"text": "Has 250 showrooms in 2023"}]}
        ],
        audit_policies=["Numerical claims require direct evidence from primary sources."],
    )
    assert "Kalyan Jewellers" in prompt
    assert "Has 250 showrooms in 2023" in prompt
    assert "Numerical claims require direct evidence" in prompt
    assert "DO NOT answer the question" in prompt


def test_planner_plan_execution_with_mock_llm():
    """Test PlannerAgent.plan() using an injected mock LLM."""
    mock_plan_payload = {
        "question": "Which Indian jewellery retailers opened the most new stores in the last two years?",
        "entities": ["Titan Company", "Kalyan Jewellers", "Senco Gold"],
        "sub_questions": [
            "Find Titan retail showroom additions for FY24 and FY25",
            "Find Kalyan showroom additions for FY24 and FY25",
            "Compare numbers to determine top 3",
        ],
        "search_queries": [
            "Titan Company Tanishq store additions FY24 FY25 investor presentation",
            "Kalyan Jewellers showroom count annual report FY24",
            "Senco Gold new store openings 2023 2024",
        ],
        "required_evidence": [
            "Quarterly investor presentations",
            "BSE regulatory filings on store count",
        ],
        "verification_requirements": [
            "Verify net vs gross store additions",
            "Check reporting periods match (FY23-FY25)",
            "Cross-check news reports with company releases",
        ],
        "parallel_tasks": [
            "Query Titan store additions",
            "Query Kalyan store additions",
            "Query Senco Gold store additions",
        ],
        "cross_check_targets": ["Store opening counts"],
    }

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        assert "RESEARCH QUESTION" in user_prompt
        # Return json wrapped in markdown code fence to verify robust regex parsing
        text = f"```json\n{json.dumps(mock_plan_payload)}\n```"
        return text, 450, 180

    planner = PlannerAgent(api_key="mock_key", llm_caller=mock_llm)
    plan, metadata = planner.plan(
        question="Which Indian jewellery retailers opened the most new stores in the last two years?"
    )

    assert isinstance(plan, ResearchPlan)
    assert plan.question == mock_plan_payload["question"]
    assert len(plan.entities) == 3
    assert len(plan.search_queries) == 3
    assert len(plan.parallel_tasks) == 3
    assert "Titan Company" in plan.entities
    assert metadata["input_tokens"] == 450
    assert metadata["output_tokens"] == 180
    assert metadata["total_tokens"] == 630
    assert metadata["latency_ms"] >= 0


def test_planner_does_not_answer_question():
    """Verify that ResearchPlan does not have an answer field."""
    # ResearchPlan must be strictly a plan, not an answer container
    assert "answer" not in ResearchPlan.model_fields


def test_planner_handles_malformed_llm_json():
    """Test that PlannerAgent raises ValueError when LLM returns invalid JSON."""
    def bad_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return "I am unable to answer because I am just an AI.", 100, 20

    planner = PlannerAgent(api_key="mock_key", llm_caller=bad_llm)
    with pytest.raises(ValueError, match="not valid JSON"):
        planner.plan("Any question")
