"""Unit tests for LangGraph research workflow and Auditor -> Feedback -> Analyst loop (Milestone 6)."""

import asyncio
import json
import os
import tempfile
import pytest

from app.agents.analyst import AnalystAgent
from app.agents.auditor import AuditorAgent
from app.agents.planner import PlannerAgent
from app.memory.database import DatabaseManager
from app.memory.research_memory import ResearchMemoryManager
from app.models.schemas import (
    AnalystAnswer,
    AuditResult,
    AuditStatus,
    Claim,
    Evidence,
    ResearchPlan,
)
from app.orchestration.workflow import (
    ResearchWorkflow,
    generate_feedback_rules_from_audits,
)
from app.tools.fetch import FetchedPage, PageFetcher
from app.tools.search import WebSearchTool


# =====================================================================
# 1. Feedback Rule Generation Tests
# =====================================================================

def test_generate_feedback_rules_from_audits():
    """Verify that audit failure statuses produce compact, actionable policy rules."""
    audits = [
        AuditResult(
            claim_id="c1",
            claim="Uncited claim",
            status=AuditStatus.MISSING_CITATION,
            citation_present=False,
            reason="No citation URL provided.",
        ),
        AuditResult(
            claim_id="c2",
            claim="Company opened 50 stores.",
            status=AuditStatus.CONTRADICTED,
            citation_present=True,
            reason="Numerical contradiction: Source states 35 stores.",
        ),
        AuditResult(
            claim_id="c3",
            claim="Vague claim",
            status=AuditStatus.UNSUPPORTED,
            citation_present=True,
            reason="Source does not mention store expansion.",
        ),
    ]

    rules = generate_feedback_rules_from_audits(audits)
    assert len(rules) == 3
    assert any("URL citation" in r for r in rules)
    assert any("Numerical metrics" in r for r in rules)
    assert any("direct passage-level evidence" in r for r in rules)


# =====================================================================
# 2. End-to-End Workflow & Learning Loop Tests
# =====================================================================

@pytest.mark.asyncio
async def test_workflow_end_to_end_and_feedback_learning_loop():
    """Verify complete LangGraph workflow execution and that audit feedback transfers to the next run."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        db_mgr = DatabaseManager(db_path=db_path)
        mem_mgr = ResearchMemoryManager(db_mgr)

        # Mock Planner
        mock_plan = ResearchPlan(
            question="Which Indian jewellery retailers opened the most stores in FY24?",
            entities=["Titan Company", "Kalyan Jewellers"],
            sub_questions=["Check store openings in FY24"],
            search_queries=["Titan store openings FY24", "Kalyan showroom additions FY24"],
            required_evidence=["Investor presentation store count"],
            verification_requirements=["Verify numbers directly from filings"],
            parallel_tasks=["Search Titan", "Search Kalyan"],
        )

        def mock_planner_llm(sys_prompt, user_prompt):
            return json.dumps(mock_plan.model_dump()), 400, 150

        planner = PlannerAgent(api_key="mock_key", llm_caller=mock_planner_llm)

        # Mock Search
        async def mock_search(query: str, max_results: int):
            if "Titan" in query:
                return [
                    {
                        "url": "https://titancompany.in/fy24-results",
                        "title": "Titan FY24 Results",
                        "snippet": "Titan added 60 stores.",
                        "score": 0.9,
                    }
                ]
            return [
                {
                    "url": "https://kalyanjewellers.net/q4",
                    "title": "Kalyan Q4 Results",
                    "snippet": "Kalyan opened 71 showrooms.",
                    "score": 0.9,
                }
            ]

        search_tool = WebSearchTool(api_key="mock_key", search_fn=mock_search)

        # Mock Fetch
        async def mock_httpx_fetch(url: str):
            if "titan" in url:
                html = "<html><title>Titan FY24</title><body><p>Titan added 60 stores in FY24.</p></body></html>"
            else:
                html = "<html><title>Kalyan Q4</title><body><p>Kalyan opened 71 showrooms in FY24.</p></body></html>"
            return html, 200

        page_fetcher = PageFetcher(tavily_api_key="mock_key", httpx_fetch_fn=mock_httpx_fetch)

        # Mock Analyst (produces 1 supported claim and 1 uncited claim to trigger audit feedback)
        mock_analyst_answer = {
            "answer": "Titan added 60 stores [https://titancompany.in/fy24-results] and unverified store claim.",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "Titan added 60 stores in FY24.",
                    "evidence_ids": ["ev_1"],
                    "citations": ["https://titancompany.in/fy24-results"],
                },
                {
                    "claim_id": "c2",
                    "text": "Kalyan opened unverified showrooms.",
                    "evidence_ids": [],
                    "citations": [],  # Intentionally missing citation to test feedback loop
                },
            ],
            "unanswered_aspects": [],
        }

        def mock_analyst_llm(sys_prompt, user_prompt):
            return json.dumps(mock_analyst_answer), 500, 160

        analyst = AnalystAgent(api_key="mock_key", llm_caller=mock_analyst_llm)

        # Mock Auditor
        def mock_auditor_llm(sys_prompt, user_prompt):
            return json.dumps({
                "status": "SUPPORTED",
                "reason": "Direct confirmation: Source states Titan added 60 stores.",
            }), 150, 40

        auditor = AuditorAgent(api_key="mock_key", page_fetcher=page_fetcher, llm_caller=mock_auditor_llm)

        workflow = ResearchWorkflow(
            memory_manager=mem_mgr,
            search_tool=search_tool,
            page_fetcher=page_fetcher,
            planner=planner,
            analyst=analyst,
            auditor=auditor,
        )

        # =============================================================
        # RUN 1: Execute Question 1
        # =============================================================
        state_1 = await workflow.execute_question(
            question="Which Indian jewellery retailers opened the most stores in FY24?",
            question_id="q01",
        )

        assert state_1.get("error") is None
        assert state_1["plan"] is not None
        assert len(state_1["search_results"]) >= 2
        assert len(state_1["evidences"]) >= 2
        assert state_1["answer"] is not None
        assert len(state_1["audit_results"]) == 2

        # Check that Auditor flagged the uncited claim as MISSING_CITATION
        audit_c2 = [ar for ar in state_1["audit_results"] if ar.claim_id == "c2"][0]
        assert audit_c2.status == AuditStatus.MISSING_CITATION

        # Check that Feedback node generated a rule for missing citation
        assert len(state_1["learned_policies"]) > 0
        assert any("citation" in p.lower() for p in state_1["learned_policies"])

        # Check metrics are computed
        metrics_1 = state_1["metrics"]
        assert metrics_1.total_tokens > 0
        assert metrics_1.estimated_cost_usd > 0.0

        # =============================================================
        # RUN 2: Execute Question 2 (Verify Closed-Loop Feedback Transfer)
        # =============================================================
        # Next run MUST automatically receive the learned policy from Run 1 via memory lookup
        state_2 = await workflow.execute_question(
            question="How many stores did Kalyan Jewellers add?",
            question_id="q02",
        )

        assert state_2.get("error") is None
        # Audit policies in Question 2 MUST contain the policy rule learned from Question 1!
        assert len(state_2["audit_policies"]) > 0
        assert any("citation" in p.lower() for p in state_2["audit_policies"])

        # Run 2 should also register a Memory HIT for Kalyan Jewellers!
        assert state_2["memory_hits"] >= 1

    finally:
        import gc
        gc.collect()
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_workflow_hard_timeout_handling():
    """Verify that workflow aborts cleanly and returns structured timeout error when time is exceeded."""
    import time

    class SlowWorkflow(ResearchWorkflow):
        async def init_memory_node(self, state):
            # Sleep longer than timeout
            await asyncio.sleep(0.3)
            return {}

    # Override timeout to 0.1s for this test
    from app.config import settings
    orig_timeout = settings.max_time_seconds
    settings.max_time_seconds = 0.1

    try:
        wf = SlowWorkflow()
        state = await wf.execute_question(
            question="Any question?",
            question_id="q_timeout",
        )
        assert "HardTimeoutExceeded" in state["error"]
        assert len(state["failures"]) > 0
    finally:
        settings.max_time_seconds = orig_timeout
