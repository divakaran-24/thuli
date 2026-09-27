"""LangGraph research workflow connecting Planner, Search, Fetch, Extraction, Analyst, Auditor, and Memory."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from langgraph.graph import END, START, StateGraph

from app.agents.analyst import AnalystAgent
from app.agents.auditor import AuditorAgent
from app.agents.planner import PlannerAgent
from app.config import settings
from app.memory.research_memory import ResearchMemoryManager
from app.models.schemas import (
    AnalystAnswer,
    AuditResult,
    AuditStatus,
    Budget,
    Claim,
    ConflictRecord,
    Evidence,
    ResearchPlan,
    RunMetrics,
    SearchResult,
)
from app.logging.trace_logger import TraceLogger
from app.orchestration.state import ResearchState
from app.tools.extraction import EvidenceExtractor
from app.tools.fetch import FetchedPage, PageFetcher
from app.tools.search import WebSearchTool
from app.verification.claim_extractor import ClaimExtractor
from app.verification.claim_verifier import ClaimVerifier
from app.verification.conflict_resolver import ConflictResolver

logger = logging.getLogger(__name__)


def generate_feedback_rules_from_audits(audit_results: List[AuditResult]) -> List[str]:
    """Convert audit failures into compact, actionable research policies."""
    rules = []
    seen = set()

    for ar in audit_results:
        rule = None
        reason_lower = ar.reason.lower()

        if ar.status == AuditStatus.MISSING_CITATION:
            rule = "All factual statements must include an explicit, verifiable source URL citation."
        elif ar.status == AuditStatus.CONTRADICTED:
            if any(w in reason_lower for w in ["numerical", "store", "value", "count", "number"]):
                rule = "Numerical metrics and counts must strictly match primary disclosures; conflicting numbers must be reconciled."
            elif any(w in reason_lower for w in ["date", "year", "fy", "timeframe"]):
                rule = "Dates, fiscal years, and time periods must be verified directly against source text."
            else:
                rule = "Claims must accurately represent the cited source text without exaggeration or speculation."
        elif ar.status == AuditStatus.UNSUPPORTED:
            if "inaccessible" in reason_lower or "fetch" in reason_lower:
                rule = "Only cite accessible, live primary web sources and avoid broken or paywalled links."
            else:
                rule = "Factual claims require direct passage-level evidence in the cited source."

        if rule and rule not in seen:
            seen.add(rule)
            rules.append(rule)

    return rules


class ResearchWorkflow:
    """Orchestrates the end-to-end research graph."""

    def __init__(
        self,
        memory_manager: Optional[ResearchMemoryManager] = None,
        search_tool: Optional[WebSearchTool] = None,
        page_fetcher: Optional[PageFetcher] = None,
        planner: Optional[PlannerAgent] = None,
        analyst: Optional[AnalystAgent] = None,
        auditor: Optional[AuditorAgent] = None,
    ) -> None:
        """Initialize all components and compile LangGraph."""
        self.memory = memory_manager or ResearchMemoryManager()
        self.search_tool = search_tool or WebSearchTool()
        self.page_fetcher = page_fetcher or PageFetcher()
        self.planner = planner or PlannerAgent()
        self.analyst = analyst or AnalystAgent()
        self.auditor = auditor or AuditorAgent(page_fetcher=self.page_fetcher)
        self.extractor = EvidenceExtractor()
        self.claim_verifier = ClaimVerifier()
        self.conflict_resolver = ConflictResolver()

        self.graph = self._build_graph()

    # =================================================================
    # Workflow Nodes
    # =================================================================

    async def init_memory_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 1: Initialize budget, start time, and lookup existing audit feedback policies."""
        budget = state.get("budget") or Budget(
            max_time_seconds=settings.max_time_seconds,
            max_llm_calls=settings.max_llm_calls,
            max_search_calls=settings.max_search_calls,
            max_fetch_calls=settings.max_fetch_calls,
        )
        policies = self.memory.get_feedback_policies(limit=4)
        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="init_memory",
                agent="ResearchMemoryManager",
                action="load_policies_and_budget",
                output_summary={"policies_loaded": len(policies), "policies": policies},
            )
        return {
            "budget": budget,
            "start_time_monotonic": time.perf_counter(),
            "audit_policies": policies,
            "failures": state.get("failures", []),
        }

    async def planner_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 2: Formulate ResearchPlan incorporating feedback policies."""
        question = state["question"]
        budget = state["budget"]

        if not budget.can_call_llm():
            logger.warning("Budget limit reached before planning.")
            return {"failures": state.get("failures", []) + ["Budget exceeded before planning"]}

        # Preliminary entity extraction from question
        prelim_entities, hits, misses = self.memory.lookup_entities([question])

        plan, meta = await asyncio.to_thread(
            self.planner.plan,
            question=question,
            known_entities=prelim_entities,
            audit_policies=state.get("audit_policies", []),
        )

        budget.consume_llm(meta["input_tokens"], meta["output_tokens"])

        # Detailed entity lookup based on Planner's identified entities
        known_entities, hits, misses = self.memory.lookup_entities(plan.entities)

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="planning",
                agent="PlannerAgent",
                action="formulate_research_plan",
                input_summary={"question": question, "known_entities_count": len(prelim_entities)},
                output_summary={
                    "entities": plan.entities,
                    "queries": plan.search_queries,
                    "sub_questions": plan.sub_questions,
                    "verification_requirements": plan.verification_requirements,
                },
                latency_ms=meta.get("latency_ms", 0.0),
            )

        return {
            "plan": plan,
            "known_entities": known_entities,
            "memory_hits": hits,
            "memory_misses": misses,
            "search_queries": plan.search_queries,
            "budget": budget,
        }

    async def parallel_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 3: Execute independent search queries in parallel."""
        queries = state.get("search_queries", [])
        budget = state["budget"]
        failures = list(state.get("failures", []))

        # Filter queries to respect search budget
        allowed_queries = queries[: max(0, budget.max_search_calls - budget.search_calls)]
        if not allowed_queries:
            return {"search_results": []}

        # Run parallel searches
        search_tasks = [self.search_tool.search(q, max_results=3) for q in allowed_queries]
        search_outputs = await asyncio.gather(*search_tasks, return_exceptions=True)

        budget.consume_search(len(allowed_queries))
        all_results: List[SearchResult] = []
        seen_urls: Set[str] = set()

        for idx, out in enumerate(search_outputs):
            if isinstance(out, Exception):
                failures.append(f"Search query '{allowed_queries[idx]}' failed: {out}")
                continue
            results, meta = out
            if meta.get("error"):
                failures.append(f"Search error ({meta.get('query')}): {meta.get('error')}")
            for res in results:
                norm_url = res.url.rstrip("/")
                if norm_url not in seen_urls:
                    seen_urls.add(norm_url)
                    all_results.append(res)

        # Sort combined results by score
        all_results.sort(key=lambda r: (r.score or 0.0), reverse=True)

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="search",
                agent="WebSearchTool",
                action="parallel_web_search",
                input_summary={"queries": allowed_queries},
                output_summary={"results_count": len(all_results), "top_domains": [r.source_domain for r in all_results[:5]]},
            )

        return {"search_results": all_results, "budget": budget, "failures": failures}

    async def parallel_fetch_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 4: Fetch target web pages in parallel using Tavily Extract + httpx fallback."""
        results = state.get("search_results", [])
        budget = state["budget"]
        failures = list(state.get("failures", []))

        # Select top unique URLs within fetch budget
        max_fetch = min(len(results), max(0, budget.max_fetch_calls - budget.fetch_calls), 6)
        target_urls = [r.url for r in results[:max_fetch]]

        if not target_urls:
            return {"fetched_pages": {}}

        fetch_tasks = [self.page_fetcher.fetch_page(u) for u in target_urls]
        fetched_pages_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        budget.consume_fetch(len(target_urls))
        page_dict: Dict[str, FetchedPage] = {}

        for idx, page in enumerate(fetched_pages_list):
            url = target_urls[idx]
            if isinstance(page, Exception):
                failures.append(f"Fetch exception for {url}: {page}")
                continue
            if not page.success:
                failures.append(f"Fetch failure for {url}: {page.error}")
            page_dict[url] = page

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="fetch",
                agent="PageFetcher",
                action="parallel_page_fetch",
                input_summary={"target_urls": target_urls},
                output_summary={"fetched_count": len(page_dict), "successful_urls": [u for u, p in page_dict.items() if p.success]},
                error="; ".join(failures) if failures else None,
            )

        return {"fetched_pages": page_dict, "budget": budget, "failures": failures}

    async def extraction_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 5: Extract concise factual evidence passages deterministically."""
        pages = state.get("fetched_pages", {})
        question = state["question"]
        plan = state.get("plan")
        entities = plan.entities if plan else []

        all_evidence: List[Evidence] = []
        for url, page in pages.items():
            if page.success:
                evs = self.extractor.extract_passages(
                    page=page,
                    question=question,
                    entities=entities,
                )
                all_evidence.extend(evs)

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="extraction",
                agent="EvidenceExtractor",
                action="extract_evidence_passages",
                output_summary={"extracted_evidence_count": len(all_evidence)},
            )

        return {"evidences": all_evidence}

    async def verification_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 6: Claim generation and conflict resolution."""
        evidences = state.get("evidences", [])
        plan = state.get("plan")
        entities = plan.entities if plan else []

        # Generate candidate claims from evidence
        claims = ClaimExtractor.extract_claims_from_evidence(evidences)

        # Detect potential conflicts between evidence sources
        conflicts: List[ConflictRecord] = []
        if len(evidences) >= 2:
            for i in range(len(evidences)):
                for j in range(i + 1, min(len(evidences), i + 3)):
                    ev_a = evidences[i]
                    ev_b = evidences[j]
                    if ev_a.publisher != ev_b.publisher:
                        rec = self.conflict_resolver.create_conflict_record(
                            conflict_id=f"conf_{i}_{j}",
                            entity=entities[0] if entities else "Retailer",
                            attribute="metric",
                            claim_a=ev_a.passage[:80],
                            claim_b=ev_b.passage[:80],
                            evidence_a=ev_a,
                            evidence_b=ev_b,
                        )
                        if rec.discrepancy_type != "unresolved":
                            conflicts.append(rec)

        # Verify claims deterministically
        evidence_map = {ev.evidence_id: ev for ev in evidences}
        fetched_urls = set(state.get("fetched_pages", {}).keys())
        verified: List[Claim] = []
        for c in claims:
            ok, _ = self.claim_verifier.verify_claim_deterministically(c, evidence_map, fetched_urls)
            if ok:
                verified.append(c)

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="verification",
                agent="ClaimVerifier",
                action="verify_claims_and_conflicts",
                output_summary={"verified_claims_count": len(verified), "conflicts_detected": len(conflicts)},
            )

        return {"verified_claims": verified, "conflicts": conflicts}

    async def analyst_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 7: Synthesize final research answer with citations."""
        question = state["question"]
        plan = state.get("plan")
        evidences = state.get("evidences", [])
        conflicts = state.get("conflicts", [])
        known_entities = state.get("known_entities", [])
        mode = state.get("mode", "NORMAL_ANALYST")
        budget = state["budget"]

        if not budget.can_call_llm():
            fallback_answer = AnalystAnswer(
                answer="Research stopped: Time or token budget limit reached.",
                claims=[],
                unanswered_aspects=[question],
                mode=mode,
            )
            return {"answer": fallback_answer}

        answer, meta = await asyncio.to_thread(
            self.analyst.synthesize,
            question=question,
            plan=plan or ResearchPlan(question=question, search_queries=[question]),
            evidence=evidences,
            conflicts=conflicts,
            memory_facts=known_entities,
            mode=mode,
        )

        budget.consume_llm(meta["input_tokens"], meta["output_tokens"])

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="synthesis",
                agent="AnalystAgent",
                action="synthesize_answer",
                input_summary={"mode": mode, "evidence_count": len(evidences)},
                output_summary={"answer_length": len(answer.answer), "claims_count": len(answer.claims)},
                latency_ms=meta.get("latency_ms", 0.0),
            )

        return {"answer": answer, "budget": budget}

    async def auditor_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 8: Independently verify every cited claim."""
        answer = state.get("answer")
        budget = state["budget"]
        fetched_pages = dict(state.get("fetched_pages", {}))

        if not answer or not answer.claims:
            return {"audit_results": []}

        audit_results, meta = await self.auditor.audit_answer(
            answer=answer,
            cached_pages=fetched_pages,
        )

        budget.consume_llm(meta["input_tokens"], meta["output_tokens"])

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="auditing",
                agent="AuditorAgent",
                action="independent_audit",
                output_summary=meta,
                latency_ms=meta.get("latency_ms", 0.0),
            )

        return {"audit_results": audit_results, "budget": budget}

    async def feedback_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 9: Convert audit results into compact learned policies and store in SQLite."""
        audit_results = state.get("audit_results", [])
        learned_rules = generate_feedback_rules_from_audits(audit_results)

        # Store compact rules in SQLite memory
        for rule in learned_rules:
            self.memory.save_feedback_policy(
                rule=rule,
                source_audit_id=state.get("question_id", "q"),
                trigger_reason="Auditor feedback loop",
            )

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="feedback",
                agent="LearningLoop",
                action="derive_operational_policies",
                output_summary={"learned_rules": learned_rules},
            )

        return {"learned_policies": learned_rules}

    async def memory_update_node(self, state: ResearchState) -> Dict[str, Any]:
        """Node 10: Persist verified run knowledge, metrics, and update SQLite."""
        budget = state["budget"]
        start_time = state.get("start_time_monotonic", time.perf_counter())
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        cost_usd, cost_inr = budget.calculate_cost(
            cost_per_m_in=settings.cost_per_million_input_tokens,
            cost_per_m_out=settings.cost_per_million_output_tokens,
            search_cost=settings.tavily_cost_per_search,
            fetch_cost=settings.tavily_cost_per_extract,
            usd_to_inr=settings.usd_to_inr_rate,
        )

        metrics = RunMetrics(
            question_id=state.get("question_id", "q_unknown"),
            latency_ms=round(latency_ms, 2),
            input_tokens=budget.input_tokens,
            output_tokens=budget.output_tokens,
            total_tokens=budget.input_tokens + budget.output_tokens,
            estimated_cost_usd=cost_usd,
            estimated_cost_inr=cost_inr,
            llm_calls=budget.llm_calls,
            search_calls=budget.search_calls,
            fetch_calls=budget.fetch_calls,
            memory_hits=state.get("memory_hits", 0),
            memory_misses=state.get("memory_misses", 0),
            failures=state.get("failures", []),
        )

        plan = state.get("plan")
        primary_entities = plan.entities if plan else []
        answer = state.get("answer")
        claims = answer.claims if answer else []

        self.memory.save_run_knowledge(
            question_id=metrics.question_id,
            question=state["question"],
            mode=state.get("mode", "NORMAL_ANALYST"),
            metrics=metrics,
            claims=claims,
            evidences=state.get("evidences", []),
            audit_results=state.get("audit_results", []),
            primary_entities=primary_entities,
        )

        tl = state.get("trace_logger")
        if tl:
            tl.log_event(
                stage="memory_update",
                agent="ResearchMemoryManager",
                action="persist_run_knowledge",
                output_summary=metrics.model_dump(),
                latency_ms=metrics.latency_ms,
            )

        return {"metrics": metrics}

    # =================================================================
    # Graph Construction
    # =================================================================

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine."""
        workflow = StateGraph(ResearchState)

        workflow.add_node("init_memory", self.init_memory_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("search", self.parallel_search_node)
        workflow.add_node("fetch", self.parallel_fetch_node)
        workflow.add_node("extraction", self.extraction_node)
        workflow.add_node("verification", self.verification_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("auditor", self.auditor_node)
        workflow.add_node("feedback", self.feedback_node)
        workflow.add_node("memory_update", self.memory_update_node)

        # Edges
        workflow.add_edge(START, "init_memory")
        workflow.add_edge("init_memory", "planner")
        workflow.add_edge("planner", "search")
        workflow.add_edge("search", "fetch")
        workflow.add_edge("fetch", "extraction")
        workflow.add_edge("extraction", "verification")
        workflow.add_edge("verification", "analyst")
        workflow.add_edge("analyst", "auditor")
        workflow.add_edge("auditor", "feedback")
        workflow.add_edge("feedback", "memory_update")
        workflow.add_edge("memory_update", END)

        return workflow.compile()

    async def execute_question(
        self,
        question: str,
        question_id: str = "q01",
        mode: str = "NORMAL_ANALYST",
    ) -> ResearchState:
        """Run the research graph with the hard 120-second timeout guarantee."""
        trace_logger = TraceLogger(question_id=question_id)
        trace_logger.log_event(
            stage="workflow_start",
            agent="WorkflowOrchestrator",
            action="start_research",
            input_summary={"question": question, "mode": mode, "question_id": question_id},
        )

        initial_state: ResearchState = {
            "question_id": question_id,
            "question": question,
            "mode": mode,
            "failures": [],
            "trace_logger": trace_logger,
        }

        try:
            # Enforce 120s hard timeout
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state),
                timeout=settings.max_time_seconds,
            )
            trace_logger.log_event(
                stage="workflow_complete",
                agent="WorkflowOrchestrator",
                action="finish_research",
                output_summary={
                    "status": "COMPLETED",
                    "metrics": final_state.get("metrics").model_dump() if final_state.get("metrics") else None,
                },
            )
            return final_state
        except asyncio.TimeoutError:
            err_msg = f"HardTimeoutExceeded: Question execution exceeded {settings.max_time_seconds}s."
            logger.error(f"Execution of question {question_id} exceeded {settings.max_time_seconds}s limit!")
            trace_logger.log_event(
                stage="workflow_timeout",
                agent="WorkflowOrchestrator",
                action="timeout_abort",
                error=err_msg,
            )
            return {
                **initial_state,
                "error": err_msg,
                "failures": [f"Timeout after {settings.max_time_seconds}s"],
            }
        except Exception as exc:
            err_msg = f"WorkflowExecutionError: {type(exc).__name__}: {str(exc)}"
            logger.error(f"Execution of question {question_id} failed with error: {exc}", exc_info=True)
            trace_logger.log_event(
                stage="workflow_error",
                agent="WorkflowOrchestrator",
                action="error_abort",
                error=err_msg,
            )
            return {
                **initial_state,
                "error": err_msg,
                "failures": [str(exc)],
            }

