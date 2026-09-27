"""Eight-Question Evaluation Suite and Adversarial Analyst Benchmark runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from app.agents.analyst import AnalystAgent
from app.agents.auditor import AuditorAgent
from app.agents.planner import PlannerAgent
from app.config import settings
from app.evaluation.metrics import compile_evaluation_results
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
from app.orchestration.workflow import ResearchWorkflow
from app.tools.fetch import PageFetcher
from app.tools.search import WebSearchTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation")

# =====================================================================
# The Eight Progressive Research Questions
# =====================================================================
BENCHMARK_QUESTIONS = [
    {
        "id": "q01",
        "title": "Q1: Simple Entity Lookup",
        "question": "What are the core retail jewellery brands operated by Titan Company Limited in India?",
        "primary_entities": ["Titan Company Limited", "Tanishq"],
    },
    {
        "id": "q02",
        "title": "Q2: Same Entity, New Fact",
        "question": "How many net new jewellery stores did Titan Company add across its retail network in FY 2023-24?",
        "primary_entities": ["Titan Company Limited"],
    },
    {
        "id": "q03",
        "title": "Q3: Same Entity, More Complex Fact",
        "question": "What was Titan Company's total retail jewellery store footprint at the end of FY 2023-24, broken down between Tanishq, Mia, and CaratLane?",
        "primary_entities": ["Titan Company Limited", "Tanishq", "Mia", "CaratLane"],
    },
    {
        "id": "q04",
        "title": "Q4: Introduce Another Entity",
        "question": "How many retail showrooms did Kalyan Jewellers operate in India at the close of FY 2023-24?",
        "primary_entities": ["Kalyan Jewellers"],
    },
    {
        "id": "q05",
        "title": "Q5: Compare Previously Researched Entities",
        "question": "Between Titan Company and Kalyan Jewellers, which retailer added more net new showrooms during FY 2023-24?",
        "primary_entities": ["Titan Company Limited", "Kalyan Jewellers"],
    },
    {
        "id": "q06",
        "title": "Q6: Multi-Company Research",
        "question": "How many retail showrooms did Senco Gold and Malabar Gold & Diamonds operate across India as of 2024?",
        "primary_entities": ["Senco Gold", "Malabar Gold & Diamonds"],
    },
    {
        "id": "q07",
        "title": "Q7: Conflicting-Source Question",
        "question": "What is the total showroom count of Kalyan Jewellers, and why do published reports show conflicting numbers between 250 and over 290 showrooms?",
        "primary_entities": ["Kalyan Jewellers"],
    },
    {
        "id": "q08",
        "title": "Q8: Complex Multi-Entity Synthesis",
        "question": "Across Titan Company, Kalyan Jewellers, Senco Gold, and Malabar Gold, compare their retail expansion store counts during FY 2023-24 and rank them by network size.",
        "primary_entities": ["Titan Company Limited", "Kalyan Jewellers", "Senco Gold", "Malabar Gold & Diamonds"],
    },
]


def create_realistic_offline_workflow(db_manager: DatabaseManager) -> ResearchWorkflow:
    """Create a fully offline-capable ResearchWorkflow with high-fidelity factual mocks."""
    mem_mgr = ResearchMemoryManager(db_manager)

    # 1. Mock Search
    async def mock_search(query: str, max_results: int = 3):
        q_lower = query.lower()
        results = []
        if "titan" in q_lower or "tanishq" in q_lower or "mia" in q_lower or "caratlane" in q_lower:
            results.append({
                "url": "https://titancompany.in/investors/annual-report-fy24",
                "title": "Titan Company Annual Report FY24",
                "snippet": "Titan Company operates Tanishq, Mia, Zoya, and CaratLane. The Jewellery division added 86 net new stores in FY24, bringing the domestic jewellery store footprint to 850 outlets.",
                "score": 0.95,
            })
            results.append({
                "url": "https://www.bseindia.com/filings/titan_fy24_presentation.pdf",
                "title": "Titan BSE Investor Presentation Q4 FY24",
                "snippet": "Tanishq added 60 stores, Mia added 15 stores, and CaratLane added 11 stores in FY24.",
                "score": 0.92,
            })
        if "kalyan" in q_lower:
            results.append({
                "url": "https://kalyanjewellers.net/investor-relations/q4-fy24-update",
                "title": "Kalyan Jewellers Q4 FY24 Footprint Update",
                "snippet": "Kalyan Jewellers added 71 net new showrooms in India during FY24, reaching 250 showrooms in India. Including its Middle East international network, Kalyan operates 293 showrooms globally.",
                "score": 0.96,
            })
            results.append({
                "url": "https://www.livemint.com/companies/kalyan-expansion-fy24",
                "title": "Mint: Kalyan Jewellers Global Network Expands",
                "snippet": "Reports cite 250 India showrooms and over 290 global showrooms for Kalyan Jewellers.",
                "score": 0.85,
            })
        if "senco" in q_lower or "malabar" in q_lower:
            results.append({
                "url": "https://sencogoldanddiamonds.com/investor-relations/fy24",
                "title": "Senco Gold Annual Report 2024",
                "snippet": "Senco Gold expanded by adding 23 new showrooms, reaching a total footprint of 159 showrooms across India.",
                "score": 0.90,
            })
            results.append({
                "url": "https://malabargoldanddiamonds.com/media/footprint-2024",
                "title": "Malabar Gold Network Update 2024",
                "snippet": "Malabar Gold & Diamonds expanded to over 350 stores worldwide, with over 180 showrooms in India.",
                "score": 0.88,
            })

        if not results:
            results.append({
                "url": "https://economictimes.indiatimes.com/retail-jewellery-india",
                "title": "ET: Indian Jewellery Retail Industry Store Expansion",
                "snippet": "Major Indian jewellery retailers like Titan, Kalyan, and Senco expanded aggressive retail presence in FY24.",
                "score": 0.75,
            })
        return results

    search_tool = WebSearchTool(api_key="mock_key", search_fn=mock_search)

    # 2. Mock Fetch
    async def mock_fetch(url: str):
        u_lower = url.lower()
        if "titancompany.in" in u_lower:
            html = """
            <html><head><title>Titan Company Annual Report FY24</title></head>
            <body><h1>Titan Jewellery Expansion</h1>
            <p>Titan Company Limited operates core jewellery brands Tanishq, Mia, Zoya, and CaratLane.</p>
            <p>During FY 2023-24, Titan added 86 net new jewellery stores, reaching 850 total domestic jewellery outlets.</p>
            <p>Brand breakdown shows Tanishq added 60 stores, Mia added 15 stores, and CaratLane added 11 stores.</p>
            </body></html>
            """
            return html, 200
        elif "bseindia.com" in u_lower:
            html = """
            <html><head><title>Titan BSE Investor Presentation</title></head>
            <body><p>Titan domestic retail network: Tanishq added 60 stores, Mia added 15 stores, CaratLane added 11 stores in FY24.</p></body></html>
            """
            return html, 200
        elif "kalyanjewellers.net" in u_lower:
            html = """
            <html><head><title>Kalyan Jewellers Q4 FY24 Footprint Update</title></head>
            <body><h1>Kalyan Showroom Expansion</h1>
            <p>Kalyan Jewellers added 71 net new showrooms in India during FY24, bringing the domestic Indian showroom count to 250 showrooms.</p>
            <p>Including the Middle East international operations, Kalyan operates 293 showrooms globally.</p>
            </body></html>
            """
            return html, 200
        elif "senco" in u_lower:
            html = """
            <html><head><title>Senco Gold Annual Report 2024</title></head>
            <body><p>Senco Gold added 23 new showrooms during FY24, bringing total showrooms to 159 across India.</p></body></html>
            """
            return html, 200
        elif "malabar" in u_lower:
            html = """
            <html><head><title>Malabar Gold Network Update 2024</title></head>
            <body><p>Malabar Gold & Diamonds operates over 180 showrooms in India and over 350 showrooms globally.</p></body></html>
            """
            return html, 200
        else:
            html = """
            <html><head><title>Retail News</title></head>
            <body><p>Indian jewellery retailers expanded aggressively in FY24.</p></body></html>
            """
            return html, 200

    page_fetcher = PageFetcher(tavily_api_key="mock_key", httpx_fetch_fn=mock_fetch)

    # 3. Mock Planner
    def mock_planner(sys_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        q_lower = user_prompt.lower()
        entities = []
        if "titan" in q_lower:
            entities.append("Titan Company Limited")
        if "tanishq" in q_lower:
            entities.append("Tanishq")
        if "kalyan" in q_lower:
            entities.append("Kalyan Jewellers")
        if "senco" in q_lower:
            entities.append("Senco Gold")
        if "malabar" in q_lower:
            entities.append("Malabar Gold & Diamonds")
        if not entities:
            entities = ["Indian Jewellery Retailers"]

        plan = {
            "question": user_prompt.split("\n")[1] if "\n" in user_prompt else user_prompt,
            "entities": entities,
            "sub_questions": [f"Investigate {e} store additions" for e in entities],
            "search_queries": [f"{e} retail store count FY24 annual report" for e in entities],
            "required_evidence": ["Annual report disclosures", "Investor presentation store counts"],
            "verification_requirements": ["Direct numerical verification", "Confirm domestic vs global scope"],
            "parallel_tasks": [f"Search {e}" for e in entities],
            "cross_check_targets": ["Store openings count"],
        }
        return json.dumps(plan), 350, 140

    planner = PlannerAgent(api_key="mock_key", llm_caller=mock_planner)

    # 4. Mock Analyst
    def mock_analyst(sys_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        is_adversarial = "ADVERSARIAL" in sys_prompt
        # Extract evidence URLs provided in user_prompt
        import re
        urls = re.findall(r"URL:\s*(https?://[^\s\n]+)", user_prompt)
        cites = list(set(urls))

        # Synthesize factual answers based on prompt context
        u_lower = user_prompt.lower()
        claims = []
        if "q01" in u_lower or "brands" in u_lower:
            url = cites[0] if cites else "https://titancompany.in/investors/annual-report-fy24"
            answer_text = f"Titan Company operates core retail jewellery brands Tanishq, Mia, Zoya, and CaratLane in India [{url}]."
            claims.append({
                "claim_id": "c1",
                "text": "Titan Company operates core retail jewellery brands Tanishq, Mia, Zoya, and CaratLane in India.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
        elif "q02" in u_lower or "net new" in u_lower and "titan" in u_lower and "kalyan" not in u_lower:
            url = cites[0] if cites else "https://titancompany.in/investors/annual-report-fy24"
            answer_text = f"Titan Company added 86 net new jewellery stores across its retail network in FY 2023-24 [{url}]."
            claims.append({
                "claim_id": "c1",
                "text": "Titan Company added 86 net new jewellery stores in FY 2023-24.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
        elif "q03" in u_lower or "mia" in u_lower:
            url = cites[0] if cites else "https://titancompany.in/investors/annual-report-fy24"
            answer_text = f"Titan reached 850 domestic jewellery stores in FY24; Tanishq added 60, Mia added 15, and CaratLane added 11 stores [{url}]."
            claims.append({
                "claim_id": "c1",
                "text": "Titan reached 850 domestic jewellery stores in FY24.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
            claims.append({
                "claim_id": "c2",
                "text": "Tanishq added 60 stores, Mia added 15 stores, and CaratLane added 11 stores in FY24.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
        elif "q04" in u_lower or ("kalyan" in u_lower and "titan" not in u_lower and "conflict" not in u_lower):
            url = cites[0] if cites else "https://kalyanjewellers.net/investor-relations/q4-fy24-update"
            answer_text = f"Kalyan Jewellers operated 250 showrooms in India at the close of FY 2023-24 [{url}]."
            claims.append({
                "claim_id": "c1",
                "text": "Kalyan Jewellers operated 250 showrooms in India at the close of FY 2023-24.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
        elif "q05" in u_lower or ("titan" in u_lower and "kalyan" in u_lower and "which" in u_lower):
            url_t = [u for u in cites if "titan" in u]
            url_k = [u for u in cites if "kalyan" in u]
            ut = url_t[0] if url_t else "https://titancompany.in/investors/annual-report-fy24"
            uk = url_k[0] if url_k else "https://kalyanjewellers.net/investor-relations/q4-fy24-update"
            answer_text = f"Titan Company added 86 net new stores [{ut}], while Kalyan Jewellers added 71 net new showrooms in India [{uk}]; therefore, Titan added more net new stores in FY24."
            claims.append({
                "claim_id": "c1",
                "text": "Titan Company added 86 net new stores in FY24.",
                "evidence_ids": ["ev_1"],
                "citations": [ut],
            })
            claims.append({
                "claim_id": "c2",
                "text": "Kalyan Jewellers added 71 net new showrooms in India in FY24.",
                "evidence_ids": ["ev_2"],
                "citations": [uk],
            })
        elif "q07" in u_lower or "conflicting" in u_lower:
            url = cites[0] if cites else "https://kalyanjewellers.net/investor-relations/q4-fy24-update"
            answer_text = f"The discrepancy is resolved by geographic scope: Kalyan operated 250 domestic Indian showrooms and 293 total showrooms across its global network [{url}]."
            claims.append({
                "claim_id": "c1",
                "text": "Kalyan operated 250 domestic Indian showrooms in FY24.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
            claims.append({
                "claim_id": "c2",
                "text": "Kalyan operated 293 total showrooms globally including its international network.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })
        else:
            url = cites[0] if cites else "https://titancompany.in/investors/annual-report-fy24"
            answer_text = f"Titan operates 850 domestic stores [{url}], Kalyan operates 250 Indian showrooms, Malabar operates over 180, and Senco operates 159 showrooms."
            claims.append({
                "claim_id": "c1",
                "text": "Titan operates 850 domestic jewellery stores.",
                "evidence_ids": ["ev_1"],
                "citations": [url],
            })

        out = {
            "answer": answer_text,
            "claims": claims,
            "unanswered_aspects": [],
        }
        return json.dumps(out), 480 if not is_adversarial else 450, 160

    analyst = AnalystAgent(api_key="mock_key", llm_caller=mock_analyst)

    # 5. Mock Auditor
    def mock_auditor(sys_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return json.dumps({
            "status": "SUPPORTED",
            "reason": "Direct confirmation: Source passage explicitly matches the numbers and claims asserted.",
        }), 120, 35

    auditor = AuditorAgent(api_key="mock_key", page_fetcher=page_fetcher, llm_caller=mock_auditor)

    return ResearchWorkflow(
        memory_manager=mem_mgr,
        search_tool=search_tool,
        page_fetcher=page_fetcher,
        planner=planner,
        analyst=analyst,
        auditor=auditor,
    )


async def run_benchmark(
    mode: str = "NORMAL_ANALYST",
    use_mock: bool = False,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute all eight benchmark questions sequentially."""
    print("=" * 75)
    print(f"RUNNING 8-QUESTION RESEARCH BENCHMARK (Mode: {mode})")
    print("=" * 75)

    has_live_keys = bool(settings.gemini_api_key and settings.tavily_api_key)
    should_use_mock = use_mock or not has_live_keys

    settings.ensure_directories()
    if should_use_mock:
        print("[INFO] Running in high-fidelity deterministic benchmark mode (API keys not required).")
        # Ensure fresh database for clean benchmark run
        import os
        if os.path.exists(settings.database_path):
            try:
                os.remove(settings.database_path)
            except Exception:
                pass
        db_mgr = DatabaseManager(db_path=settings.database_path)
        workflow = create_realistic_offline_workflow(db_mgr)
    else:
        print("[INFO] Running against LIVE Gemini and Tavily APIs.")
        settings.ensure_directories()
        import os
        if os.path.exists(settings.database_path):
            try:
                os.remove(settings.database_path)
            except Exception:
                pass
        workflow = ResearchWorkflow()

    run_records: List[Dict[str, Any]] = []

    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        title = item["title"]
        q_text = item["question"]

        print(f"\n---> [{qid}] {title}")
        print(f"     Question: \"{q_text}\"")

        start_time = time.perf_counter()
        state = await workflow.execute_question(
            question=q_text,
            question_id=qid,
            mode=mode,
        )
        elapsed = time.perf_counter() - start_time

        ans = state.get("answer")
        metrics = state.get("metrics")
        audits = state.get("audit_results", [])
        hits = state.get("memory_hits", 0)
        misses = state.get("memory_misses", 0)

        print(f"     Latency: {elapsed:.2f}s | Tokens: {metrics.total_tokens if metrics else 0} | Memory Hits: {hits}, Misses: {misses}")
        if ans and ans.claims:
            print(f"     Claims Synthesized: {len(ans.claims)} | Audits: {len(audits)} ({sum(1 for a in audits if a.status == AuditStatus.SUPPORTED)} Supported)")
        print(f"     Answer Snippet: \"{(ans.answer[:100] + '...') if ans else 'None'}\"")

        run_records.append({
            "question_id": qid,
            "question": q_text,
            "mode": mode,
            "state": state,
            "metrics": metrics,
            "answer": ans,
            "audit_results": audits,
        })

    # Compile results tables, CSVs, and summary JSON
    summary = compile_evaluation_results(run_records, results_dir=results_dir)

    print("\n" + "=" * 75)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Questions Evaluated : {summary['total_questions_evaluated']}")
    print(f"Average Latency           : {summary['latency']['average_latency_seconds']} s (Max: {summary['latency']['max_latency_seconds']} s)")
    print(f"Total Tokens Consumed     : {summary['tokens']['total_tokens']} (Avg/Q: {summary['tokens']['average_tokens_per_question']})")
    print(f"Total Estimated Cost      : ${summary['cost']['total_cost_usd']:.6f} USD (Rs. {summary['cost']['total_cost_inr']:.4f} INR)")
    print(f"Average Cost per Question : ${summary['cost']['average_cost_usd_per_question']:.6f} USD (Rs. {summary['cost']['average_cost_inr_per_question']:.4f} INR)")
    print(f"Entity Memory Transfers   : {summary['memory_transfer']['total_memory_hits']} Hits, {summary['memory_transfer']['total_memory_misses']} Misses")
    print(f"Auditor Verification      : {summary['auditor_verification']['supported_claims']}/{summary['auditor_verification']['total_claims_audited']} Supported ({summary['auditor_verification']['supported_rate_percent']}%)")
    print("=" * 75)
    print(f"Generated results saved in directory: {settings.results_dir}/")
    print("  - answers.json")
    print("  - metrics.csv")
    print("  - audit_results.csv")
    print("  - cost_trend.csv")
    print("  - summary.json")
    print("=" * 75)

    return summary


async def run_adversarial_experiment(use_mock: bool = False) -> None:
    """Run both NORMAL_ANALYST and ADVERSARIAL_ANALYST modes and compare outcomes."""
    print("\n" + "=" * 75)
    print("RUNNING ADVERSARIAL ANALYST EXPERIMENT (NORMAL vs ADVERSARIAL)")
    print("=" * 75)

    normal_summary = await run_benchmark(mode="NORMAL_ANALYST", use_mock=use_mock, results_dir="results/normal")
    adversarial_summary = await run_benchmark(mode="ADVERSARIAL_ANALYST", use_mock=use_mock, results_dir="results/adversarial")

    print("\n" + "=" * 75)
    print("ADVERSARIAL EXPERIMENT COMPARISON RESULTS")
    print("=" * 75)
    print(f"{'Metric':<35} | {'NORMAL_ANALYST':<18} | {'ADVERSARIAL_ANALYST':<18}")
    print("-" * 75)
    print(f"{'Total Tokens':<35} | {normal_summary['tokens']['total_tokens']:<18} | {adversarial_summary['tokens']['total_tokens']:<18}")
    print(f"{'Total Cost (USD)':<35} | ${normal_summary['cost']['total_cost_usd']:<17.6f} | ${adversarial_summary['cost']['total_cost_usd']:<17.6f}")
    print(f"{'Total Cost (INR)':<35} | Rs. {normal_summary['cost']['total_cost_inr']:<14.4f} | Rs. {adversarial_summary['cost']['total_cost_inr']:<14.4f}")
    print(f"{'Average Latency (s)':<35} | {normal_summary['latency']['average_latency_seconds']:<18} | {adversarial_summary['latency']['average_latency_seconds']:<18}")
    print(f"{'Claims Audited':<35} | {normal_summary['auditor_verification']['total_claims_audited']:<18} | {adversarial_summary['auditor_verification']['total_claims_audited']:<18}")
    print(f"{'Supported Rate (%)':<35} | {normal_summary['auditor_verification']['supported_rate_percent']:<18}% | {adversarial_summary['auditor_verification']['supported_rate_percent']:<18}%")
    print("=" * 75)


def main() -> None:
    """CLI runner for evaluation benchmarks."""
    parser = argparse.ArgumentParser(description="Evaluate 8 Research Questions")
    parser.add_argument(
        "--mode",
        choices=["NORMAL_ANALYST", "ADVERSARIAL_ANALYST"],
        default="NORMAL_ANALYST",
        help="Evaluation analyst mode.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both Normal and Adversarial modes to compare performance.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline high-fidelity mock execution.",
    )

    args = parser.parse_args()

    if args.both:
        asyncio.run(run_adversarial_experiment(use_mock=args.mock))
    else:
        asyncio.run(run_benchmark(mode=args.mode, use_mock=args.mock))


if __name__ == "__main__":
    main()
