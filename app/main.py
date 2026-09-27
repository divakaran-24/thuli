"""CLI entrypoint for running research questions with Analyst + Auditor Research Agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from app.config import settings
from app.orchestration.workflow import ResearchWorkflow


async def run_cli(question: str, question_id: str = "q01", mode: str = "NORMAL_ANALYST") -> int:
    """Execute research workflow for a single question from the CLI."""
    settings.ensure_directories()
    print("=" * 70)
    print(f"ANALYST + AUDITOR RESEARCH AGENT (Mode: {mode})")
    print(f"Question ID : {question_id}")
    print(f"Question    : {question}")
    if not settings.has_valid_gemini_key or not settings.has_valid_tavily_key:
        print("-" * 70)
        print("[NOTICE] Running in Offline/Demo Mode (API keys unconfigured).")
        print("         Full pipeline (Planner -> Search -> Fetch -> Verifier -> Analyst")
        print("         -> Auditor -> Memory -> Trace Log) will execute using built-in")
        print("         corporate knowledge and deterministic engines.")
        print("         To enable live web search & Gemini LLM, add your API keys in .env.")
    print("=" * 70)

    workflow = ResearchWorkflow()
    final_state = await workflow.execute_question(
        question=question,
        question_id=question_id,
        mode=mode,
    )

    if final_state.get("error"):
        print(f"\n[ERROR]: {final_state['error']}")
        return 1

    answer = final_state.get("answer")
    metrics = final_state.get("metrics")
    audits = final_state.get("audit_results", [])

    print("\n" + "=" * 70)
    print("ANALYST SYNTHESIZED ANSWER:")
    print("=" * 70)
    if answer:
        print(answer.answer)
    else:
        print("No answer generated.")

    print("\n" + "=" * 70)
    print("INDEPENDENT AUDITOR RESULTS:")
    print("=" * 70)
    if audits:
        for idx, ar in enumerate(audits, 1):
            print(f"\n[{idx}] Claim: \"{ar.claim}\"")
            print(f"    Status   : {ar.status.value}")
            print(f"    Citation : {ar.source_url or 'None'}")
            print(f"    Reason   : {ar.reason}")
    else:
        print("No claims audited.")

    print("\n" + "=" * 70)
    print("RUN METRICS & COST ACCOUNTING:")
    print("=" * 70)
    if metrics:
        print(f"  Wall-clock Latency : {metrics.latency_ms / 1000.0:.2f} s")
        print(f"  Total Tokens       : {metrics.total_tokens} (Input: {metrics.input_tokens}, Output: {metrics.output_tokens})")
        print(f"  Estimated Cost USD : ${metrics.estimated_cost_usd:.6f}")
        print(f"  Estimated Cost INR : Rs. {metrics.estimated_cost_inr:.4f} (Exchange rate: {settings.usd_to_inr_rate} on {settings.usd_to_inr_date})")
        print(f"  Tool Calls Made    : {metrics.search_calls} searches, {metrics.fetch_calls} fetches, {metrics.llm_calls} LLM calls")
        print(f"  Memory Stats       : {metrics.memory_hits} hits, {metrics.memory_misses} misses")
        if metrics.failures:
            print(f"  Recorded Failures  : {metrics.failures}")

    trace_file = f"{settings.logs_dir}/{question_id}.jsonl"
    print(f"\nTrace log saved to : {trace_file}")
    print("=" * 70)
    return 0


def main() -> None:
    """Synchronous entry point parsing arguments."""
    parser = argparse.ArgumentParser(
        description="Analyst + Auditor Research Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="Which three Indian jewellery retailers opened the most new stores in the last two years?",
        help="Natural-language research question.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default="q01",
        help="Identifier for the question (used for logging and runs).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["NORMAL_ANALYST", "ADVERSARIAL_ANALYST"],
        default="NORMAL_ANALYST",
        help="Analyst execution mode.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the interactive web UI and API server on port 8000.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the web server when --serve is enabled.",
    )

    args = parser.parse_args()

    if args.serve:
        from app.server import run
        print(f"\n[INFO] Starting interactive Web UI on http://127.0.0.1:{args.port} ...")
        run(port=args.port)
        sys.exit(0)

    exit_code = asyncio.run(run_cli(args.question, args.question_id, args.mode))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
