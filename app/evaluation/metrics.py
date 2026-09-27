"""Evaluation metrics aggregator and report generator producing CSV and JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.config import settings
from app.models.schemas import AuditResult, AuditStatus, RunMetrics


def compile_evaluation_results(
    run_records: List[Dict[str, Any]],
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile run records into results/ tables and compute summary statistics.

    Produces:
    - results/answers.json
    - results/metrics.csv
    - results/audit_results.csv
    - results/cost_trend.csv
    - results/summary.json
    """
    out_dir = Path(results_dir or settings.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    cost_trend_rows: List[Dict[str, Any]] = []
    answers_dict: Dict[str, Any] = {}

    total_supported = 0
    total_unsupported = 0
    total_contradicted = 0
    total_missing_citations = 0

    for idx, record in enumerate(run_records, 1):
        qid = record["question_id"]
        q_text = record["question"]
        mode = record.get("mode", "NORMAL_ANALYST")
        metrics: Optional[RunMetrics] = record.get("metrics")
        if not metrics:
            metrics = RunMetrics(
                latency_ms=120000.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                estimated_cost_inr=0.0,
                llm_calls=0,
                search_calls=0,
                fetch_calls=0,
                memory_hits=0,
                memory_misses=0,
                failures=["Timeout or execution failure"],
            )
        answer = record.get("answer")
        audits: List[AuditResult] = record.get("audit_results", [])

        # 1. Metrics row
        metrics_rows.append({
            "question_id": qid,
            "mode": mode,
            "latency_ms": metrics.latency_ms,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "estimated_cost_usd": metrics.estimated_cost_usd,
            "estimated_cost_inr": metrics.estimated_cost_inr,
            "llm_calls": metrics.llm_calls,
            "search_calls": metrics.search_calls,
            "fetch_calls": metrics.fetch_calls,
            "memory_hits": metrics.memory_hits,
            "memory_misses": metrics.memory_misses,
            "failures": "; ".join(metrics.failures) if metrics.failures else "None",
        })

        # 2. Cost trend row
        cost_trend_rows.append({
            "question_index": idx,
            "question_id": qid,
            "mode": mode,
            "total_tokens": metrics.total_tokens,
            "estimated_cost_usd": metrics.estimated_cost_usd,
            "estimated_cost_inr": metrics.estimated_cost_inr,
            "memory_hits": metrics.memory_hits,
            "memory_misses": metrics.memory_misses,
            "latency_seconds": round(metrics.latency_ms / 1000.0, 2),
        })

        # 3. Audit results rows
        for ar in audits:
            if ar.status == AuditStatus.SUPPORTED:
                total_supported += 1
            elif ar.status == AuditStatus.UNSUPPORTED:
                total_unsupported += 1
            elif ar.status == AuditStatus.CONTRADICTED:
                total_contradicted += 1
            elif ar.status == AuditStatus.MISSING_CITATION:
                total_missing_citations += 1

            audit_rows.append({
                "question_id": qid,
                "mode": mode,
                "claim_id": ar.claim_id,
                "claim": ar.claim,
                "status": ar.status.value,
                "citation_present": ar.citation_present,
                "source_url": ar.source_url or "None",
                "reason": ar.reason,
            })

        # 4. Answers dict
        answers_dict[qid] = {
            "question": q_text,
            "mode": mode,
            "answer": answer.answer if answer else "No answer generated.",
            "claims": [c.model_dump() for c in (answer.claims if answer else [])],
            "unanswered_aspects": answer.unanswered_aspects if answer else [],
        }

    # DataFrames export
    df_metrics = pd.DataFrame(metrics_rows)
    df_audits = pd.DataFrame(audit_rows)
    df_cost_trend = pd.DataFrame(cost_trend_rows)

    df_metrics.to_csv(out_dir / "metrics.csv", index=False)
    df_audits.to_csv(out_dir / "audit_results.csv", index=False)
    df_cost_trend.to_csv(out_dir / "cost_trend.csv", index=False)

    with open(out_dir / "answers.json", "w", encoding="utf-8") as f:
        json.dump(answers_dict, f, indent=2)

    # Compute high-level summary
    num_questions = max(1, len(run_records))
    total_tokens = sum(m["total_tokens"] for m in metrics_rows)
    total_cost_usd = sum(m["estimated_cost_usd"] for m in metrics_rows)
    total_cost_inr = sum(m["estimated_cost_inr"] for m in metrics_rows)
    avg_latency_ms = sum(m["latency_ms"] for m in metrics_rows) / num_questions
    max_latency_ms = max((m["latency_ms"] for m in metrics_rows), default=0.0)

    summary = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "total_questions_evaluated": len(run_records),
        "latency": {
            "average_latency_seconds": round(avg_latency_ms / 1000.0, 2),
            "max_latency_seconds": round(max_latency_ms / 1000.0, 2),
        },
        "tokens": {
            "total_tokens": total_tokens,
            "average_tokens_per_question": round(total_tokens / num_questions, 1),
        },
        "cost": {
            "total_cost_usd": round(total_cost_usd, 6),
            "average_cost_usd_per_question": round(total_cost_usd / num_questions, 6),
            "total_cost_inr": round(total_cost_inr, 4),
            "average_cost_inr_per_question": round(total_cost_inr / num_questions, 4),
            "exchange_rate_usd_to_inr": settings.usd_to_inr_rate,
            "exchange_rate_date": settings.usd_to_inr_date,
        },
        "tool_usage": {
            "average_search_calls_per_question": round(sum(m["search_calls"] for m in metrics_rows) / num_questions, 2),
            "average_fetch_calls_per_question": round(sum(m["fetch_calls"] for m in metrics_rows) / num_questions, 2),
            "average_llm_calls_per_question": round(sum(m["llm_calls"] for m in metrics_rows) / num_questions, 2),
        },
        "memory_transfer": {
            "total_memory_hits": sum(m["memory_hits"] for m in metrics_rows),
            "total_memory_misses": sum(m["memory_misses"] for m in metrics_rows),
        },
        "auditor_verification": {
            "total_claims_audited": len(audit_rows),
            "supported_claims": total_supported,
            "unsupported_claims": total_unsupported,
            "contradicted_claims": total_contradicted,
            "missing_citations": total_missing_citations,
            "supported_rate_percent": round((total_supported / max(1, len(audit_rows))) * 100.0, 2),
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
