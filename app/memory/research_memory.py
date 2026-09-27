"""ResearchMemoryManager: Orchestrates entity knowledge lookup, persistence, and audit policies."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from app.memory.database import DatabaseManager
from app.memory.entity_memory import EntityMemory
from app.models.schemas import AuditResult, AuditStatus, Claim, Evidence, RunMetrics

logger = logging.getLogger(__name__)


class ResearchMemoryManager:
    """High-level manager for entity memory lookup, run recording, and learned policy retrieval."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        """Initialize ResearchMemoryManager."""
        self.db = db_manager or DatabaseManager()
        self.entity_memory = EntityMemory(self.db)

    def lookup_entities(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], int, int]:
        """Lookup previously researched knowledge for a list of entities.

        Returns:
            Tuple of (list_of_known_entity_data, memory_hits, memory_misses).
        """
        known_entities: List[Dict[str, Any]] = []
        hits = 0
        misses = 0

        for entity_name in entities:
            clean_name = entity_name.strip()
            if not clean_name:
                continue

            knowledge = self.entity_memory.get_entity_knowledge(clean_name)
            if knowledge and knowledge.get("verified_claims"):
                hits += 1
                known_entities.append(knowledge)
                logger.info(f"Memory HIT: Found {len(knowledge['verified_claims'])} verified facts for '{clean_name}'.")
            else:
                misses += 1
                logger.info(f"Memory MISS: No verified historical facts for '{clean_name}'.")

        return known_entities, hits, misses

    def save_run_knowledge(
        self,
        question_id: str,
        question: str,
        mode: str,
        metrics: RunMetrics,
        claims: List[Claim],
        evidences: List[Evidence],
        audit_results: List[AuditResult],
        primary_entities: Optional[List[str]] = None,
    ) -> None:
        """Persist sources, evidence, claims, audits, and run metrics from a completed question run."""
        # 1. Save all sources and evidence
        for ev in evidences:
            self.entity_memory.save_source(
                url=ev.url,
                domain=ev.publisher,
                title=ev.title,
                publisher=ev.publisher,
                quality_score=0.7,
            )
            self.entity_memory.save_evidence(ev)

        # 2. Map claims to audit status
        audit_status_map = {ar.claim_id: ar.status for ar in audit_results}

        # Resolve or create primary entities
        entity_ids = []
        if primary_entities:
            for ent_name in primary_entities:
                if ent_name.strip():
                    eid = self.entity_memory.get_or_create_entity(ent_name.strip())
                    entity_ids.append(eid)

        default_entity_id = entity_ids[0] if entity_ids else self.entity_memory.get_or_create_entity("General Market")

        # 3. Save claims linked to entities and audited status
        for claim in claims:
            status = audit_status_map.get(claim.claim_id, AuditStatus.UNSUPPORTED)
            db_status = "SUPPORTED" if status == AuditStatus.SUPPORTED else status.value
            target_entity_id = default_entity_id

            # If claim text mentions a specific entity, bind to it
            if primary_entities:
                for idx, ent_name in enumerate(primary_entities):
                    if ent_name.lower() in claim.text.lower() and idx < len(entity_ids):
                        target_entity_id = entity_ids[idx]
                        break

            self.entity_memory.save_claim(
                entity_id=target_entity_id,
                claim=claim,
                status=db_status,
            )

        # 4. Save audit results
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            for ar in audit_results:
                audit_id = f"aud_{question_id}_{ar.claim_id}"
                conn.execute(
                    """
                    INSERT INTO audit_results (
                        id, run_id, claim_id, claim_text, status, citation_present,
                        source_url, evidence_passage, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status = excluded.status,
                        reason = excluded.reason;
                    """,
                    (
                        audit_id,
                        question_id,
                        ar.claim_id,
                        ar.claim,
                        ar.status.value,
                        1 if ar.citation_present else 0,
                        ar.source_url,
                        ar.evidence,
                        ar.reason,
                        now,
                    ),
                )

            # 5. Save run metrics
            conn.execute(
                """
                INSERT INTO runs (
                    id, question_id, question, mode, latency_ms, input_tokens, output_tokens,
                    total_tokens, estimated_cost_usd, estimated_cost_inr, llm_calls,
                    search_calls, fetch_calls, memory_hits, memory_misses, failures, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    latency_ms = excluded.latency_ms,
                    total_tokens = excluded.total_tokens,
                    estimated_cost_usd = excluded.estimated_cost_usd;
                """,
                (
                    f"run_{question_id}",
                    question_id,
                    question,
                    mode,
                    metrics.latency_ms,
                    metrics.input_tokens,
                    metrics.output_tokens,
                    metrics.total_tokens,
                    metrics.estimated_cost_usd,
                    metrics.estimated_cost_inr,
                    metrics.llm_calls,
                    metrics.search_calls,
                    metrics.fetch_calls,
                    metrics.memory_hits,
                    metrics.memory_misses,
                    json.dumps(metrics.failures),
                    "COMPLETED",
                    now,
                ),
            )
            conn.commit()

        logger.info(f"Persisted research run knowledge for {question_id} ({len(claims)} claims, {len(audit_results)} audits).")

    def save_feedback_policy(self, rule: str, source_audit_id: str, trigger_reason: str) -> str:
        """Store a compact operational lesson in the feedback table."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        import hashlib
        policy_id = f"pol_{hashlib.md5(rule.encode('utf-8')).hexdigest()[:8]}"
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, policy_rule, source_audit_id, trigger_reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING;
                """,
                (policy_id, rule.strip(), source_audit_id, trigger_reason, now),
            )
            conn.commit()
        logger.info(f"Recorded feedback policy: '{rule}' ({policy_id})")
        return policy_id

    def get_feedback_policies(self, limit: int = 5) -> List[str]:
        """Retrieve recent compact feedback policy rules for injection into Planner."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT policy_rule FROM feedback ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
            return [r["policy_rule"] for r in rows]
