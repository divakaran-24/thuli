"""Entity-centric memory repository for persistent facts, evidence, and sources."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set

from app.memory.database import DatabaseManager
from app.models.schemas import Claim, Evidence

logger = logging.getLogger(__name__)


class EntityMemory:
    """Manages persistent entity knowledge in SQLite: entity -> verified facts -> evidence -> sources."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize with DatabaseManager."""
        self.db = db_manager

    @staticmethod
    def _now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _hash_id(prefix: str, value: str) -> str:
        h = hashlib.sha256(value.lower().strip().encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{h}"

    def get_or_create_entity(self, name: str, aliases: Optional[List[str]] = None) -> str:
        """Retrieve existing entity ID or insert a new entity record."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Entity name cannot be empty.")

        new_aliases = set(a.strip() for a in (aliases or []) if a.strip())

        with self.db.get_connection() as conn:
            # Check for existing entity by exact name
            row = conn.execute(
                "SELECT id, aliases FROM entities WHERE LOWER(name) = LOWER(?);", (clean_name,)
            ).fetchone()

            if row:
                entity_id = row["id"]
                existing_aliases = set(json.loads(row["aliases"] or "[]"))
                combined = existing_aliases.union(new_aliases)
                if len(combined) > len(existing_aliases):
                    conn.execute(
                        "UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?;",
                        (json.dumps(sorted(list(combined))), self._now_iso(), entity_id),
                    )
                    conn.commit()
                return entity_id

            # Also check if clean_name matches any existing entity's aliases
            rows = conn.execute("SELECT id, aliases FROM entities;").fetchall()
            for r in rows:
                existing_aliases = set(json.loads(r["aliases"] or "[]"))
                if clean_name.lower() in [a.lower() for a in existing_aliases]:
                    return r["id"]

            # Create new entity
            entity_id = self._hash_id("ent", clean_name)
            now = self._now_iso()
            conn.execute(
                "INSERT INTO entities (id, name, aliases, created_at, updated_at) VALUES (?, ?, ?, ?, ?);",
                (entity_id, clean_name, json.dumps(sorted(list(new_aliases))), now, now),
            )
            conn.commit()
            logger.info(f"Created new memory entity: '{clean_name}' ({entity_id})")
            return entity_id

    def find_entity(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """Find entity record by name or alias."""
        clean_query = name_or_alias.strip().lower()
        with self.db.get_connection() as conn:
            # Direct name match
            row = conn.execute(
                "SELECT id, name, aliases, created_at, updated_at FROM entities WHERE LOWER(name) = ?;",
                (clean_query,),
            ).fetchone()
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "aliases": json.loads(row["aliases"] or "[]"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

            # Search within aliases or fuzzy substring
            rows = conn.execute("SELECT id, name, aliases, created_at, updated_at FROM entities;").fetchall()
            for r in rows:
                aliases = [a.lower() for a in json.loads(r["aliases"] or "[]")]
                if clean_query in aliases or clean_query in r["name"].lower() or r["name"].lower() in clean_query:
                    return {
                        "id": r["id"],
                        "name": r["name"],
                        "aliases": json.loads(r["aliases"] or "[]"),
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }

        return None

    def save_source(
        self, url: str, domain: str, title: str, publisher: str, quality_score: float = 0.5
    ) -> str:
        """Insert or update a known web source."""
        source_id = self._hash_id("src", url)
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, url, domain, title, publisher, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    publisher = excluded.publisher,
                    quality_score = excluded.quality_score;
                """,
                (source_id, url.strip(), domain, title, publisher, quality_score, self._now_iso()),
            )
            conn.commit()
        return source_id

    def save_evidence(self, evidence: Evidence, source_id: Optional[str] = None) -> str:
        """Insert an evidence passage linked to a source."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO evidence (id, source_id, url, title, publisher, published_date, passage, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    evidence.evidence_id,
                    source_id or self._hash_id("src", evidence.url),
                    evidence.url,
                    evidence.title,
                    evidence.publisher,
                    evidence.published_date,
                    evidence.passage,
                    evidence.retrieved_at,
                ),
            )
            conn.commit()
        return evidence.evidence_id

    def save_claim(self, entity_id: str, claim: Claim, status: str = "VERIFIED") -> str:
        """Insert a verified claim and bind it to its supporting evidence."""
        claim_id = claim.claim_id
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO claims (id, entity_id, text, confidence, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    confidence = excluded.confidence;
                """,
                (
                    claim_id,
                    entity_id,
                    claim.text.strip(),
                    claim.confidence or 1.0,
                    status,
                    self._now_iso(),
                ),
            )

            # Link evidence passages if they exist in the database
            for eid in claim.evidence_ids:
                row = conn.execute("SELECT 1 FROM evidence WHERE id = ?;", (eid,)).fetchone()
                if row:
                    conn.execute(
                        "INSERT OR IGNORE INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?);",
                        (claim_id, eid),
                    )

            # Update entity updated_at
            conn.execute(
                "UPDATE entities SET updated_at = ? WHERE id = ?;",
                (self._now_iso(), entity_id),
            )
            conn.commit()
        return claim_id

    def get_entity_knowledge(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete entity-centric knowledge: verified facts, evidence, and known sources."""
        ent = self.find_entity(name_or_alias)
        if not ent:
            return None

        entity_id = ent["id"]
        with self.db.get_connection() as conn:
            # Query all claims associated with this entity
            claims_rows = conn.execute(
                """
                SELECT c.id, c.text, c.confidence, c.status, c.created_at
                FROM claims c
                WHERE c.entity_id = ? AND c.status IN ('VERIFIED', 'SUPPORTED')
                ORDER BY c.created_at DESC;
                """,
                (entity_id,),
            ).fetchall()

            verified_claims = []
            known_sources: Set[str] = set()

            for crow in claims_rows:
                cid = crow["id"]
                # Query evidence linked to this claim
                ev_rows = conn.execute(
                    """
                    SELECT e.id, e.url, e.title, e.publisher, e.published_date, e.passage
                    FROM evidence e
                    JOIN claim_evidence ce ON e.id = ce.evidence_id
                    WHERE ce.claim_id = ?;
                    """,
                    (cid,),
                ).fetchall()

                evidence_items = []
                citations = []
                for erow in ev_rows:
                    citations.append(erow["url"])
                    known_sources.add(erow["url"])
                    evidence_items.append({
                        "evidence_id": erow["id"],
                        "url": erow["url"],
                        "publisher": erow["publisher"],
                        "passage": erow["passage"],
                    })

                verified_claims.append({
                    "claim_id": cid,
                    "text": crow["text"],
                    "confidence": crow["confidence"],
                    "status": crow["status"],
                    "citations": sorted(list(set(citations))),
                    "evidence": evidence_items,
                })

            # Query audit history for this entity's claims
            audit_rows = conn.execute(
                """
                SELECT a.id, a.claim_text, a.status, a.reason, a.created_at
                FROM audit_results a
                JOIN claims c ON a.claim_id = c.id
                WHERE c.entity_id = ?
                ORDER BY a.created_at DESC;
                """,
                (entity_id,),
            ).fetchall()

            audits = [
                {
                    "audit_id": a["id"],
                    "claim": a["claim_text"],
                    "status": a["status"],
                    "reason": a["reason"],
                    "created_at": a["created_at"],
                }
                for a in audit_rows
            ]

        return {
            "entity_id": entity_id,
            "name": ent["name"],
            "aliases": ent["aliases"],
            "verified_claims": verified_claims,
            "known_sources": sorted(list(known_sources)),
            "audit_history": audits,
            "last_researched": ent["updated_at"],
        }
