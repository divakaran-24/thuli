"""Conflict resolver: Identifies and explains disagreements between credible sources."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.models.schemas import ConflictRecord, Evidence

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Detects and resolves discrepancies between sources regarding numbers, dates, or scope."""

    @staticmethod
    def inspect_discrepancy(
        text_a: str,
        text_b: str,
        date_a: Optional[str] = None,
        date_b: Optional[str] = None,
    ) -> Tuple[str, str, bool]:
        """Analyze differences between two competing factual statements.

        Returns:
            Tuple of (discrepancy_type, explanation, is_resolved).
        """
        lower_a = text_a.lower()
        lower_b = text_b.lower()

        # 1. Geographic Scope (India vs Global/Middle East)
        is_global_a = "global" in lower_a or "middle east" in lower_a or "international" in lower_a or "worldwide" in lower_a
        is_global_b = "global" in lower_b or "middle east" in lower_b or "international" in lower_b or "worldwide" in lower_b
        is_india_a = "india only" in lower_a or "domestic" in lower_a or "in india" in lower_a
        is_india_b = "india only" in lower_b or "domestic" in lower_b or "in india" in lower_b

        if (is_global_a and is_india_b) or (is_global_b and is_india_a):
            global_src = "Source A" if is_global_a else "Source B"
            india_src = "Source B" if is_global_a else "Source A"
            return (
                "geographic_scope",
                f"Resolved: {global_src} measures total international footprint, while {india_src} reflects domestic Indian locations.",
                True,
            )

        # 2. Financial Year (FY) vs Calendar Year (CY)
        is_fy_a = bool(re.search(r"\bfy\s*(?:20)?\d{2}\b", lower_a))
        is_fy_b = bool(re.search(r"\bfy\s*(?:20)?\d{2}\b", lower_b))
        is_cy_a = bool(re.search(r"\bcalendar\s*year\s*(?:20)?\d{2}\b|\bcy\s*(?:20)?\d{2}\b", lower_a))
        is_cy_b = bool(re.search(r"\bcalendar\s*year\s*(?:20)?\d{2}\b|\bcy\s*(?:20)?\d{2}\b", lower_b))

        if (is_fy_a and is_cy_b) or (is_fy_b and is_cy_a):
            return (
                "FY_vs_CY",
                "Resolved: Sources use different accounting periods (one reports Indian Fiscal Year April-March, the other reports Calendar Year Jan-Dec).",
                True,
            )

        # 3. Net additions vs Gross openings
        is_net_a = "net" in lower_a
        is_net_b = "net" in lower_b
        is_gross_a = "gross" in lower_a or "opened" in lower_a and not is_net_a
        is_gross_b = "gross" in lower_b or "opened" in lower_b and not is_net_b

        if (is_net_a and not is_net_b) or (is_net_b and not is_net_a):
            net_src = "Source A" if is_net_a else "Source B"
            gross_src = "Source B" if is_net_a else "Source A"
            return (
                "definition_net_vs_gross",
                f"Resolved: {net_src} accounts for net additions after store closures, while {gross_src} measures total gross openings.",
                True,
            )

        # 4. Reporting Date / Recency
        if date_a and date_b and date_a != date_b:
            if date_a > date_b:
                return (
                    "reporting_date",
                    f"Resolved: Source A ({date_a}) provides more recent/updated numbers superseding Source B ({date_b}).",
                    True,
                )
            elif date_b > date_a:
                return (
                    "reporting_date",
                    f"Resolved: Source B ({date_b}) provides more recent/updated numbers superseding Source A ({date_a}).",
                    True,
                )

        # 5. Announced / Target vs Actually Opened
        is_target_a = "plans to" in lower_a or "targets" in lower_a or "aims to" in lower_a or "guidance" in lower_a
        is_actual_a = "opened" in lower_a or "completed" in lower_a or "added" in lower_a
        is_target_b = "plans to" in lower_b or "targets" in lower_b or "aims to" in lower_b or "guidance" in lower_b
        is_actual_b = "opened" in lower_b or "completed" in lower_b or "added" in lower_b

        if (is_target_a and is_actual_b) or (is_target_b and is_actual_a):
            target_src = "Source A" if is_target_a else "Source B"
            actual_src = "Source B" if is_target_a else "Source A"
            return (
                "announced_vs_actual",
                f"Resolved: {target_src} states forward-looking management guidance/targets, while {actual_src} confirms actual completed openings.",
                True,
            )

        # Unresolved discrepancy
        return (
            "unresolved",
            "Discrepancy could not be definitively resolved from available evidence without speculation.",
            False,
        )

    def create_conflict_record(
        self,
        conflict_id: str,
        entity: str,
        attribute: str,
        claim_a: str,
        claim_b: str,
        evidence_a: Evidence,
        evidence_b: Evidence,
    ) -> ConflictRecord:
        """Create and analyze a ConflictRecord between two competing pieces of evidence."""
        discrepancy_type, explanation, is_resolved = self.inspect_discrepancy(
            text_a=evidence_a.passage,
            text_b=evidence_b.passage,
            date_a=evidence_a.published_date,
            date_b=evidence_b.published_date,
        )

        record = ConflictRecord(
            conflict_id=conflict_id,
            entity=entity,
            attribute=attribute,
            source_a=evidence_a.url,
            source_b=evidence_b.url,
            claim_a=claim_a,
            claim_b=claim_b,
            discrepancy_type=discrepancy_type,
            resolution=explanation,
            resolved=is_resolved,
        )
        logger.info(f"Conflict Record created: {conflict_id} ({discrepancy_type}) -> resolved={is_resolved}")
        return record
