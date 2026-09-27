"""Deterministic and semantic claim verification to ensure strict factual fidelity."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.models.schemas import Claim, Evidence

logger = logging.getLogger(__name__)


def extract_numbers_and_years(text: str) -> List[str]:
    """Extract all numbers, percentages, currency, and year figures from text."""
    # Matches integers, decimals, comma-separated numbers, including attached prefixes like FY24
    raw_matches = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?%?", text)
    cleaned = []
    for m in raw_matches:
        norm = m.replace(",", "").rstrip("%").strip()
        if norm:
            cleaned.append(norm)
    return cleaned


class ClaimVerifier:
    """Verifies factual claims against cited evidence using deterministic checks and semantic validation."""

    def __init__(
        self,
        semantic_verifier_fn: Optional[Callable[[str, str], Tuple[bool, str]]] = None,
    ) -> None:
        """Initialize ClaimVerifier.

        Args:
            semantic_verifier_fn: Optional callable for LLM semantic verification.
                                  Signature: (claim_text, evidence_passage) -> (is_supported, reason)
        """
        self.semantic_verifier_fn = semantic_verifier_fn

    @staticmethod
    def verify_citations_present(claim: Claim) -> Tuple[bool, str]:
        """Check if the claim contains at least one valid citation URL."""
        if not claim.citations:
            return False, "Missing citation: No source citations attached to claim."

        valid_urls = [
            c for c in claim.citations
            if isinstance(c, str) and (c.startswith("http://") or c.startswith("https://"))
        ]
        if not valid_urls:
            return False, f"Invalid citation: Citations {claim.citations} are not valid URLs."

        return True, "Citations present and valid."

    @staticmethod
    def verify_urls_were_fetched(claim: Claim, fetched_urls: Set[str]) -> Tuple[bool, str]:
        """Verify that cited URLs were actually fetched by the system."""
        for url in claim.citations:
            # Normalize trailing slash
            norm_url = url.rstrip("/")
            matching = [u for u in fetched_urls if u.rstrip("/") == norm_url]
            if not matching:
                return False, f"Unverified source: Cited URL '{url}' was never fetched during research."
        return True, "All cited URLs were verified as fetched."

    @staticmethod
    def verify_numerical_values(
        claim: Claim, evidence_map: Dict[str, Evidence]
    ) -> Tuple[bool, str]:
        """Verify that every number/statistic asserted in the claim exists in cited evidence."""
        claim_numbers = extract_numbers_and_years(claim.text)
        if not claim_numbers:
            return True, "No numerical values to cross-check."

        # Aggregate all text from cited evidence objects
        cited_evidence_texts = []
        for eid in claim.evidence_ids:
            if eid in evidence_map:
                cited_evidence_texts.append(evidence_map[eid].passage)

        combined_evidence = " ".join(cited_evidence_texts)
        if not combined_evidence.strip():
            return False, f"Missing evidence: No evidence passages found for IDs {claim.evidence_ids}."

        evidence_numbers = set(extract_numbers_and_years(combined_evidence))

        missing_numbers = []
        for num in claim_numbers:
            # Check if num or stripped num exists in evidence
            if num not in evidence_numbers:
                # Also check as substring
                if num not in combined_evidence and num.replace("%", "") not in evidence_numbers:
                    missing_numbers.append(num)

        if missing_numbers:
            return (
                False,
                f"Numerical discrepancy: Value(s) {missing_numbers} from claim text not found in cited evidence.",
            )

        return True, "All numerical values confirmed in cited evidence."

    @staticmethod
    def check_generic_index_citation(claim: Claim) -> Tuple[bool, str]:
        """Reject claims that cite generic index or landing pages without specific disclosure."""
        from app.tools.fetch import is_generic_index_url

        for cite in claim.citations:
            if is_generic_index_url(cite):
                return (
                    False,
                    f"Generic index citation: URL '{cite}' is a generic index/portal page and cannot support specific factual claims.",
                )
        return True, "No generic index citations detected."

    @staticmethod
    def check_source_status(
        claim: Claim, evidence_map: Dict[str, Evidence]
    ) -> Tuple[bool, str]:
        """Ensure supporting evidence does not originate from failed or rejected sources."""
        from app.models.schemas import SourceStatus

        for eid in claim.evidence_ids:
            if eid in evidence_map:
                ev = evidence_map[eid]
                if getattr(ev, "source_status", None) in (SourceStatus.FETCH_FAILED, SourceStatus.REJECTED):
                    return (
                        False,
                        f"Source unavailable: Evidence '{eid}' originates from a failed or rejected source ({ev.url}).",
                    )
        return True, "Supporting sources are valid and fetched."

    @staticmethod
    def check_scope_matches(
        claim: Claim, evidence_map: Dict[str, Evidence]
    ) -> Tuple[bool, str]:
        """Verify business segment, timeframe/period, metric, and geography alignment between claim and evidence."""
        for eid in claim.evidence_ids:
            if eid not in evidence_map:
                continue
            ev = evidence_map[eid]
            ev_p = ev.passage.lower()
            claim_t = claim.text.lower()
            claim_nums = extract_numbers_and_years(claim.text)

            # 1. Business Segment Scope (Titan-type error: Company-Wide vs Jewellery segment)
            claim_wants_jewellery = bool(
                claim.business_segment == "Jewellery"
                or re.search(r"\b(jewellery|jewelry|tanishq|mia|zoya|caratlane)\b", claim_t)
            )
            source_is_company_wide = bool(
                ev.business_segment == "Company-Wide"
                or any(phrase in ev_p for phrase in [
                    "across all its business segments",
                    "across the company",
                    "across all businesses",
                    "watches, eyecare and jewellery",
                    "watches, eyewear and jewellery",
                ])
            )
            if claim_wants_jewellery and source_is_company_wide:
                # If claim asserts a number representing company-wide store additions
                cw_match = re.search(
                    r"(\d+)\s+net\s+(?:new\s+)?stores.*?(?:across\s+all|across\s+the\s+company|across\s+its\s+business|all\s+businesses)",
                    ev_p,
                )
                if not cw_match:
                    cw_match = re.search(
                        r"(?:across\s+all|across\s+the\s+company|across\s+its\s+business|all\s+businesses).*?(\d+)\s+net\s+(?:new\s+)?stores",
                        ev_p,
                    )
                if cw_match and cw_match.group(1) in claim_nums:
                    return (
                        False,
                        f"Scope mismatch: Figure {cw_match.group(1)} represents company-wide additions across all businesses, not jewellery-segment stores.",
                    )
                if ev.business_segment == "Company-Wide" and not re.search(r"jewell?ery\s+(?:added|opened)\s+(\d+)", ev_p):
                    return (
                        False,
                        "Scope mismatch: Source reports company-wide store additions (across all businesses), not jewellery segment.",
                    )

            # 2. Reporting Period Scope (Q4 FY24 vs full FY24)
            claim_wants_full_fy = bool(
                claim.period == "FY24"
                or (("fy24" in claim_t or "fy 2024" in claim_t or "full year" in claim_t) and "q4" not in claim_t)
            )
            source_is_q4 = bool(
                ev.period_type == "Q4"
                or any(phrase in ev_p for phrase in ["in q4", "during q4", "fourth quarter", "q4 fy24"])
            )
            if claim_wants_full_fy and source_is_q4:
                # If claim uses the Q4 number as the full FY figure
                q4_nums = extract_numbers_and_years(ev.passage)
                overlapping = [n for n in claim_nums if n in q4_nums and n not in ["24", "2024"]]
                if overlapping:
                    return (
                        False,
                        f"Period mismatch: Figure(s) {overlapping} represent Q4 FY24 store additions, which cannot be substituted for full FY24 figures.",
                    )

            # 3. Metric Scope (Net store additions vs Gross store openings)
            claim_asserts_openings = bool(
                claim.metric == "gross_store_openings"
                or re.search(r"\b(opened\s+\d+|store\s+openings?|opened\s+the\s+most|gross\s+(?:store\s+)?openings?)\b", claim_t)
            )
            source_reports_net = bool(
                ev.metric == "net_store_additions"
                or any(phrase in ev_p for phrase in ["net store", "net additions", "net new", "net showroom", "added on net basis"])
            )
            if claim_asserts_openings and source_reports_net and not any(phrase in ev_p for phrase in ["gross openings", "opened gross", "gross additions"]):
                return (
                    False,
                    "Metric mismatch: Source reports net store additions, not gross store openings.",
                )

            # Reverse check: Claim asserts net additions, source reports gross openings
            claim_asserts_net = bool(claim.metric == "net_store_additions" or "net" in claim_t)
            source_reports_gross = bool(
                ev.metric == "gross_store_openings"
                or any(phrase in ev_p for phrase in ["gross store openings", "gross openings", "store openings"])
            )
            if claim_asserts_net and source_reports_gross and "net" not in ev_p:
                return (
                    False,
                    "Metric mismatch: Source reports gross store openings without net adjustment.",
                )

            # 4. Geography Scope (India vs Global / Middle East)
            claim_asserts_india = bool(claim.geography == "India" or "in india" in claim_t)
            source_is_global = bool(
                ev.geography == "Global"
                or any(phrase in ev_p for phrase in ["globally", "middle east", "in india and the middle east", "international"])
            )
            if claim_asserts_india and source_is_global:
                m_global = re.search(r"(\d+)\s+net\s+new\s+showrooms\s+globally", ev_p)
                if m_global and m_global.group(1) in claim_nums:
                    return (
                        False,
                        f"Geography mismatch: Figure {m_global.group(1)} is a global showroom total (including Middle East), not India-only.",
                    )

        return True, "Scope, period, and metric definitions match."

    def gate_claim(
        self,
        claim: Claim,
        evidence_map: Dict[str, Evidence],
        fetched_urls: Set[str],
    ) -> Tuple[bool, str]:
        """Strict Python Claim Gate: Enforce evidence existence, fetch success, citation validity, and scope matching."""
        # 1. Check citations presence
        cite_ok, cite_reason = self.verify_citations_present(claim)
        if not cite_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = cite_reason
            return False, cite_reason

        # 2. Check for generic index pages
        gen_ok, gen_reason = self.check_generic_index_citation(claim)
        if not gen_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = gen_reason
            return False, gen_reason

        # 3. Check that cited URLs were actually fetched
        url_ok, url_reason = self.verify_urls_were_fetched(claim, fetched_urls)
        if not url_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = url_reason
            return False, url_reason

        # 4. Check evidence ID linkage
        if not claim.evidence_ids:
            for cite in claim.citations:
                norm_c = cite.rstrip("/").lower()
                for ev_id, ev in evidence_map.items():
                    if ev.url.rstrip("/").lower() == norm_c:
                        claim.evidence_ids.append(ev_id)

        resolved_eids = []
        for eid in claim.evidence_ids:
            if eid in evidence_map:
                resolved_eids.append(eid)
            else:
                for cite in claim.citations:
                    norm_c = cite.rstrip("/").lower()
                    for ev_id, ev in evidence_map.items():
                        if ev.url.rstrip("/").lower() == norm_c and ev_id not in resolved_eids:
                            resolved_eids.append(ev_id)
                            break

        if resolved_eids:
            claim.evidence_ids = resolved_eids
        else:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = f"Unresolvable evidence ID: '{claim.evidence_ids}' not found in active evidence pool."
            return False, claim.rejection_reason

        # 5. Check source status (reject failed or unavailable sources)
        src_ok, src_reason = self.check_source_status(claim, evidence_map)
        if not src_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = src_reason
            return False, src_reason

        # 6. Check scope, period, and metric alignment
        scope_ok, scope_reason = self.check_scope_matches(claim, evidence_map)
        if not scope_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = scope_reason
            return False, scope_reason

        # 7. Check numerical values
        num_ok, num_reason = self.verify_numerical_values(claim, evidence_map)
        if not num_ok:
            claim.is_verified = False
            claim.verification_status = "REJECTED"
            claim.rejection_reason = num_reason
            return False, num_reason

        # All deterministic checks passed
        claim.is_verified = True
        claim.verification_status = "VERIFIED"
        claim.rejection_reason = None
        return True, "Deterministic verification passed."

    def verify_claim_deterministically(
        self,
        claim: Claim,
        evidence_map: Dict[str, Evidence],
        fetched_urls: Set[str],
    ) -> Tuple[bool, str]:
        """Perform deterministic Python verification on a claim without LLM calls.

        Returns:
            Tuple of (is_valid, reason).
        """
        return self.gate_claim(claim, evidence_map, fetched_urls)

    def verify(
        self,
        claim: Claim,
        evidence_map: Dict[str, Evidence],
        fetched_urls: Set[str],
    ) -> Tuple[bool, str]:
        """Verify claim completely, running deterministic checks first then optional semantic check."""
        det_ok, det_reason = self.verify_claim_deterministically(
            claim=claim,
            evidence_map=evidence_map,
            fetched_urls=fetched_urls,
        )
        if not det_ok:
            return False, det_reason

        # If semantic verification function is supplied, invoke it
        if self.semantic_verifier_fn:
            passages = [evidence_map[eid].passage for eid in claim.evidence_ids if eid in evidence_map]
            combined_passage = " ".join(passages)
            sem_ok, sem_reason = self.semantic_verifier_fn(claim.text, combined_passage)
            if not sem_ok:
                return False, f"Semantic verification failed: {sem_reason}"

        return True, "Claim verified successfully."
