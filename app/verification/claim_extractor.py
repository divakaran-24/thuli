"""Claim extractor: Extracts and decomposes factual statements from evidence and text."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from app.models.schemas import Claim, Evidence

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extracts explicit, atomic factual claims from evidence or synthesized text."""

    @staticmethod
    def extract_claims_from_evidence(
        evidences: List[Evidence],
        entity: Optional[str] = None,
    ) -> List[Claim]:
        """Convert evidence passages into discrete candidate factual claims.

        Args:
            evidences: List of Evidence objects.
            entity: Optional entity name to tag claims with.

        Returns:
            List of Claim objects referencing the source evidence IDs and citations.
        """
        claims: List[Claim] = []
        claim_counter = 1

        for ev in evidences:
            # Split passage into sentences
            sentences = re.split(r"(?<=[.!?])\s+", ev.passage.strip())
            for sent in sentences:
                sent_clean = sent.strip()
                # Keep sentences that contain factual numbers, dates, or significant length
                has_number = bool(re.search(r"\b\d+\b", sent_clean))
                if len(sent_clean) >= 25 and (has_number or len(sent_clean) >= 40):
                    claim = Claim(
                        claim_id=f"c_{claim_counter}",
                        text=sent_clean,
                        evidence_ids=[ev.evidence_id],
                        citations=[ev.url],
                        entity=ev.entity or entity or ev.publisher,
                        metric=ev.metric,
                        business_segment=ev.business_segment,
                        geography=ev.geography,
                        period=ev.reported_period,
                        source_type=ev.source_type,
                        confidence=1.0,
                    )
                    claims.append(claim)
                    claim_counter += 1

        logger.info(f"Extracted {len(claims)} candidate claims from {len(evidences)} evidence passages.")
        return claims

    @staticmethod
    def extract_claims_from_text(
        text: str,
        evidence_pool: Optional[List[Evidence]] = None,
    ) -> List[Claim]:
        """Extract atomic factual claims and inline citations from synthesized text.

        Looks for inline URL references or bracketed citations (e.g., [https://...] or [Source: URL]).
        """
        claims: List[Claim] = []
        evidence_url_to_ids: Dict[str, List[str]] = {}
        if evidence_pool:
            for ev in evidence_pool:
                norm_url = ev.url.rstrip("/")
                if norm_url not in evidence_url_to_ids:
                    evidence_url_to_ids[norm_url] = []
                evidence_url_to_ids[norm_url].append(ev.evidence_id)

        # Split text into lines/paragraphs or sentences
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        claim_counter = 1

        for line in lines:
            # Skip high-level section headers
            if line.startswith("#") or line.lower().startswith("sources:") or line.lower().startswith("references:"):
                continue

            # Find all URLs in line
            url_matches = re.findall(r"https?://[^\s)\]]+", line)
            cleaned_urls = [u.rstrip(".,;:)") for u in url_matches]

            # Split line into sentences if line has multiple sentences
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue

                sent_urls = [u for u in cleaned_urls if u in sent_clean]
                # If sentence didn't have URL directly, associate line's URLs
                assoc_urls = sent_urls if sent_urls else cleaned_urls

                # Find associated evidence IDs
                assoc_eids = []
                for u in assoc_urls:
                    norm = u.rstrip("/")
                    if norm in evidence_url_to_ids:
                        assoc_eids.extend(evidence_url_to_ids[norm])

                claim = Claim(
                    claim_id=f"ac_{claim_counter}",
                    text=sent_clean,
                    evidence_ids=list(set(assoc_eids)),
                    citations=assoc_urls,
                    confidence=1.0 if assoc_urls else 0.5,
                )
                claims.append(claim)
                claim_counter += 1

        return claims
