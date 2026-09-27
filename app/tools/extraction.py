"""Evidence extraction: Deterministic relevance filtering and structured Evidence generation."""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
from typing import List, Optional, Set

from app.models.schemas import Evidence
from app.tools.fetch import FetchedPage

logger = logging.getLogger(__name__)

# Boilerplate phrases commonly found in web scraping that should be discarded
BOILERPLATE_PATTERNS = [
    r"cookie\s*policy",
    r"privacy\s*policy",
    r"terms\s*of\s*service",
    r"all\s*rights\s*reserved",
    r"subscribe\s*to\s*our\s*newsletter",
    r"sign\s*up\s*for\s*free",
    r"accept\s*all\s*cookies",
    r"javascript\s*is\s*disabled",
    r"please\s*enable\s*javascript",
]


class EvidenceExtractor:
    """Extracts concise, factual passages from fetched webpages to minimize LLM token usage."""

    def __init__(self, max_passages_per_page: int = 3, max_passage_chars: int = 600) -> None:
        """Initialize EvidenceExtractor.

        Args:
            max_passages_per_page: Maximum number of relevant passages to extract per page.
            max_passage_chars: Maximum character length for any extracted passage.
        """
        self.max_passages_per_page = max_passages_per_page
        self.max_passage_chars = max_passage_chars

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        """Check if candidate text is navigational/cookie/boilerplate fluff."""
        text_lower = text.lower()
        if len(text.strip()) < 40:
            return True
        for pattern in BOILERPLATE_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple alphanumeric tokenizer for keyword matching."""
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        # Filter common stop words
        stop_words = {
            "the", "and", "for", "with", "this", "that", "from", "have", "were", "what",
            "which", "when", "where", "who", "more", "most", "about", "into", "over",
        }
        return set(w for w in words if w not in stop_words)

    def extract_passages(
        self,
        page: FetchedPage,
        question: str,
        entities: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> List[Evidence]:
        """Extract only the most relevant, evidence-dense passages from a fetched page.

        Args:
            page: FetchedPage containing cleaned webpage text and metadata.
            question: Original research question.
            entities: Target entities to prioritize.
            keywords: Additional search terms or sub-question terms.

        Returns:
            List of strict Evidence objects.
        """
        from app.models.schemas import SourceStatus

        if not page.success or not page.text.strip():
            logger.debug(f"Cannot extract evidence from unsuccessful/empty page: {page.url}")
            return []

        if getattr(page, "source_status", None) in (SourceStatus.FETCH_FAILED, SourceStatus.REJECTED):
            logger.warning(f"Skipping evidence extraction from unavailable/rejected page: {page.url}")
            return []

        # Prepare scoring query terms
        query_terms = self._tokenize(question)
        if entities:
            for entity in entities:
                query_terms.update(self._tokenize(entity))
        if keywords:
            for kw in keywords:
                query_terms.update(self._tokenize(kw))

        # Split page text into candidate paragraphs/blocks
        raw_paragraphs = page.text.split("\n\n")
        candidates = []
        for p in raw_paragraphs:
            cleaned = " ".join(p.split()).strip()
            if len(cleaned) > self.max_passage_chars:
                # Split large paragraphs on sentences
                sentences = re.split(r"(?<=[.!?])\s+", cleaned)
                current_chunk = []
                current_len = 0
                for s in sentences:
                    if current_len + len(s) > self.max_passage_chars and current_chunk:
                        chunk_str = " ".join(current_chunk)
                        if not self._is_boilerplate(chunk_str):
                            candidates.append(chunk_str)
                        current_chunk = [s]
                        current_len = len(s)
                    else:
                        current_chunk.append(s)
                        current_len += len(s)
                if current_chunk:
                    chunk_str = " ".join(current_chunk)
                    if not self._is_boilerplate(chunk_str):
                        candidates.append(chunk_str)
            else:
                if not self._is_boilerplate(cleaned):
                    candidates.append(cleaned)

        if not candidates:
            return []

        # Score candidates deterministically
        scored_candidates = []
        url_hash = hashlib.md5(page.url.encode("utf-8")).hexdigest()[:8]

        for idx, candidate in enumerate(candidates):
            candidate_tokens = self._tokenize(candidate)
            overlap = query_terms.intersection(candidate_tokens)
            score = len(overlap)

            # Boost score if candidate contains numbers/metrics (crucial for research verification)
            has_numbers = bool(re.search(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", candidate))
            if has_numbers:
                score += 1.5

            # Boost if candidate contains key financial/store terms
            has_financial_terms = bool(
                re.search(r"\b(store|stores|showroom|showrooms|fy2\d|fiscal|q[1-4]|addition|opened|net|gross)\b", candidate.lower())
            )
            if has_financial_terms:
                score += 2.0

            # Boost if specific entity name appears directly
            if entities:
                for entity in entities:
                    if entity.lower() in candidate.lower():
                        score += 3.0

            if score > 0:
                scored_candidates.append((score, idx, candidate))

        # Sort descending by relevance score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top passages
        selected = scored_candidates[: self.max_passages_per_page]
        evidences: List[Evidence] = []
        retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for rank, (score, orig_idx, passage_text) in enumerate(selected):
            evidence_id = f"ev_{url_hash}_{rank + 1}"
            p_lower = passage_text.lower()

            # Determine metric definition
            metric = None
            if any(w in p_lower for w in ["net store additions", "net additions", "net new stores", "net new showrooms", "added on net basis"]):
                metric = "net_store_additions"
            elif any(w in p_lower for w in ["gross store openings", "gross openings", "store openings", "showrooms opened", "opened", "launched"]):
                metric = "gross_store_openings"
            elif any(w in p_lower for w in ["total stores", "total showrooms", "store count reached", "network stands at"]):
                metric = "total_store_count"

            # Determine period
            period_type = None
            reported_period = None
            if "q4" in p_lower or "fourth quarter" in p_lower:
                period_type = "Q4"
                reported_period = "Q4 FY24"
            elif "fy24" in p_lower or "fy 2024" in p_lower or "fy2023-24" in p_lower or "fiscal 2024" in p_lower:
                period_type = "FY"
                reported_period = "FY24"
            elif "cy24" in p_lower or "cy 2024" in p_lower or "calendar year 2024" in p_lower:
                period_type = "CY"
                reported_period = "CY2024"

            # Determine business segment
            segment = None
            if any(w in p_lower for w in ["across all its business segments", "across the company", "watches, eyecare and jewellery", "across all businesses"]):
                segment = "Company-Wide"
            elif any(w in p_lower for w in ["jewellery", "jewelry", "tanishq", "mia", "zoya", "caratlane", "kalyan showrooms"]):
                segment = "Jewellery"

            # Determine geography
            geography = "India"
            if any(w in p_lower for w in ["globally", "middle east", "international", "global"]):
                geography = "Global"

            # Determine entity
            matched_entity = None
            if entities:
                for ent in entities:
                    if ent.lower() in p_lower:
                        matched_entity = ent
                        break

            evidence = Evidence(
                evidence_id=evidence_id,
                source_id=page.publisher,
                url=page.url,
                title=page.title,
                publisher=page.publisher,
                published_date=page.published_date,
                passage=passage_text,
                retrieved_at=retrieved_at,
                source_type=getattr(page, "source_type", "LIVE"),
                source_status=getattr(page, "source_status", SourceStatus.EXTRACTED),
                period_type=period_type,
                reported_period=reported_period,
                metric=metric,
                business_segment=segment,
                geography=geography,
                entity=matched_entity,
            )
            evidences.append(evidence)

        logger.info(f"Extracted {len(evidences)} evidence passages from {page.url}")
        return evidences
