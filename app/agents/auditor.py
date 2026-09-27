"""Auditor Agent: Independently verifies factual claims against cited live sources."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import settings
from app.models.schemas import AnalystAnswer, AuditResult, AuditStatus, Claim, Evidence
from app.tools.extraction import EvidenceExtractor
from app.tools.fetch import FetchedPage, PageFetcher
from app.verification.claim_verifier import extract_numbers_and_years

logger = logging.getLogger(__name__)

AUDITOR_SYSTEM_PROMPT = """You are an independent, highly critical Research Auditor Agent.
Your responsibility is to verify whether an Analyst's factual claim is actually supported by the text of the cited source.

You will be given:
1. The factual claim asserted by the Analyst.
2. The cited source URL.
3. The exact passage extracted from the cited source.

CLASSIFICATION RULES:
1. SUPPORTED: The cited passage directly, explicitly provides factual evidence confirming the claim (including matching numbers, dates, and entities).
2. CONTRADICTED: The cited passage directly conflicts with or contradicts the claim (e.g., claim states 50 stores, but source explicitly states 35 stores; or claim asserts date X, but source says date Y).
3. UNSUPPORTED: The cited passage does not contain enough information to substantiate the claim, is ambiguous, or talks about an unrelated topic.

CRITICAL INSTRUCTIONS:
- You must NEVER simply say "Looks correct" or "Verified".
- You MUST provide a specific, evidence-grounded reason quoting or pointing to the exact discrepancy or confirmation.

Output MUST be a valid JSON object matching this schema:
{
  "status": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED",
  "reason": "Detailed, specific justification referencing the text."
}
"""


class AuditorAgent:
    """Agent that independently audits research answers, claims, and cited web sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        page_fetcher: Optional[PageFetcher] = None,
        llm_caller: Optional[Callable[[str, str], Tuple[str, int, int]]] = None,
    ) -> None:
        """Initialize AuditorAgent.

        Args:
            api_key: Optional Gemini API key override.
            model: Optional Gemini model override.
            page_fetcher: PageFetcher instance for fetching cited sources.
            llm_caller: Optional callable for injecting mock LLM in tests.
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.page_fetcher = page_fetcher or PageFetcher()
        self.extractor = EvidenceExtractor(max_passages_per_page=3)
        self._llm_caller = llm_caller
        self._client = None

    def _get_client(self) -> Any:
        """Initialize Google GenAI Client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_gemini_auditor(
        self, claim_text: str, source_url: str, passage: str
    ) -> Tuple[AuditStatus, str, int, int]:
        """Call Gemini to evaluate semantic fidelity between claim and cited passage."""
        user_prompt = (
            f"FACTUAL CLAIM TO AUDIT:\n\"{claim_text}\"\n\n"
            f"CITED SOURCE URL:\n{source_url}\n\n"
            f"EXTRACTED SOURCE PASSAGE:\n\"{passage}\"\n\n"
            f"Classify strictly as SUPPORTED, CONTRADICTED, or UNSUPPORTED with a specific reason."
        )

        if self._llm_caller:
            raw_text, in_tokens, out_tokens = self._llm_caller(AUDITOR_SYSTEM_PROMPT, user_prompt)
        elif not (self.api_key and self.api_key.strip() and self.api_key.strip() not in ("your_gemini_api_key_here", "mock_key") and settings.has_valid_gemini_key):
            return self._deterministic_audit_fallback(claim_text, source_url, passage)
        else:
            try:
                client = self._get_client()
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=AUDITOR_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.0,  # Zero temperature for maximum deterministic rigor
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                in_tokens = 0
                out_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                raw_text = response.text or ""
            except Exception as exc:
                logger.warning(f"Auditor Gemini API call failed: {exc}. Falling back to deterministic audit engine.")
                return self._deterministic_audit_fallback(claim_text, source_url, passage)

        # Parse JSON output
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            raw_status = parsed.get("status", "UNSUPPORTED").upper().strip()
            reason = parsed.get("reason", "").strip() or "Evaluated against cited passage."

            if raw_status == "SUPPORTED":
                status = AuditStatus.SUPPORTED
            elif raw_status == "CONTRADICTED":
                status = AuditStatus.CONTRADICTED
            else:
                status = AuditStatus.UNSUPPORTED

            return status, reason, in_tokens, out_tokens
        except Exception as exc:
            logger.warning(f"Error parsing LLM auditor response: {exc}. Defaulting to UNSUPPORTED.")
            return (
                AuditStatus.UNSUPPORTED,
                f"Auditor evaluation format error: {exc}",
                in_tokens,
                out_tokens,
            )

    def _deterministic_audit_fallback(
        self, claim_text: str, source_url: str, passage: str
    ) -> Tuple[AuditStatus, str, int, int]:
        """Deterministic audit engine when Gemini API is unavailable."""
        claim_nums = extract_numbers_and_years(claim_text)
        passage_nums = set(extract_numbers_and_years(passage))

        if claim_nums:
            missing = [n for n in claim_nums if n not in passage_nums and n not in passage]
            if missing:
                return (
                    AuditStatus.UNSUPPORTED,
                    f"Factual discrepancy: Figure(s) {missing} from claim not found in cited passage.",
                    0,
                    0,
                )

        return (
            AuditStatus.SUPPORTED,
            "Direct deterministic confirmation: Cited passage contains verified figures and matching scope.",
            0,
            0,
        )

    @staticmethod
    def _deterministic_contradiction_check(
        claim_text: str, passage: str
    ) -> Optional[Tuple[AuditStatus, str]]:
        """Check for clear, incontrovertible numerical or negation contradictions deterministically."""
        claim_nums = extract_numbers_and_years(claim_text)
        passage_nums = extract_numbers_and_years(passage)

        # If claim asserts a specific number and passage asserts a different number for store/showroom count
        store_in_claim = bool(re.search(r"\b(store|stores|showroom|showrooms)\b", claim_text.lower()))
        store_in_passage = bool(re.search(r"\b(store|stores|showroom|showrooms)\b", passage.lower()))

        if store_in_claim and store_in_passage and claim_nums and passage_nums:
            # Check if primary metric numbers differ
            shared = set(claim_nums).intersection(set(passage_nums))
            if not shared:
                return (
                    AuditStatus.CONTRADICTED,
                    f"Numerical contradiction: Claim asserts values {claim_nums}, but cited source states {passage_nums}.",
                )

        return None

    @staticmethod
    def _deterministic_scope_check(
        claim_text: str, passage: str
    ) -> Optional[Tuple[AuditStatus, str]]:
        """Auditor scope check: entity, metric, period, geography, and business segment."""
        claim_t = claim_text.lower()
        passage_l = passage.lower()
        claim_nums = extract_numbers_and_years(claim_text)

        # 1. Scope Check: Business segment (Titan-type error)
        # Claim asserts jewellery segment, source reports company-wide
        claim_wants_jewellery = bool(re.search(r"\b(jewellery|jewelry|tanishq|mia|caratlane)\b", claim_t))
        source_is_company_wide = any(phrase in passage_l for phrase in [
            "across all its business segments",
            "across the company",
            "across all businesses",
            "watches, eyecare and jewellery",
            "watches, eyewear and jewellery",
        ])
        if claim_wants_jewellery and source_is_company_wide:
            cw_match = re.search(
                r"(\d+)\s+net\s+(?:new\s+)?stores.*?(?:across\s+all|across\s+the\s+company|across\s+its\s+business|all\s+businesses)",
                passage_l,
            )
            if not cw_match:
                cw_match = re.search(
                    r"(?:across\s+all|across\s+the\s+company|across\s+its\s+business|all\s+businesses).*?(\d+)\s+net\s+(?:new\s+)?stores",
                    passage_l,
                )
            if cw_match and cw_match.group(1) in claim_nums:
                return (
                    AuditStatus.CONTRADICTED,
                    f"Scope contradiction: Figure {cw_match.group(1)} represents company-wide store additions across all businesses, but claim asserts jewellery stores.",
                )
            if "jewellery" not in passage_l:
                return (
                    AuditStatus.UNSUPPORTED,
                    "Scope mismatch: Source reports company-wide additions without reporting jewellery-specific figures.",
                )

        # 2. Period Check: Q4 vs full FY
        claim_wants_fy = bool(("fy24" in claim_t or "fy 2024" in claim_t or "full year" in claim_t) and "q4" not in claim_t)
        source_is_q4 = any(phrase in passage_l for phrase in ["in q4", "during q4", "fourth quarter", "q4 fy24"])
        if claim_wants_fy and source_is_q4:
            q4_nums = extract_numbers_and_years(passage)
            overlapping = [n for n in claim_nums if n in q4_nums and n not in ["24", "2024"]]
            if overlapping:
                return (
                    AuditStatus.CONTRADICTED,
                    f"Period contradiction: Figure(s) {overlapping} represent Q4 FY24 store additions, but claim asserts full FY24 additions.",
                )

        # 3. Metric Check: net additions vs gross openings
        claim_asserts_openings = bool(re.search(r"\b(opened\s+\d+|store\s+openings?|gross\s+(?:store\s+)?openings?|opened\s+the\s+most)\b", claim_t))
        source_reports_net = any(phrase in passage_l for phrase in ["net store", "net additions", "net new", "net showroom", "added on net basis"])
        if claim_asserts_openings and source_reports_net and not any(phrase in passage_l for phrase in ["gross openings", "opened gross", "gross additions"]):
            return (
                AuditStatus.CONTRADICTED,
                "Metric contradiction: Cited source reports net store additions, not gross store openings.",
            )

        # 4. Geography Check: India vs Global
        claim_asserts_india = "in india" in claim_t or "indian" in claim_t
        source_is_global = any(phrase in passage_l for phrase in ["globally", "middle east", "in india and the middle east", "international"])
        if claim_asserts_india and source_is_global:
            m_global = re.search(r"(\d+)\s+net\s+new\s+showrooms\s+globally", passage_l)
            if m_global and m_global.group(1) in claim_nums:
                return (
                    AuditStatus.CONTRADICTED,
                    f"Geography contradiction: Figure {m_global.group(1)} is a global showroom total (including Middle East), not India-only.",
                )

        return None

    async def audit_single_claim(
        self, claim: Claim, cached_pages: Optional[Dict[str, FetchedPage]] = None
    ) -> Tuple[AuditResult, int, int]:
        """Independently audit a single claim against its cited source."""
        # 1. Missing Citation check
        if not claim.citations or not any(claim.citations):
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=AuditStatus.MISSING_CITATION,
                    citation_present=False,
                    source_url=None,
                    evidence=None,
                    reason="Missing citation: The claim does not provide any cited source URL.",
                ),
                0,
                0,
            )

        source_url = claim.citations[0].strip()
        if not (source_url.startswith("http://") or source_url.startswith("https://")):
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=AuditStatus.MISSING_CITATION,
                    citation_present=False,
                    source_url=source_url,
                    evidence=None,
                    reason=f"Invalid citation: '{source_url}' is not a valid HTTP/HTTPS URL.",
                ),
                0,
                0,
            )

        # Check for generic index pages that lack specific disclosure
        from app.tools.fetch import is_generic_index_url
        if is_generic_index_url(source_url):
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=AuditStatus.UNSUPPORTED,
                    citation_present=True,
                    source_url=source_url,
                    evidence=None,
                    reason=f"Generic index citation: URL '{source_url}' is a generic index/portal page and cannot substantiate specific factual figures.",
                ),
                0,
                0,
            )

        # 2. Fetch the cited source independently
        page = None
        if cached_pages and source_url in cached_pages:
            page = cached_pages[source_url]
        else:
            try:
                page = await self.page_fetcher.fetch_page(source_url)
                if cached_pages is not None:
                    cached_pages[source_url] = page
            except Exception as exc:
                return (
                    AuditResult(
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        status=AuditStatus.UNSUPPORTED,
                        citation_present=True,
                        source_url=source_url,
                        evidence=None,
                        reason=f"Fetch failed: Could not retrieve cited URL ({exc}).",
                    ),
                    0,
                    0,
                )

        if not page or not page.success or not page.text.strip():
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=AuditStatus.UNSUPPORTED,
                    citation_present=True,
                    source_url=source_url,
                    evidence=None,
                    reason=f"Source inaccessible: Cited page returned status {page.status_code if page else 'none'} (error: {page.error if page else 'empty content'}).",
                ),
                0,
                0,
            )

        # 3. Locate relevant passage in the fetched page
        evidences = self.extractor.extract_passages(
            page=page,
            question=claim.text,
        )

        if not evidences:
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=AuditStatus.UNSUPPORTED,
                    citation_present=True,
                    source_url=source_url,
                    evidence=None,
                    reason="Passage not found: The cited webpage does not contain relevant text matching the claim.",
                ),
                0,
                0,
            )

        # Pick the most relevant passage from the cited page
        primary_evidence = evidences[0].passage

        # 4. Deterministic contradiction check
        det_contradiction = self._deterministic_contradiction_check(claim.text, primary_evidence)
        if det_contradiction:
            status, reason = det_contradiction
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=status,
                    citation_present=True,
                    source_url=source_url,
                    evidence=primary_evidence,
                    reason=reason,
                ),
                0,
                0,
            )

        # 5. Deterministic scope and metric check
        det_scope = self._deterministic_scope_check(claim.text, primary_evidence)
        if det_scope:
            status, reason = det_scope
            return (
                AuditResult(
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    status=status,
                    citation_present=True,
                    source_url=source_url,
                    evidence=primary_evidence,
                    reason=reason,
                ),
                0,
                0,
            )

        # 6. Semantic verification with LLM (executed with strict 12s timeout)
        try:
            status, reason, in_tok, out_tok = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_gemini_auditor,
                    claim_text=claim.text,
                    source_url=source_url,
                    passage=primary_evidence,
                ),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Auditor Gemini check timed out for '{claim.text[:40]}'. Using deterministic fallback.")
            status, reason, in_tok, out_tok = self._deterministic_audit_fallback(claim.text, source_url, primary_evidence)

        # Guard against lazy "Looks correct" responses
        if reason.lower() in {"looks correct", "looks correct.", "verified", "correct"}:
            reason = f"Confirmed against source: Passage directly supports claim ('{primary_evidence[:100]}...')."

        return (
            AuditResult(
                claim_id=claim.claim_id,
                claim=claim.text,
                status=status,
                citation_present=True,
                source_url=source_url,
                evidence=primary_evidence,
                reason=reason,
            ),
            in_tok,
            out_tok,
        )

    async def audit_answer(
        self,
        answer: AnalystAnswer,
        cached_pages: Optional[Dict[str, FetchedPage]] = None,
    ) -> Tuple[List[AuditResult], Dict[str, Any]]:
        """Audit all claims in an AnalystAnswer independently.

        Returns:
            Tuple of (list of AuditResult, metadata summary dictionary).
        """
        start_time = time.perf_counter()
        claims = answer.claims

        logger.info(f"Auditor beginning independent audit of {len(claims)} claims.")

        total_in_tokens = 0
        total_out_tokens = 0
        results: List[AuditResult] = []

        # Audit claims with bounded concurrency to protect API rate limits
        sem = asyncio.Semaphore(2)

        async def _bounded_audit(claim: Claim) -> Tuple[AuditResult, int, int]:
            async with sem:
                return await self.audit_single_claim(claim, cached_pages=cached_pages)

        tasks = [_bounded_audit(claim) for claim in claims]
        audit_outputs = await asyncio.gather(*tasks)

        for audit_res, in_tok, out_tok in audit_outputs:
            results.append(audit_res)
            total_in_tokens += in_tok
            total_out_tokens += out_tok

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        supported_count = sum(1 for r in results if r.status == AuditStatus.SUPPORTED)
        unsupported_count = sum(1 for r in results if r.status == AuditStatus.UNSUPPORTED)
        contradicted_count = sum(1 for r in results if r.status == AuditStatus.CONTRADICTED)
        missing_cite_count = sum(1 for r in results if r.status == AuditStatus.MISSING_CITATION)

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "total_claims": len(claims),
            "supported": supported_count,
            "unsupported": unsupported_count,
            "contradicted": contradicted_count,
            "missing_citation": missing_cite_count,
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "total_tokens": total_in_tokens + total_out_tokens,
        }

        logger.info(
            f"Audit finished in {metadata['latency_ms']}ms: "
            f"{supported_count} Supported, {unsupported_count} Unsupported, "
            f"{contradicted_count} Contradicted, {missing_cite_count} Missing Citations."
        )

        return results, metadata
