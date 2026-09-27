"""Analyst Agent: Synthesizes concise, evidence-backed answers with strict citations."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

from app.config import settings
from app.models.schemas import AnalystAnswer, Claim, ConflictRecord, Evidence, ResearchPlan
from app.verification.claim_verifier import ClaimVerifier

logger = logging.getLogger(__name__)

NORMAL_SYSTEM_PROMPT = """You are an expert Research Analyst Agent.
Your responsibility is to synthesize a factual, precise, and concise research answer based SOLELY on the verified evidence provided to you.

STRICT OPERATIONAL RULES:
1. Every factual statement, number, date, or metric MUST have a direct citation to the source URL.
2. NEVER invent facts, sources, URLs, dates, numbers, or people.
3. If an aspect of the question cannot be answered from the provided evidence, you MUST explicitly state:
   "The available sources did not establish [X]."
4. Never make assumptions or guesses.
5. If conflicting numbers were identified and resolved, state the resolution clearly (e.g. FY vs CY, domestic vs global).
6. Format citations inline using direct URLs, e.g. "Titan added 60 stores [https://titancompany.in/news]."
7. You MUST cite ONLY URLs that appear in the AVAILABLE EVIDENCE PASSAGES. Do NOT cite URLs that are not in the evidence list.

Output MUST be a valid JSON object matching this schema:
{
  "answer": "Your full synthesized answer containing inline citations.",
  "claims": [
    {
      "claim_id": "c1",
      "text": "Specific factual claim asserted in the answer.",
      "evidence_ids": ["ev_1"],
      "citations": ["https://exact-url-cited.com"]
    }
  ],
  "unanswered_aspects": [
    "The available sources did not establish X."
  ]
}
"""

ADVERSARIAL_SYSTEM_PROMPT = """You are an expert Research Analyst Agent operating under STRICT ADVERSARIAL AUDITING.

WARNING: An independent Auditor Agent will inspect and verify every single factual claim you produce against live fetched sources.
- Any claim missing a direct citation will be penalized as MISSING_CITATION.
- Any claim whose numbers, dates, or assertions do not strictly match the cited source passage will be marked CONTRADICTED or UNSUPPORTED.
- You must use ONLY directly relevant evidence and precise citations.
- You MUST cite ONLY URLs that appear in the AVAILABLE EVIDENCE PASSAGES.
- If information is missing or uncertain, do NOT attempt to fill the gap. Explicitly write: "The available sources did not establish [X]."
- Do NOT guess. Zero hallucinations tolerated.

Output MUST be a valid JSON object matching this schema:
{
  "answer": "Your rigorous synthesized answer containing inline citations.",
  "claims": [
    {
      "claim_id": "c1",
      "text": "Specific factual claim asserted in the answer.",
      "evidence_ids": ["ev_1"],
      "citations": ["https://exact-url-cited.com"]
    }
  ],
  "unanswered_aspects": [
    "The available sources did not establish X."
  ]
}
"""


class AnalystAgent:
    """Agent responsible for factual synthesis, citation binding, and claim generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        adversarial_model: Optional[str] = None,
        llm_caller: Optional[Callable[[str, str], tuple[str, int, int]]] = None,
    ) -> None:
        """Initialize AnalystAgent."""
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.adversarial_model = adversarial_model or settings.gemini_adversarial_model
        self._llm_caller = llm_caller
        self._client = None
        self.verifier = ClaimVerifier()

    def _get_client(self) -> Any:
        """Initialize the google.genai Client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def build_user_prompt(
        self,
        question: str,
        plan: ResearchPlan,
        evidence: List[Evidence],
        conflicts: Optional[List[ConflictRecord]] = None,
        memory_facts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construct the prompt containing research evidence for answer synthesis."""
        prompt_parts = [
            f"RESEARCH QUESTION:\n{question.strip()}\n",
            f"REQUIRED EVIDENCE FOCUS:\n" + "\n".join(f"- {req}" for req in plan.required_evidence) + "\n",
        ]

        if memory_facts:
            mem_lines = []
            for item in memory_facts:
                entity = item.get("name", "Unknown")
                claims = item.get("verified_claims", [])
                for c in claims:
                    mem_lines.append(f"- [{entity}] {c.get('text', '')} (Citations: {c.get('citations', [])})")
            if mem_lines:
                prompt_parts.append("PREVIOUSLY VERIFIED MEMORY KNOWLEDGE:\n" + "\n".join(mem_lines) + "\n")

        if conflicts:
            conflict_lines = []
            for c in conflicts:
                status = "RESOLVED" if c.resolved else "UNRESOLVED"
                conflict_lines.append(
                    f"- Conflict ({status}, Type: {c.discrepancy_type}): {c.claim_a} vs {c.claim_b}. Resolution: {c.resolution}"
                )
            prompt_parts.append("RESOLVED CONFLICTS & DISCREPANCIES:\n" + "\n".join(conflict_lines) + "\n")

        # Provide evidence passages with explicit IDs and URLs
        evidence_lines = []
        for ev in evidence:
            date_info = f" (Date: {ev.published_date})" if ev.published_date else ""
            evidence_lines.append(
                f"[{ev.evidence_id}] Source: {ev.publisher} | URL: {ev.url}{date_info}\n"
                f"Passage: \"{ev.passage}\"\n"
            )

        prompt_parts.append(
            f"AVAILABLE EVIDENCE PASSAGES ({len(evidence)} items):\n"
            + ("\n".join(evidence_lines) if evidence_lines else "NO EVIDENCE FOUND.")
        )

        prompt_parts.append(
            "\nGenerate the synthesized JSON research answer now with citations and individual claims."
        )
        return "\n".join(prompt_parts)

    def _call_gemini(
        self, system_prompt: str, user_prompt: str, active_model: str
    ) -> tuple[str, int, int]:
        """Call Gemini API for synthesis."""
        if self._llm_caller:
            return self._llm_caller(system_prompt, user_prompt)

        # Fallback if Gemini API key is missing or placeholder
        if not (self.api_key and self.api_key.strip() and self.api_key.strip() != "your_gemini_api_key_here"):
            return self._fallback_synthesize(user_prompt)

        try:
            client = self._get_client()
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for factual fidelity
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            response = client.models.generate_content(
                model=active_model,
                contents=user_prompt,
                config=config,
            )

            in_tokens = 0
            out_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            raw_text = response.text or ""
            return raw_text, in_tokens, out_tokens
        except Exception as exc:
            logger.warning(f"Analyst Gemini API call failed: {exc}. Using fallback synthesis.")
            return self._fallback_synthesize(user_prompt)

    def _fallback_synthesize(self, user_prompt: str) -> tuple[str, int, int]:
        """Deterministic fallback when GEMINI_API_KEY is not configured: generates zero speculative claims."""
        logger.info("Using evidence-grounded deterministic synthesis engine (GEMINI_API_KEY not configured).")
        # Do not hallucinate or guess claims; synthesize strictly from verified evidence in synthesize()
        data = {
            "answer": "",
            "claims": [],
            "missing_evidence": [],
            "unresolved_conflicts": [],
        }
        return json.dumps(data), 50, 50

    @staticmethod
    def _parse_json(raw_text: str) -> Dict[str, Any]:
        """Extract JSON cleanly from model response."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return json.loads(cleaned[start_idx : end_idx + 1])
        return json.loads(cleaned) if cleaned else {}

    def synthesize(
        self,
        question: str,
        plan: ResearchPlan,
        evidence: List[Evidence],
        conflicts: Optional[List[ConflictRecord]] = None,
        memory_facts: Optional[List[Dict[str, Any]]] = None,
        mode: str = "NORMAL_ANALYST",
    ) -> tuple[AnalystAnswer, Dict[str, Any]]:
        """Synthesize answer strictly from verified research evidence.

        Workflow:
        1. If evidence is empty, return explicit uncertainty declaration.
        2. Generate candidate claims from evidence objects.
        3. Pass candidate claims through the Python Claim Gate (scope, period, metric, source status).
        4. Check like-for-like comparability if question demands a ranking.
        5. Generate final answer strictly from verified claims.
        """
        start_time = time.perf_counter()
        is_adversarial = mode == "ADVERSARIAL_ANALYST"
        system_prompt = ADVERSARIAL_SYSTEM_PROMPT if is_adversarial else NORMAL_SYSTEM_PROMPT
        active_model = self.adversarial_model if is_adversarial else self.model

        evidence_map: Dict[str, Evidence] = {ev.evidence_id: ev for ev in evidence}
        fetched_urls: Set[str] = {ev.url for ev in evidence}

        # 1. No evidence -> explicit uncertainty statement
        if not evidence:
            answer_text = (
                "I could not verify this from the available sources. "
                "The available sources did not establish sufficient evidence to answer this question."
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return AnalystAnswer(
                answer=answer_text,
                claims=[],
                unanswered_aspects=[f"No verified evidence found for '{question}'."],
                mode=mode,
            ), {
                "latency_ms": round(latency_ms, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "mode": mode,
                "model": active_model,
                "claims_count": 0,
            }

        user_prompt = self.build_user_prompt(
            question=question,
            plan=plan,
            evidence=evidence,
            conflicts=conflicts,
            memory_facts=memory_facts,
        )

        logger.info(f"Synthesizing answer ({mode}) for question: '{question[:60]}...'")

        raw_text, in_tokens, out_tokens = self._call_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            active_model=active_model,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 2. Extract candidate claims
        from app.verification.claim_extractor import ClaimExtractor

        parsed = {}
        try:
            parsed = self._parse_json(raw_text)
        except Exception:
            pass

        llm_claims_raw = parsed.get("claims", [])
        candidate_claims: List[Claim] = []

        if llm_claims_raw:
            for i, rc in enumerate(llm_claims_raw):
                t = rc.get("text", "").strip()
                if t:
                    cites = rc.get("citations", [])
                    if not cites:
                        extracted = re.findall(r"https?://[^\s)\]]+", t)
                        cites = [u.rstrip(".,;)\"']") for u in extracted]
                    if not cites and parsed.get("answer"):
                        extracted = re.findall(r"https?://[^\s)\]]+", parsed["answer"])
                        cites = [u.rstrip(".,;)\"']") for u in extracted]
                    if not cites and evidence:
                        cites = [evidence[0].url]

                    eids = rc.get("evidence_ids", [])
                    if not eids:
                        for c_url in cites:
                            norm_u = c_url.rstrip("/").lower()
                            for ev in evidence:
                                if ev.url.rstrip("/").lower() == norm_u and ev.evidence_id not in eids:
                                    eids.append(ev.evidence_id)
                    if not eids and evidence:
                        eids = [evidence[0].evidence_id]

                    candidate_claims.append(
                        Claim(
                            claim_id=rc.get("claim_id", f"c_llm_{i+1}"),
                            text=t,
                            evidence_ids=eids,
                            citations=cites,
                            entity=rc.get("entity"),
                            metric=rc.get("metric"),
                            period=rc.get("period"),
                            business_segment=rc.get("business_segment"),
                            geography=rc.get("geography"),
                        )
                    )
        else:
            # Deterministic extraction directly from Evidence objects
            candidate_claims = ClaimExtractor.extract_claims_from_evidence(evidence)

        # 3. Pass candidate claims through the Python Claim Gate
        verified_claims: List[Claim] = []
        rejected_claims: List[Claim] = []

        for c in candidate_claims:
            ok, reason = self.verifier.gate_claim(c, evidence_map, fetched_urls)
            if ok:
                verified_claims.append(c)
            else:
                rejected_claims.append(c)
                logger.info(
                    f"CLAIM_REJECTED: id={c.claim_id}, reason='{reason}', "
                    f"metric='{c.metric}', period='{c.period}', "
                    f"segment='{c.business_segment}', text='{c.text}'"
                )

        # 4. Check ranking questions for comparability
        is_ranking = any(w in question.lower() for w in ["most", "highest", "rank", "which", "leader", "largest"])

        if not verified_claims:
            # All candidate claims rejected or no verified claims
            answer_text = (
                "I could not verify this from the available sources. "
                "The available sources did not establish sufficient evidence to answer this question."
            )
            unanswered = [
                f"Candidate claims rejected by verification gate: {[c.rejection_reason for c in rejected_claims[:3]]}"
            ]
        elif is_ranking:
            # Check like-for-like comparability
            entities = {c.entity for c in verified_claims if c.entity}
            metrics = {c.metric for c in verified_claims if c.metric}
            periods = {c.period for c in verified_claims if c.period}
            segments = {c.business_segment for c in verified_claims if c.business_segment}
            geographies = {c.geography for c in verified_claims if c.geography}

            has_incomparability = (
                len(metrics) > 1 or len(periods) > 1 or len(segments) > 1 or len(geographies) > 1
                or any(m is None for m in metrics)
            )

            if has_incomparability:
                answer_text = (
                    "I could not establish a reliable like-for-like ranking from the available evidence.\n"
                    "The available disclosures report differing metrics, periods, or business scopes across retailers:\n"
                )
                for vc in verified_claims:
                    cite_url = vc.citations[0] if vc.citations else "source"
                    answer_text += f"- {vc.text} [Source: {cite_url}]\n"
                unanswered = ["Incomparable metrics or scopes prevented a direct ranking."]
            else:
                # Fully comparable
                answer_text = f"Based on verified comparable disclosures for {list(periods)[0] if periods else 'the period'}:\n"
                for vc in verified_claims:
                    cite_url = vc.citations[0] if vc.citations else "source"
                    answer_text += f"- {vc.text} [Source: {cite_url}]\n"
                unanswered = []
        else:
            # Standard question: prioritize the fluent synthesized answer if verified claims exist
            raw_ans = parsed.get("answer", "").strip() if isinstance(parsed, dict) else ""
            if raw_ans and len(raw_ans) > 20 and any(c in raw_ans for c in ["http://", "https://", "["]):
                answer_text = raw_ans
            else:
                parts = []
                for vc in verified_claims:
                    cite_url = vc.citations[0] if vc.citations else "source"
                    parts.append(f"{vc.text} [{cite_url}]")
                answer_text = " ".join(parts) if parts else (raw_ans or "No verified answer could be synthesized.")
            unanswered = []

        # Return candidate claims so Auditor can audit all proposed claims (catching unsupported ones)
        claims_for_auditor = candidate_claims if llm_claims_raw else verified_claims

        analyst_answer = AnalystAnswer(
            answer=answer_text,
            claims=claims_for_auditor,
            unanswered_aspects=unanswered,
            mode=mode,
        )

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "mode": mode,
            "model": active_model,
            "claims_count": len(verified_claims),
            "rejected_claims_count": len(rejected_claims),
        }

        logger.info(
            f"Analyst ({mode}) finished in {metadata['latency_ms']}ms with {len(verified_claims)} verified claims "
            f"({len(rejected_claims)} rejected)."
        )
        return analyst_answer, metadata
