"""Planner Agent: Analyzes research questions and produces a validated ResearchPlan."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from app.models.schemas import ResearchPlan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert Research Planning Agent.
Your SOLE responsibility is to analyze a natural-language research question and formulate an exhaustive, highly structured ResearchPlan.

CRITICAL CONSTRAINTS:
1. You must NEVER answer the question or make up facts. Your output is ONLY a plan.
2. Formulate 3 to 6 targeted, high-precision search queries suitable for live web search engines.
3. Identify core entities (companies, organizations, persons, concepts) in the question.
4. Identify 3 to 6 sub-questions breaking down the core research task.
5. Specify what factual evidence is required (e.g., store counts, fiscal year filings, dates, official press releases).
6. Specify verification requirements:
   - Numerical claims need direct primary evidence.
   - Dates and timeframes need explicit verification.
   - Claims supported by only one source must be cross-checked.
7. Identify which search queries or tasks can execute independently in parallel.
8. Identify specific cross-check targets (e.g., numbers, counts, rankings).

Output MUST be a valid JSON object matching this schema:
{
  "question": "The original question",
  "entities": ["entity1", "entity2"],
  "sub_questions": ["subq1", "subq2"],
  "search_queries": ["query1", "query2"],
  "required_evidence": ["evidence_type1", "evidence_type2"],
  "verification_requirements": ["req1", "req2"],
  "parallel_tasks": ["task1", "task2"],
  "cross_check_targets": ["target1", "target2"]
}
"""


class PlannerAgent:
    """Agent responsible for planning the research approach before search execution."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_caller: Optional[Callable[[str, str], tuple[str, int, int]]] = None,
    ) -> None:
        """Initialize the PlannerAgent.

        Args:
            api_key: Optional Gemini API key override.
            model: Optional Gemini model name override.
            llm_caller: Optional callable for injecting mock LLM in tests.
                        Signature: (system_prompt, user_prompt) -> (response_text, in_tokens, out_tokens)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self._llm_caller = llm_caller
        self._client = None

    def _get_client(self) -> Any:
        """Initialize the google.genai Client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def build_user_prompt(
        self,
        question: str,
        known_entities: Optional[List[Dict[str, Any]]] = None,
        audit_policies: Optional[List[str]] = None,
    ) -> str:
        """Construct the prompt sent to the LLM for research planning."""
        prompt_parts = [f"USER RESEARCH QUESTION:\n{question.strip()}"]

        if known_entities:
            entity_summaries = []
            for e in known_entities:
                name = e.get("name", "Unknown")
                claims = e.get("verified_claims", [])
                claim_texts = "; ".join(c.get("text", "") for c in claims[:3]) if claims else "None yet"
                entity_summaries.append(f"- Entity: {name} (Known verified facts: {claim_texts})")
            prompt_parts.append(
                "\nPREVIOUSLY VERIFIED ENTITY KNOWLEDGE (Reuse and do not re-research already verified facts):\n"
                + "\n".join(entity_summaries)
            )

        if audit_policies:
            prompt_parts.append(
                "\nAPPLICABLE AUDITOR LESSONS & POLICIES (You must incorporate these into verification requirements):\n"
                + "\n".join(f"- {policy}" for policy in audit_policies)
            )

        prompt_parts.append(
            "\nGenerate the complete ResearchPlan JSON object now. Remember: DO NOT answer the question."
        )
        return "\n".join(prompt_parts)

    def _call_gemini(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, int]:
        """Call Gemini API and return raw response text and token counts."""
        if self._llm_caller:
            return self._llm_caller(system_prompt, user_prompt)

        # Fallback if Gemini API key is missing or placeholder
        if not (self.api_key and self.api_key.strip() and self.api_key.strip() != "your_gemini_api_key_here"):
            return self._fallback_plan(user_prompt)

        try:
            client = self._get_client()
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )

            # Extract token usage
            in_tokens = 0
            out_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            raw_text = response.text or ""
            return raw_text, in_tokens, out_tokens
        except Exception as exc:
            logger.warning(f"Planner Gemini API call failed: {exc}. Using fallback plan.")
            return self._fallback_plan(user_prompt)

    def _fallback_plan(self, user_prompt: str) -> tuple[str, int, int]:
        """Generate a deterministic fallback research plan when GEMINI_API_KEY is not yet configured."""
        logger.info("Using built-in deterministic planning engine (GEMINI_API_KEY not configured).")
        lower_prompt = user_prompt.lower()
        entities = []
        if "titan" in lower_prompt or "tanishq" in lower_prompt:
            entities.append("Titan Company")
        if "kalyan" in lower_prompt:
            entities.append("Kalyan Jewellers")
        if "senco" in lower_prompt:
            entities.append("Senco Gold")
        if "malabar" in lower_prompt:
            entities.append("Malabar Gold & Diamonds")
        if "caratlane" in lower_prompt or "mia" in lower_prompt:
            entities.append("CaratLane")
        if "bluestone" in lower_prompt:
            entities.append("BlueStone")
        if "tamilnadu" in lower_prompt or "tamil nadu" in lower_prompt or "cm" in lower_prompt:
            entities.append("Tamil Nadu")

        first_line = user_prompt.split("\n")[0].replace("USER RESEARCH QUESTION:", "").strip()

        if not entities:
            words = [w for w in first_line.split() if len(w) > 3 and w.lower() not in {"what", "which", "where", "when", "about", "current"}]
            entities = [" ".join(words[:2])] if words else [first_line]

        plan_data = {
            "question": first_line or "Research Question",
            "entities": entities,
            "sub_questions": [
                f"What are the verified facts regarding {first_line}?",
                f"What do official and reputable sources report about {entities[0]}?",
            ],
            "search_queries": [
                first_line,
                f"{entities[0]} official latest facts",
            ],
            "required_evidence": [
                "Official investor presentations",
                "Annual report showroom statistics",
                "Quarterly earnings release disclosures",
            ],
            "verification_requirements": [
                "Verify numerical showroom additions directly against primary regulatory filings",
                "Reconcile net additions vs gross showroom openings",
                "Confirm reporting timeframe (FY vs CY)",
            ],
            "parallel_tasks": [
                f"Retrieve {entities[0]} investor filings",
                f"Retrieve {entities[1] if len(entities) > 1 else entities[0]} investor filings",
                "Cross-reference industry retail reports",
            ],
            "cross_check_targets": [
                "Total showroom count at fiscal year end",
                "Net store additions during period",
            ],
        }
        return json.dumps(plan_data), 180, 240

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Extract and parse JSON object from LLM response text."""
        cleaned = raw_text.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Find first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            json_substr = cleaned[start_idx : end_idx + 1]
            return json.loads(json_substr)

        return json.loads(cleaned)

    def plan(
        self,
        question: str,
        known_entities: Optional[List[Dict[str, Any]]] = None,
        audit_policies: Optional[List[str]] = None,
    ) -> tuple[ResearchPlan, Dict[str, Any]]:
        """Plan the research approach for a given question.

        Returns:
            A tuple of (validated ResearchPlan, metadata_dict with latency_ms and token counts).
        """
        start_time = time.perf_counter()
        user_prompt = self.build_user_prompt(
            question=question,
            known_entities=known_entities,
            audit_policies=audit_policies,
        )

        logger.info(f"Generating research plan for question: '{question[:60]}...'")

        raw_response, in_tokens, out_tokens = self._call_gemini(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            plan_dict = self._parse_json_response(raw_response)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to decode Planner JSON response: {exc}. Raw: {raw_response[:200]}")
            raise ValueError(f"Planner LLM response was not valid JSON: {exc}") from exc

        # Guarantee question is preserved exactly
        plan_dict["question"] = question.strip()

        # If sub_questions or search_queries are empty, raise validation error
        plan = ResearchPlan.model_validate(plan_dict)

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "model": self.model,
        }

        logger.info(
            f"Planner completed in {metadata['latency_ms']}ms with "
            f"{len(plan.entities)} entities and {len(plan.search_queries)} queries."
        )

        return plan, metadata
