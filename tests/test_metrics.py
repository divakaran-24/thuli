"""Unit tests for Budget constraints, Cost accounting, Concurrency, and TraceLogger (Milestone 7)."""

import asyncio
import os
import tempfile
import time
import pytest

from app.logging.trace_logger import TraceLogger, sanitize_secrets
from app.models.schemas import Budget, RunMetrics


# =====================================================================
# 1. Budget Limits and Cost Accounting Tests
# =====================================================================

def test_budget_call_and_token_limits():
    """Verify that Budget enforces strict operational limits for calls and tokens."""
    budget = Budget(
        max_llm_calls=2,
        max_search_calls=2,
        max_fetch_calls=2,
        max_input_tokens=2000,
        max_output_tokens=500,
        max_time_seconds=120.0,
    )

    assert budget.can_call_llm() is True
    assert budget.can_search() is True
    assert budget.can_fetch() is True

    # Consume 1 search and 1 fetch
    budget.consume_search(1)
    budget.consume_fetch(1)
    assert budget.search_calls == 1
    assert budget.fetch_calls == 1

    # Consume second search -> reaches limit
    budget.consume_search(1)
    assert budget.can_search() is False

    # Consume LLM tokens
    budget.consume_llm(input_tokens=1000, output_tokens=300)
    assert budget.can_call_llm() is True
    assert budget.llm_calls == 1

    # Second LLM call reaches max_llm_calls limit
    budget.consume_llm(input_tokens=500, output_tokens=100)
    assert budget.can_call_llm() is False


def test_budget_hard_time_limit():
    """Verify is_time_exceeded triggers when elapsed time exceeds max_time_seconds."""
    budget = Budget(max_time_seconds=0.05)
    assert budget.is_time_exceeded() is False
    time.sleep(0.06)
    assert budget.is_time_exceeded() is True
    assert budget.can_call_llm() is False


def test_exact_cost_calculation_usd_and_inr():
    """Verify deterministic token and tool cost calculation in USD and INR."""
    budget = Budget()
    budget.consume_llm(input_tokens=1_000_000, output_tokens=1_000_000)
    budget.consume_search(count=10)
    budget.consume_fetch(count=10)

    # Pricing: $0.15 / 1M prompt, $0.60 / 1M completion, $0.005 / search, $0.005 / fetch
    # Token cost = 0.15 + 0.60 = 0.75
    # Tool cost = 10 * 0.005 + 10 * 0.005 = 0.10
    # Expected total USD = 0.85
    # Expected INR at 86.50 = 0.85 * 86.50 = 73.525
    usd, inr = budget.calculate_cost(
        cost_per_m_in=0.15,
        cost_per_m_out=0.60,
        search_cost=0.005,
        fetch_cost=0.005,
        usd_to_inr=86.50,
    )
    assert usd == 0.85
    assert inr == 73.525


# =====================================================================
# 2. Trace Logger & Secret Sanitization Tests
# =====================================================================

def test_trace_logger_event_recording():
    """Verify TraceLogger writes and reads chronological JSONL trace events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = TraceLogger(question_id="q01", logs_dir=tmp_dir)

        # Log planning event
        logger.log_event(
            stage="planning",
            agent="PlannerAgent",
            action="formulate_plan",
            input_summary={"question": "What is Kalyan store count?"},
            output_summary={"queries": ["query1", "query2"]},
            latency_ms=150.2,
        )

        # Log search event
        logger.log_event(
            stage="search",
            agent="WebSearchTool",
            action="execute_search",
            input_summary={"query": "query1"},
            output_summary={"results": 3},
            latency_ms=300.5,
        )

        events = logger.read_events()
        assert len(events) == 2
        assert events[0]["stage"] == "planning"
        assert events[0]["agent"] == "PlannerAgent"
        assert events[0]["latency_ms"] == 150.2
        assert events[1]["stage"] == "search"
        assert events[1]["output_summary"]["results"] == 3


def test_trace_logger_scrubs_api_secrets():
    """Verify that TraceLogger automatically scrubs API keys from logged events."""
    fake_gemini_key = "AIzaSyD-fakeKeySecret1234567890abcdef"
    fake_tavily_key = "tvly-fakeKeySecret1234567890abcdef123"

    input_data = {
        "gemini_key": fake_gemini_key,
        "message": f"Calling API with {fake_tavily_key}",
    }

    sanitized = sanitize_secrets(input_data)
    assert fake_gemini_key not in sanitized["gemini_key"]
    assert "[MASKED_API_KEY]" in sanitized["gemini_key"]
    assert fake_tavily_key not in sanitized["message"]
    assert "[MASKED_API_KEY]" in sanitized["message"]


# =====================================================================
# 3. Parallelism Concurrency Tests
# =====================================================================

@pytest.mark.asyncio
async def test_async_parallel_concurrency():
    """Verify that asyncio.gather runs multiple I/O bound tasks concurrently."""
    async def mock_async_task(duration: float):
        await asyncio.sleep(duration)
        return duration

    start_time = time.perf_counter()
    # Run 4 tasks of 0.1s each concurrently
    tasks = [mock_async_task(0.1) for _ in range(4)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_time

    assert len(results) == 4
    # If run sequentially, would take >= 0.4s. Concurrently should take < 0.25s
    assert elapsed < 0.25
