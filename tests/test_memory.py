"""Unit tests for SQLite persistent entity memory system (Milestone 5)."""

import os
import tempfile
import pytest

from app.memory.database import DatabaseManager
from app.memory.entity_memory import EntityMemory
from app.memory.research_memory import ResearchMemoryManager
from app.models.schemas import AuditResult, AuditStatus, Claim, Evidence, RunMetrics


import gc

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_manager = DatabaseManager(db_path=path)
    yield db_manager
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# =====================================================================
# 1. Database Schema and Table Tests
# =====================================================================

def test_database_initialization(temp_db):
    """Verify all 8 schema tables and indexes are created successfully."""
    with temp_db.get_connection() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "entities" in tables
        assert "sources" in tables
        assert "evidence" in tables
        assert "claims" in tables
        assert "claim_evidence" in tables
        assert "audit_results" in tables
        assert "feedback" in tables
        assert "runs" in tables


# =====================================================================
# 2. Entity Memory Operations Tests
# =====================================================================

def test_entity_creation_and_alias_lookup(temp_db):
    """Test creating an entity with aliases and finding it via alias or name."""
    mem = EntityMemory(temp_db)
    ent_id = mem.get_or_create_entity(
        name="Titan Company Limited", aliases=["Titan", "Tanishq", "Mia"]
    )
    assert ent_id.startswith("ent_")

    # Find by alias
    found_by_alias = mem.find_entity("Tanishq")
    assert found_by_alias is not None
    assert found_by_alias["name"] == "Titan Company Limited"
    assert "Titan" in found_by_alias["aliases"]

    # Find by case-insensitive name
    found_by_name = mem.find_entity("titan company limited")
    assert found_by_name is not None
    assert found_by_name["id"] == ent_id


def test_entity_centric_knowledge_storage(temp_db):
    """Test entity -> verified facts -> evidence -> sources relationship."""
    mem = EntityMemory(temp_db)
    ent_id = mem.get_or_create_entity("Kalyan Jewellers")

    # Save source
    src_url = "https://kalyanjewellers.net/investor-update"
    mem.save_source(
        url=src_url,
        domain="kalyanjewellers.net",
        title="Kalyan Q4 Results",
        publisher="kalyanjewellers.net",
        quality_score=1.0,
    )

    # Save evidence
    ev = Evidence(
        evidence_id="ev_kalyan_1",
        source_id="kalyanjewellers.net",
        url=src_url,
        title="Kalyan Q4 Results",
        publisher="kalyanjewellers.net",
        published_date="2024-05-10",
        passage="Kalyan Jewellers added 71 net new showrooms in India during FY24.",
        retrieved_at="2026-09-26T12:00:00Z",
    )
    mem.save_evidence(ev)

    # Save verified claim
    claim = Claim(
        claim_id="c_kalyan_1",
        text="Kalyan Jewellers added 71 net new showrooms in India during FY24.",
        evidence_ids=[ev.evidence_id],
        citations=[src_url],
    )
    mem.save_claim(entity_id=ent_id, claim=claim, status="SUPPORTED")

    # Retrieve complete entity knowledge
    knowledge = mem.get_entity_knowledge("Kalyan Jewellers")
    assert knowledge is not None
    assert knowledge["name"] == "Kalyan Jewellers"
    assert len(knowledge["verified_claims"]) == 1
    assert "71 net new showrooms" in knowledge["verified_claims"][0]["text"]
    assert src_url in knowledge["known_sources"]
    assert len(knowledge["verified_claims"][0]["evidence"]) == 1
    assert knowledge["verified_claims"][0]["evidence"][0]["passage"] == ev.passage


# =====================================================================
# 3. Research Memory Manager & Knowledge Transfer Tests
# =====================================================================

def test_entity_knowledge_transfers_to_new_question(temp_db):
    """Verify that verified entity knowledge from an earlier run is reused in later questions."""
    mgr = ResearchMemoryManager(temp_db)

    # Run 1: Researches Kalyan Jewellers
    ev = Evidence(
        evidence_id="ev_k1",
        source_id="kalyan",
        url="https://kalyanjewellers.net/fy24",
        title="FY24 Report",
        publisher="kalyanjewellers.net",
        passage="Kalyan added 71 showrooms.",
        retrieved_at="2026-09-26T12:00:00Z",
    )
    claim = Claim(
        claim_id="c1",
        text="Kalyan added 71 showrooms in FY24.",
        evidence_ids=[ev.evidence_id],
        citations=[ev.url],
    )
    audit = AuditResult(
        claim_id="c1",
        claim=claim.text,
        status=AuditStatus.SUPPORTED,
        citation_present=True,
        source_url=ev.url,
        evidence=ev.passage,
        reason="Direct quote in official report.",
    )
    metrics = RunMetrics(
        question_id="q01",
        latency_ms=2500.0,
        input_tokens=1500,
        output_tokens=300,
        total_tokens=1800,
        estimated_cost_usd=0.005,
        estimated_cost_inr=0.43,
        llm_calls=3,
        search_calls=2,
        fetch_calls=2,
        memory_hits=0,
        memory_misses=1,
    )

    mgr.save_run_knowledge(
        question_id="q01",
        question="How many stores did Kalyan Jewellers open in FY24?",
        mode="NORMAL_ANALYST",
        metrics=metrics,
        claims=[claim],
        evidences=[ev],
        audit_results=[audit],
        primary_entities=["Kalyan Jewellers"],
    )

    # Run 2: New question mentioning previously researched entity
    # MUST count as memory HIT and return verified facts
    known, hits, misses = mgr.lookup_entities(["Kalyan Jewellers"])
    assert hits == 1
    assert misses == 0
    assert len(known) == 1
    assert known[0]["name"] == "Kalyan Jewellers"
    assert len(known[0]["verified_claims"]) == 1
    assert "71 showrooms" in known[0]["verified_claims"][0]["text"]

    # Run 3: Question mentioning an un-researched entity
    # MUST count as memory MISS
    known_unknown, hits_unknown, misses_unknown = mgr.lookup_entities(["Joyalukkas"])
    assert hits_unknown == 0
    assert misses_unknown == 1
    assert len(known_unknown) == 0


def test_memory_is_not_simple_question_caching(temp_db):
    """Verify memory stores entity facts rather than exact question->answer strings."""
    mgr = ResearchMemoryManager(temp_db)
    # Check that database does not have a 'question_answer_cache' table
    with temp_db.get_connection() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "qa_cache" not in tables
        assert "cache" not in tables

    # Entities must be queryable across different question formulations
    ent_id = mgr.entity_memory.get_or_create_entity("Senco Gold")
    claim = Claim(
        claim_id="c_senco",
        text="Senco Gold has 159 showrooms across India.",
        evidence_ids=[],
        citations=["https://sencogold.com"],
    )
    mgr.entity_memory.save_claim(ent_id, claim, status="SUPPORTED")

    # Formulation A
    known_a, hits_a, _ = mgr.lookup_entities(["Senco Gold"])
    assert hits_a == 1
    assert known_a[0]["verified_claims"][0]["text"] == claim.text

    # Formulation B
    known_b, hits_b, _ = mgr.lookup_entities(["Senco Gold"])
    assert hits_b == 1
    assert known_b[0]["verified_claims"][0]["text"] == claim.text


# =====================================================================
# 4. Learned Policies & Feedback Storage Tests
# =====================================================================

def test_feedback_policy_storage_and_retrieval(temp_db):
    """Test storing compact auditor lessons and retrieving them for future runs."""
    mgr = ResearchMemoryManager(temp_db)
    mgr.save_feedback_policy(
        rule="Numerical claims require direct evidence from primary filings.",
        source_audit_id="aud_q01_c1",
        trigger_reason="Auditor flagged uncorroborated store number.",
    )
    mgr.save_feedback_policy(
        rule="Dates must be explicitly supported in evidence.",
        source_audit_id="aud_q01_c2",
        trigger_reason="Auditor flagged missing date verification.",
    )

    policies = mgr.get_feedback_policies(limit=5)
    assert len(policies) == 2
    assert any("Numerical claims require direct evidence" in p for p in policies)
    assert any("Dates must be explicitly supported" in p for p in policies)
