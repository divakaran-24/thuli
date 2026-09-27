"""Unit tests for ClaimVerifier, ClaimExtractor, ConflictResolver, and AnalystAgent (Milestone 3)."""

import json
import pytest

from app.agents.analyst import AnalystAgent
from app.models.schemas import AnalystAnswer, Claim, ConflictRecord, Evidence, ResearchPlan, SourceStatus
from app.verification.claim_extractor import ClaimExtractor
from app.verification.claim_verifier import ClaimVerifier, extract_numbers_and_years
from app.verification.conflict_resolver import ConflictResolver


# =====================================================================
# 1. Claim Verifier Deterministic Checks Tests
# =====================================================================

def test_extract_numbers_and_years():
    """Test extracting numbers, decimals, percentages, and commas."""
    text = "Titan added 150 stores in FY24, achieving 18.5% growth and 3,000 total outlets."
    nums = extract_numbers_and_years(text)
    assert "150" in nums
    assert "24" in nums
    assert "18.5" in nums
    assert "3000" in nums


def test_verify_citations_present():
    """Test detection of missing or invalid citations."""
    claim_valid = Claim(
        claim_id="c1",
        text="Titan opened 60 stores.",
        evidence_ids=["ev_1"],
        citations=["https://titancompany.in/news"],
    )
    ok, msg = ClaimVerifier.verify_citations_present(claim_valid)
    assert ok is True

    claim_missing = Claim(
        claim_id="c2",
        text="Titan opened 60 stores.",
        evidence_ids=["ev_1"],
        citations=[],
    )
    ok, msg = ClaimVerifier.verify_citations_present(claim_missing)
    assert ok is False
    assert "Missing citation" in msg


def test_verify_urls_were_fetched():
    """Test verifying that cited URLs were actually fetched."""
    fetched = {"https://titancompany.in/annual-report", "https://bseindia.com/filing"}

    claim_ok = Claim(
        claim_id="c1",
        text="Some fact",
        evidence_ids=["ev_1"],
        citations=["https://titancompany.in/annual-report/"],
    )
    ok, _ = ClaimVerifier.verify_urls_were_fetched(claim_ok, fetched)
    assert ok is True

    claim_unfetched = Claim(
        claim_id="c2",
        text="Some fact",
        evidence_ids=["ev_2"],
        citations=["https://invented-source.com/news"],
    )
    ok, msg = ClaimVerifier.verify_urls_were_fetched(claim_unfetched, fetched)
    assert ok is False
    assert "was never fetched" in msg


def test_verify_numerical_values_match_and_mismatch():
    """Test checking that numbers asserted in claim exist in cited evidence passage."""
    evidence = Evidence(
        evidence_id="ev_1",
        source_id="titancompany.in",
        url="https://titancompany.in/filing",
        title="Filing",
        publisher="titancompany.in",
        passage="Titan Company expanded its network by adding 60 net new stores during FY24.",
        retrieved_at="2026-09-26T12:00:00Z",
    )
    evidence_map = {"ev_1": evidence}

    # Matching numbers: "60" and "24" both present in passage
    claim_matching = Claim(
        claim_id="c1",
        text="Titan opened 60 stores in FY24.",
        evidence_ids=["ev_1"],
        citations=[evidence.url],
    )
    ok, _ = ClaimVerifier.verify_numerical_values(claim_matching, evidence_map)
    assert ok is True

    # Hallucinated number: "95" stores not in passage
    claim_hallucinated = Claim(
        claim_id="c2",
        text="Titan opened 95 stores in FY24.",
        evidence_ids=["ev_1"],
        citations=[evidence.url],
    )
    ok, msg = ClaimVerifier.verify_numerical_values(claim_hallucinated, evidence_map)
    assert ok is False
    assert "Numerical discrepancy" in msg
    assert "95" in msg


def test_complete_deterministic_verification():
    """Test verify_claim_deterministically end-to-end."""
    verifier = ClaimVerifier()
    evidence = Evidence(
        evidence_id="ev_kalyan",
        source_id="kalyanjewellers.net",
        url="https://kalyanjewellers.net/q4",
        title="Q4 Results",
        publisher="kalyanjewellers.net",
        passage="Kalyan Jewellers added 71 net new showrooms in India during FY24.",
        retrieved_at="2026-09-26T12:00:00Z",
    )
    evidence_map = {"ev_kalyan": evidence}
    fetched_urls = {"https://kalyanjewellers.net/q4"}

    claim = Claim(
        claim_id="c1",
        text="Kalyan Jewellers added 71 net new showrooms in India during FY24.",
        evidence_ids=["ev_kalyan"],
        citations=["https://kalyanjewellers.net/q4"],
    )

    valid, reason = verifier.verify_claim_deterministically(claim, evidence_map, fetched_urls)
    assert valid is True
    assert "passed" in reason.lower()


# =====================================================================
# 2. Claim Extractor Tests
# =====================================================================

def test_extract_claims_from_evidence():
    """Test generating candidate atomic claims from evidence passages."""
    ev = Evidence(
        evidence_id="ev_1",
        source_id="senco",
        url="https://sencogold.com/expansion",
        title="Expansion",
        publisher="sencogold.com",
        passage="Senco Gold opened 23 new showrooms in FY24. The total showroom count reached 159 across India.",
        retrieved_at="2026-09-26T12:00:00Z",
    )
    claims = ClaimExtractor.extract_claims_from_evidence([ev], entity="Senco Gold")
    assert len(claims) >= 1
    assert any("23" in c.text for c in claims)
    assert claims[0].citations == [ev.url]
    assert claims[0].evidence_ids == [ev.evidence_id]


def test_extract_claims_from_text_with_inline_urls():
    """Test extracting claims and inline citations from an answer text."""
    answer = (
        "Titan opened 60 stores in FY24 [https://titancompany.in/report].\n"
        "Kalyan Jewellers expanded with 71 showrooms [https://kalyanjewellers.net/q4]."
    )
    claims = ClaimExtractor.extract_claims_from_text(answer)
    assert len(claims) == 2
    assert "https://titancompany.in/report" in claims[0].citations
    assert "https://kalyanjewellers.net/q4" in claims[1].citations


# =====================================================================
# 3. Conflict Resolver Tests
# =====================================================================

def test_conflict_geographic_scope():
    """Test resolving discrepancy caused by domestic vs global scope."""
    resolver = ConflictResolver()
    disc_type, expl, resolved = resolver.inspect_discrepancy(
        text_a="Kalyan Jewellers operates 250 showrooms in India only.",
        text_b="Kalyan Jewellers operates 290 showrooms in its global international network.",
    )
    assert resolved is True
    assert disc_type == "geographic_scope"
    assert "international" in expl or "domestic" in expl


def test_conflict_fy_vs_cy():
    """Test resolving discrepancy between Fiscal Year and Calendar Year."""
    resolver = ConflictResolver()
    disc_type, expl, resolved = resolver.inspect_discrepancy(
        text_a="Titan opened 86 stores in FY24.",
        text_b="Titan opened 102 stores in calendar year 2024.",
    )
    assert resolved is True
    assert disc_type == "FY_vs_CY"
    assert "Fiscal Year" in expl or "Calendar Year" in expl


def test_conflict_net_vs_gross():
    """Test resolving net additions vs gross openings."""
    resolver = ConflictResolver()
    disc_type, expl, resolved = resolver.inspect_discrepancy(
        text_a="Retailer recorded 50 net new store additions.",
        text_b="Retailer opened 65 gross stores across the country.",
    )
    assert resolved is True
    assert disc_type == "definition_net_vs_gross"
    assert "net" in expl and "gross" in expl


def test_conflict_unresolved():
    """Test that irreconcilable conflicting numbers are marked unresolved."""
    resolver = ConflictResolver()
    disc_type, expl, resolved = resolver.inspect_discrepancy(
        text_a="The company opened 40 stores.",
        text_b="The company opened 50 stores.",
    )
    assert resolved is False
    assert disc_type == "unresolved"
    assert "could not be definitively resolved" in expl


# =====================================================================
# 4. Analyst Agent Tests (Normal & Adversarial Modes)
# =====================================================================

def test_analyst_normal_mode_synthesis():
    """Test AnalystAgent synthesis in NORMAL_ANALYST mode with mock LLM."""
    plan = ResearchPlan(
        question="How many stores did Titan open in FY24?",
        entities=["Titan Company"],
        sub_questions=["Check store openings"],
        search_queries=["Titan FY24 store additions"],
        required_evidence=["Annual report"],
        verification_requirements=["Verify store count"],
    )
    evidence = [
        Evidence(
            evidence_id="ev_titan",
            source_id="titancompany.in",
            url="https://titancompany.in/report",
            title="Titan FY24 Report",
            publisher="titancompany.in",
            passage="Titan Company added 60 net new jewellery stores in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    ]

    mock_response = {
        "answer": "Titan Company added 60 net new jewellery stores in FY24 [https://titancompany.in/report].",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Titan Company added 60 net new jewellery stores in FY24.",
                "evidence_ids": ["ev_titan"],
                "citations": ["https://titancompany.in/report"],
            }
        ],
        "unanswered_aspects": [],
    }

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        assert "RESEARCH QUESTION" in user_prompt
        assert "Titan Company" in user_prompt
        return json.dumps(mock_response), 500, 150

    analyst = AnalystAgent(api_key="mock_key", llm_caller=mock_llm)
    answer, meta = analyst.synthesize(
        question="How many stores did Titan open in FY24?",
        plan=plan,
        evidence=evidence,
        mode="NORMAL_ANALYST",
    )

    assert isinstance(answer, AnalystAnswer)
    assert answer.mode == "NORMAL_ANALYST"
    assert "60 net new jewellery stores" in answer.answer
    assert len(answer.claims) == 1
    assert answer.claims[0].citations == ["https://titancompany.in/report"]
    assert meta["input_tokens"] == 500
    assert meta["output_tokens"] == 150


def test_analyst_adversarial_mode_synthesis():
    """Test AnalystAgent synthesis in ADVERSARIAL_ANALYST mode."""
    plan = ResearchPlan(
        question="How many stores did Kalyan open?",
        entities=["Kalyan Jewellers"],
        sub_questions=["Check Kalyan showrooms"],
        search_queries=["Kalyan showroom count"],
        required_evidence=["Investor presentation"],
        verification_requirements=["Strict verification"],
    )
    evidence = [
        Evidence(
            evidence_id="ev_kalyan",
            source_id="kalyanjewellers.net",
            url="https://kalyanjewellers.net/investor",
            title="Kalyan Q4",
            publisher="kalyanjewellers.net",
            passage="Kalyan Jewellers added 71 net new showrooms during FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    ]

    mock_response = {
        "answer": "Kalyan Jewellers added 71 net new showrooms during FY24 [https://kalyanjewellers.net/investor].",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Kalyan Jewellers added 71 net new showrooms during FY24.",
                "evidence_ids": ["ev_kalyan"],
                "citations": ["https://kalyanjewellers.net/investor"],
            }
        ],
        "unanswered_aspects": [],
    }

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        # Verify strict adversarial prompt was used
        assert "ADVERSARIAL AUDITING" in system_prompt
        return json.dumps(mock_response), 600, 160

    analyst = AnalystAgent(api_key="mock_key", llm_caller=mock_llm)
    answer, meta = analyst.synthesize(
        question="How many stores did Kalyan open?",
        plan=plan,
        evidence=evidence,
        mode="ADVERSARIAL_ANALYST",
    )

    assert answer.mode == "ADVERSARIAL_ANALYST"
    assert len(answer.claims) == 1
    assert answer.claims[0].citations == ["https://kalyanjewellers.net/investor"]


def test_analyst_no_evidence_states_unavailable():
    """Test that when evidence is empty, Analyst explicitly states information was not established."""
    plan = ResearchPlan(
        question="How many stores did Unknown Retailer open?",
        entities=["Unknown Retailer"],
        sub_questions=["Search"],
        search_queries=["Unknown retailer stores"],
        required_evidence=["Filings"],
        verification_requirements=["Check"],
    )

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return "{}", 200, 50

    analyst = AnalystAgent(api_key="mock_key", llm_caller=mock_llm)
    answer, meta = analyst.synthesize(
        question="How many stores did Unknown Retailer open?",
        plan=plan,
        evidence=[],  # No evidence
        mode="NORMAL_ANALYST",
    )

    assert "The available sources did not establish" in answer.answer
    assert len(answer.unanswered_aspects) > 0


# =====================================================================
# 6. Correctness & Claim Gate Tests (Scope, Metrics, Generic BSE, Citations)
# =====================================================================

def test_claim_gate_missing_citation():
    """Verify that a claim with missing citation is rejected by Claim Gate."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_no_cite",
        text="Titan added 86 stores.",
        evidence_ids=["ev_1"],
        citations=[],
    )
    evidence_map = {
        "ev_1": Evidence(
            evidence_id="ev_1",
            source_id="s1",
            url="https://titan.in/report",
            title="Titan Report",
            publisher="titan.in",
            passage="Titan added 86 stores.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    }
    fetched_urls = {"https://titan.in/report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.is_verified is False
    assert claim.verification_status == "REJECTED"
    assert "Missing citation" in reason


def test_claim_gate_generic_bse_citation():
    """Verify that generic BSE landing page citations are strictly rejected."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_bse",
        text="Titan added 86 jewellery stores in FY24.",
        evidence_ids=["ev_bse"],
        citations=["https://www.bseindia.com/corporates/ann.html"],
    )
    evidence_map = {
        "ev_bse": Evidence(
            evidence_id="ev_bse",
            source_id="bse",
            url="https://www.bseindia.com/corporates/ann.html",
            title="BSE Corporate Announcements",
            publisher="bseindia.com",
            passage="Titan added 86 jewellery stores in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    }
    fetched_urls = {"https://www.bseindia.com/corporates/ann.html"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Generic index citation" in reason


def test_claim_gate_404_source_fetch_failed():
    """Verify that evidence from failed or unavailable sources is rejected."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_fail",
        text="Titan added 86 stores.",
        evidence_ids=["ev_fail"],
        citations=["https://titan.in/missing-404"],
    )
    evidence_map = {
        "ev_fail": Evidence(
            evidence_id="ev_fail",
            source_id="titan",
            url="https://titan.in/missing-404",
            title="Missing Page",
            publisher="titan.in",
            passage="Titan added 86 stores.",
            retrieved_at="2026-09-26T12:00:00Z",
            source_status=SourceStatus.FETCH_FAILED,
        )
    }
    fetched_urls = {"https://titan.in/missing-404"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Source unavailable" in reason


def test_claim_gate_unsupported_claim():
    """Verify that claims with numbers not present in cited passage are rejected."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_unsup",
        text="Titan added 999 jewellery stores.",
        evidence_ids=["ev_titan"],
        citations=["https://titan.in/report"],
    )
    evidence_map = {
        "ev_titan": Evidence(
            evidence_id="ev_titan",
            source_id="titan",
            url="https://titan.in/report",
            title="Titan Report",
            publisher="titan.in",
            passage="Titan added 86 net new stores.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    }
    fetched_urls = {"https://titan.in/report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Numerical discrepancy" in reason


def test_claim_gate_contradicted_claim():
    """Verify that contradictory statistics are caught and rejected."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_contra",
        text="Kalyan Jewellers added 120 net new showrooms during FY24.",
        evidence_ids=["ev_kalyan"],
        citations=["https://kalyan.in/report"],
    )
    evidence_map = {
        "ev_kalyan": Evidence(
            evidence_id="ev_kalyan",
            source_id="kalyan",
            url="https://kalyan.in/report",
            title="Kalyan Report",
            publisher="kalyan.in",
            passage="Kalyan Jewellers added 71 net new showrooms during FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
        )
    }
    fetched_urls = {"https://kalyan.in/report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Numerical discrepancy" in reason


def test_claim_gate_titan_company_wide_vs_segment():
    """Test Titan-type error: Company-wide store additions (86) claimed as jewellery stores."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_titan_scope",
        text="Titan Company added 86 net jewellery stores in FY24.",
        evidence_ids=["ev_titan_cw"],
        citations=["https://titancompany.in/report"],
        business_segment="Jewellery",
    )
    evidence_map = {
        "ev_titan_cw": Evidence(
            evidence_id="ev_titan_cw",
            source_id="titan",
            url="https://titancompany.in/report",
            title="Titan Annual Report",
            publisher="titancompany.in",
            passage="Titan Company added 86 net stores in FY24 across all its business segments (watches, eyewear, jewellery).",
            retrieved_at="2026-09-26T12:00:00Z",
            business_segment="Company-Wide",
        )
    }
    fetched_urls = {"https://titancompany.in/report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Scope mismatch" in reason


def test_claim_gate_q4_vs_full_fy():
    """Test period error: Q4 additions (29) claimed as full FY24 additions."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_q4_mismatch",
        text="Titan added 29 jewellery stores during full FY24.",
        evidence_ids=["ev_q4"],
        citations=["https://titancompany.in/q4-report"],
        period="FY24",
    )
    evidence_map = {
        "ev_q4": Evidence(
            evidence_id="ev_q4",
            source_id="titan",
            url="https://titancompany.in/q4-report",
            title="Titan Q4",
            publisher="titancompany.in",
            passage="Titan added 29 jewellery stores in Q4 FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
            period_type="Q4",
        )
    }
    fetched_urls = {"https://titancompany.in/q4-report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Period mismatch" in reason


def test_claim_gate_net_vs_gross():
    """Test metric error: Net additions (35) claimed as gross openings."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_net_gross",
        text="Kalyan opened 35 stores in FY24.",
        evidence_ids=["ev_net"],
        citations=["https://kalyan.in/results"],
        metric="gross_store_openings",
    )
    evidence_map = {
        "ev_net": Evidence(
            evidence_id="ev_net",
            source_id="kalyan",
            url="https://kalyan.in/results",
            title="Kalyan Results",
            publisher="kalyan.in",
            passage="Kalyan recorded 35 net additions in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
            metric="net_store_additions",
        )
    }
    fetched_urls = {"https://kalyan.in/results"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Metric mismatch" in reason


def test_claim_gate_india_vs_global():
    """Test geography error: Global showroom additions (71) claimed as India-only."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_geo",
        text="Kalyan Jewellers added 71 net new showrooms in India.",
        evidence_ids=["ev_geo"],
        citations=["https://kalyan.in/footprint"],
        geography="India",
    )
    evidence_map = {
        "ev_geo": Evidence(
            evidence_id="ev_geo",
            source_id="kalyan",
            url="https://kalyan.in/footprint",
            title="Kalyan Footprint",
            publisher="kalyan.in",
            passage="Kalyan Jewellers added 71 net new showrooms globally including Middle East.",
            retrieved_at="2026-09-26T12:00:00Z",
            geography="Global",
        )
    }
    fetched_urls = {"https://kalyan.in/footprint"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is False
    assert claim.verification_status == "REJECTED"
    assert "Geography mismatch" in reason


def test_ranking_with_incomparable_metrics():
    """Test that ranking questions with disparate metrics refuse to rank and explain incomparability."""
    plan = ResearchPlan(
        question="Which Indian jewellery retailer opened the most new stores in FY24?",
        entities=["Titan", "Senco"],
        search_queries=["Titan FY24", "Senco FY24"],
    )
    evidence = [
        Evidence(
            evidence_id="ev_1",
            source_id="titan",
            url="https://titan.in/report",
            title="Titan Report",
            publisher="titan.in",
            passage="Titan added 86 net new stores in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
            entity="Titan",
            metric="net_store_additions",
            period="FY24",
        ),
        Evidence(
            evidence_id="ev_2",
            source_id="senco",
            url="https://senco.in/report",
            title="Senco Report",
            publisher="senco.in",
            passage="Senco opened 23 new showrooms in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
            entity="Senco",
            metric="gross_store_openings",
            period="FY24",
        ),
    ]

    analyst = AnalystAgent(api_key="mock_key", llm_caller=lambda s, u: ("{}", 100, 50))
    answer, _ = analyst.synthesize(
        question="Which Indian jewellery retailer opened the most new stores in FY24?",
        plan=plan,
        evidence=evidence,
        mode="NORMAL_ANALYST",
    )

    assert "I could not establish a reliable like-for-like ranking from the available evidence" in answer.answer
    assert "differing metrics, periods, or business scopes" in answer.answer


def test_claim_gate_successful_evidence_backed_claim():
    """Verify that a claim with verified evidence, valid citation, and matching scope passes."""
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="c_valid",
        text="Titan added 60 jewellery stores in India in FY24.",
        evidence_ids=["ev_valid"],
        citations=["https://titancompany.in/fy24-annual-report"],
        entity="Titan",
        metric="net_store_additions",
        period="FY24",
        business_segment="Jewellery",
        geography="India",
    )
    evidence_map = {
        "ev_valid": Evidence(
            evidence_id="ev_valid",
            source_id="titan",
            url="https://titancompany.in/fy24-annual-report",
            title="Titan Annual Report",
            publisher="titancompany.in",
            passage="Titan added 60 jewellery stores in India in FY24.",
            retrieved_at="2026-09-26T12:00:00Z",
            entity="Titan",
            metric="net_store_additions",
            period="FY24",
            business_segment="Jewellery",
            geography="India",
            source_status=SourceStatus.FETCHED,
        )
    }
    fetched_urls = {"https://titancompany.in/fy24-annual-report"}
    ok, reason = verifier.gate_claim(claim, evidence_map, fetched_urls)
    assert ok is True
    assert claim.is_verified is True
    assert claim.verification_status == "VERIFIED"
    assert claim.rejection_reason is None


@pytest.mark.asyncio
async def test_failed_source_recovery():
    """Test that when a source 404s, it is marked unavailable and evidence is only extracted from successful alternative."""
    from app.tools.fetch import PageFetcher
    from app.tools.extraction import EvidenceExtractor

    async def mock_fetch(url: str):
        if "404" in url:
            return "Not Found", 404
        return "<html><body><p>Titan added 60 jewellery stores in FY24.</p></body></html>", 200

    fetcher = PageFetcher(tavily_api_key="mock_key", httpx_fetch_fn=mock_fetch)
    extractor = EvidenceExtractor()

    page_fail = await fetcher.fetch_page("https://titan.in/broken-404")
    assert page_fail.success is False
    assert page_fail.source_status == SourceStatus.FETCH_FAILED

    ev_fail = extractor.extract_passages(page_fail, question="How many stores did Titan open?", entities=["Titan"])
    assert len(ev_fail) == 0  # Never creates evidence from failed pages

    page_succ = await fetcher.fetch_page("https://titan.in/valid-report")
    assert page_succ.success is True
    assert page_succ.source_status == SourceStatus.FETCHED

    ev_succ = extractor.extract_passages(page_succ, question="How many stores did Titan open?", entities=["Titan"])
    assert len(ev_succ) > 0  # Successfully extracted from alternative


def test_no_evidence_uncertainty_statement():
    """Verify that when no evidence is available, Analyst states explicit uncertainty."""
    plan = ResearchPlan(
        question="Which jewellery retailer opened the most stores in FY24?",
        entities=["Titan"],
        search_queries=["stores opened"],
    )
    analyst = AnalystAgent(api_key="mock_key", llm_caller=lambda s, u: ("{}", 100, 50))
    answer, _ = analyst.synthesize(
        question="Which jewellery retailer opened the most stores in FY24?",
        plan=plan,
        evidence=[],
    )
    assert "I could not verify this from the available sources." in answer.answer
    assert len(answer.claims) == 0


def test_demo_evidence_cannot_be_presented_as_live():
    """Verify that DEMO evidence carries source_type='DEMO' and does not masquerade as 'LIVE'."""
    ev_demo = Evidence(
        evidence_id="ev_demo_1",
        source_id="demo",
        url="https://demo.research/titan-fy24-annual-report",
        title="Demo Titan Report",
        publisher="demo.research",
        passage="Titan added 86 net stores in FY24.",
        retrieved_at="2026-09-26T12:00:00Z",
        source_type="DEMO",
    )
    claims = ClaimExtractor.extract_claims_from_evidence([ev_demo])
    assert len(claims) == 1
    assert claims[0].source_type == "DEMO"
    assert claims[0].source_type != "LIVE"

