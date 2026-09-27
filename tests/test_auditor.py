"""Unit tests for AuditorAgent (Milestone 4)."""

import json
import pytest

from app.agents.auditor import AuditorAgent
from app.models.schemas import AnalystAnswer, AuditResult, AuditStatus, Claim
from app.tools.fetch import FetchedPage, PageFetcher


# =====================================================================
# 1. Auditor Missing Citation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_missing_citation():
    """Verify that claims without a citation are flagged as MISSING_CITATION."""
    claim = Claim(
        claim_id="c1",
        text="Titan opened 100 new stores in 2024.",
        evidence_ids=[],
        citations=[],
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, in_tok, out_tok = await auditor.audit_single_claim(claim)

    assert result.status == AuditStatus.MISSING_CITATION
    assert result.citation_present is False
    assert "Missing citation" in result.reason
    assert result.evidence is None


@pytest.mark.asyncio
async def test_auditor_invalid_url_citation():
    """Verify that non-http URLs are flagged as MISSING_CITATION."""
    claim = Claim(
        claim_id="c2",
        text="Titan opened 100 new stores.",
        evidence_ids=[],
        citations=["not-a-valid-url"],
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, in_tok, out_tok = await auditor.audit_single_claim(claim)

    assert result.status == AuditStatus.MISSING_CITATION
    assert result.citation_present is False
    assert "Invalid citation" in result.reason


# =====================================================================
# 2. Source Inaccessible / Fetch Failure Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_source_fetch_failure():
    """Verify that when a cited source fails to fetch (404), claim is marked UNSUPPORTED."""
    async def mock_404_fetch(url: str):
        return "Not found", 404

    fetcher = PageFetcher(tavily_api_key="mock_key", httpx_fetch_fn=mock_404_fetch)
    auditor = AuditorAgent(api_key="mock_key", page_fetcher=fetcher)

    claim = Claim(
        claim_id="c3",
        text="Kalyan opened 40 showrooms.",
        evidence_ids=[],
        citations=["https://kalyanjewellers.net/broken-link"],
    )

    result, in_tok, out_tok = await auditor.audit_single_claim(claim)
    assert result.status == AuditStatus.UNSUPPORTED
    assert result.citation_present is True
    assert "Source inaccessible" in result.reason or "Fetch failed" in result.reason


# =====================================================================
# 3. Contradicted Claim Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_numerical_contradiction():
    """Verify that when claim numbers conflict with cited source, claim is marked CONTRADICTED."""
    source_url = "https://example.com/company-results"
    sample_text = (
        "During the fiscal year, Company X accelerated its network expansion "
        "by successfully opening 35 stores across domestic markets."
    )

    mock_page = FetchedPage(
        url=source_url,
        title="Company X Annual Results",
        publisher="example.com",
        published_date="2024-05-01",
        text=sample_text,
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=50.0,
        success=True,
    )

    claim = Claim(
        claim_id="c4",
        text="Company X opened 50 stores during the fiscal year.",
        evidence_ids=[],
        citations=[source_url],
    )

    auditor = AuditorAgent(api_key="mock_key")
    result, in_tok, out_tok = await auditor.audit_single_claim(
        claim, cached_pages={source_url: mock_page}
    )

    assert result.status == AuditStatus.CONTRADICTED
    assert "contradiction" in result.reason.lower()
    assert "50" in result.reason or "35" in result.reason
    assert result.evidence is not None


# =====================================================================
# 4. Supported Claim Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_supported_claim():
    """Verify that when cited source confirms claim, claim is marked SUPPORTED."""
    source_url = "https://titancompany.in/press-release"
    sample_text = (
        "Titan Company Limited expanded its jewellery retail footprint "
        "by adding 60 net new stores during FY2024."
    )

    mock_page = FetchedPage(
        url=source_url,
        title="Titan FY24 Press Release",
        publisher="titancompany.in",
        published_date="2024-05-15",
        text=sample_text,
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=45.0,
        success=True,
    )

    claim = Claim(
        claim_id="c5",
        text="Titan Company added 60 net new stores during FY2024.",
        evidence_ids=[],
        citations=[source_url],
    )

    # Mock semantic LLM check
    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return json.dumps({
            "status": "SUPPORTED",
            "reason": "Direct confirmation: Source states Titan added 60 net new stores in FY2024."
        }), 200, 50

    auditor = AuditorAgent(api_key="mock_key", llm_caller=mock_llm)
    result, in_tok, out_tok = await auditor.audit_single_claim(
        claim, cached_pages={source_url: mock_page}
    )

    assert result.status == AuditStatus.SUPPORTED
    assert "Direct confirmation" in result.reason
    assert result.evidence is not None
    assert "60 net new stores" in result.evidence


# =====================================================================
# 5. Full Answer Audit Aggregation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_audit_answer_aggregation():
    """Verify audit_answer audits multiple claims concurrently and produces complete summary metrics."""
    source_url = "https://titancompany.in/report"
    page = FetchedPage(
        url=source_url,
        title="Titan Report",
        publisher="titancompany.in",
        published_date="2024-05-01",
        text="Titan Company added 60 net new stores in FY24.",
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=30.0,
        success=True,
    )

    claim1 = Claim(
        claim_id="c1",
        text="Titan Company added 60 net new stores in FY24.",
        evidence_ids=[],
        citations=[source_url],
    )
    claim2 = Claim(
        claim_id="c2",
        text="Titan Company operates 5000 stores.",
        evidence_ids=[],
        citations=[],  # Missing citation
    )

    answer = AnalystAnswer(
        answer="Titan added 60 stores [https://titancompany.in/report] and operates 5000 stores.",
        claims=[claim1, claim2],
        unanswered_aspects=[],
        mode="NORMAL_ANALYST",
    )

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return json.dumps({
            "status": "SUPPORTED",
            "reason": "Direct match in source report."
        }), 150, 40

    auditor = AuditorAgent(api_key="mock_key", llm_caller=mock_llm)
    results, meta = await auditor.audit_answer(answer, cached_pages={source_url: page})

    assert len(results) == 2
    assert meta["total_claims"] == 2
    assert meta["supported"] == 1
    assert meta["missing_citation"] == 1
    assert results[0].status == AuditStatus.SUPPORTED
    assert results[1].status == AuditStatus.MISSING_CITATION


# =====================================================================
# 5. Auditor Scope & Generic Citation Tests
# =====================================================================

@pytest.mark.asyncio
async def test_auditor_rejects_generic_bse_citation():
    """Verify Auditor flags generic BSE landing URL as UNSUPPORTED."""
    claim = Claim(
        claim_id="c_bse",
        text="Titan Company added 86 net jewellery stores.",
        evidence_ids=[],
        citations=["https://www.bseindia.com/corporates/ann.html"],
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, _, _ = await auditor.audit_single_claim(claim)
    assert result.status == AuditStatus.UNSUPPORTED
    assert "Generic index" in result.reason or "portal page" in result.reason


@pytest.mark.asyncio
async def test_auditor_scope_mismatch_company_wide_vs_segment():
    """Verify Auditor catches company-wide stores claimed as jewellery segment."""
    url = "https://titancompany.in/reports/annual-2024"
    page = FetchedPage(
        url=url,
        title="Titan Report",
        publisher="titancompany.in",
        published_date="2024-05-01",
        text="Titan added 86 net stores across all its business segments including watches, eyecare and jewellery.",
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=20.0,
        success=True,
    )
    claim = Claim(
        claim_id="c_titan",
        text="Titan Company added 86 net jewellery stores in FY24.",
        evidence_ids=[],
        citations=[url],
        business_segment="Jewellery",
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, _, _ = await auditor.audit_single_claim(claim, cached_pages={url: page})
    assert result.status == AuditStatus.CONTRADICTED
    assert "Scope mismatch" in result.reason or "company-wide" in result.reason


@pytest.mark.asyncio
async def test_auditor_scope_mismatch_q4_vs_full_fy():
    """Verify Auditor catches Q4 figures claimed as full FY figures."""
    url = "https://titancompany.in/reports/q4-fy24"
    page = FetchedPage(
        url=url,
        title="Titan Q4 Update",
        publisher="titancompany.in",
        published_date="2024-05-01",
        text="Titan added 29 jewellery stores in Q4 FY24.",
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=20.0,
        success=True,
    )
    claim = Claim(
        claim_id="c_q4",
        text="Titan added 29 jewellery stores during full FY24.",
        evidence_ids=[],
        citations=[url],
        period="FY24",
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, _, _ = await auditor.audit_single_claim(claim, cached_pages={url: page})
    assert result.status == AuditStatus.CONTRADICTED
    assert "Period mismatch" in result.reason or "Q4" in result.reason


@pytest.mark.asyncio
async def test_auditor_metric_mismatch_net_vs_gross():
    """Verify Auditor catches net additions claimed as stores opened."""
    url = "https://kalyan.in/report"
    page = FetchedPage(
        url=url,
        title="Kalyan Report",
        publisher="kalyan.in",
        published_date="2024-05-01",
        text="Kalyan Jewellers recorded 35 net additions in FY24.",
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=20.0,
        success=True,
    )
    claim = Claim(
        claim_id="c_net",
        text="Kalyan Jewellers opened 35 stores in FY24.",
        evidence_ids=[],
        citations=[url],
        metric="gross_store_openings",
    )
    auditor = AuditorAgent(api_key="mock_key")
    result, _, _ = await auditor.audit_single_claim(claim, cached_pages={url: page})
    assert result.status == AuditStatus.CONTRADICTED
    assert "Metric mismatch" in result.reason or "net" in result.reason

