"""Unit tests for WebSearchTool, PageFetcher, and EvidenceExtractor (Milestone 2)."""

import pytest

from app.models.schemas import Evidence, SearchResult
from app.tools.extraction import EvidenceExtractor
from app.tools.fetch import FetchedPage, PageFetcher
from app.tools.search import evaluate_source_quality, WebSearchTool


# =====================================================================
# 1. Source Quality Scoring Tests
# =====================================================================

def test_source_quality_tier_evaluation():
    """Verify source quality scoring prioritizes primary and high-credibility domains."""
    # Tier 1: Primary regulatory & company investor portals
    assert evaluate_source_quality("https://www.bseindia.com/corporates/ann.html") == 1.0
    assert evaluate_source_quality("https://titancompany.in/investors") == 1.0
    assert evaluate_source_quality("https://kalyanjewellers.net/investor-relations") == 1.0

    # Tier 1: Government domains
    assert evaluate_source_quality("https://commerce.gov.in/reports") == 0.95

    # Tier 2: Reputable financial & business press
    assert evaluate_source_quality("https://www.reuters.com/business/retail") == 0.85
    assert evaluate_source_quality("https://economictimes.indiatimes.com/markets") == 0.85
    assert evaluate_source_quality("https://www.livemint.com/companies") == 0.85

    # Tier 3: Industry publications
    assert evaluate_source_quality("https://retailjewellerindia.com/news") == 0.70

    # Generic or unknown domains
    assert evaluate_source_quality("https://random-lifestyle-blog.xyz/post") == 0.35


# =====================================================================
# 2. Web Search Tool Tests
# =====================================================================

@pytest.mark.asyncio
async def test_web_search_empty_query():
    """Test that an empty search query returns empty results with error metadata."""
    tool = WebSearchTool(api_key="mock_key")
    results, meta = await tool.search("   ")
    assert results == []
    assert meta["error"] == "Empty query"
    assert meta["results_count"] == 0


@pytest.mark.asyncio
async def test_web_search_with_mock():
    """Test WebSearchTool parses mock Tavily results into ranked SearchResult objects."""
    async def mock_search(query: str, max_results: int):
        return [
            {
                "url": "https://random-blog.com/jewellery",
                "title": "Jewellery trends 2024",
                "snippet": "Many stores opened.",
                "score": 0.8,
            },
            {
                "url": "https://www.bseindia.com/filings/titan.pdf",
                "title": "Titan Company BSE Filing FY24",
                "snippet": "Titan added 150 net new jewellery stores during FY24.",
                "score": 0.9,
            },
        ]

    tool = WebSearchTool(api_key="mock_key", search_fn=mock_search)
    results, meta = await tool.search("Titan store openings")

    assert len(results) == 2
    assert meta["results_count"] == 2
    assert meta["error"] is None

    # bseindia.com should rank first due to higher source quality tier
    assert results[0].source_domain == "bseindia.com"
    assert results[0].score > results[1].score
    assert isinstance(results[0], SearchResult)


@pytest.mark.asyncio
async def test_web_search_timeout_handling():
    """Test that search timeout is handled gracefully without crashing."""
    import asyncio

    async def slow_search(query: str, max_results: int):
        await asyncio.sleep(0.5)
        return []

    tool = WebSearchTool(api_key="mock_key", timeout=0.1, search_fn=slow_search)
    results, meta = await tool.search("Titan query")

    assert results == []
    assert meta["results_count"] == 0
    assert meta["error"] is not None


# =====================================================================
# 3. Page Fetcher Tests (Tavily Extract & Fallback)
# =====================================================================

@pytest.mark.asyncio
async def test_fetcher_primary_tavily_extract():
    """Test PageFetcher uses Tavily Extract when it succeeds."""
    async def mock_tavily_extract(url: str):
        return {
            "results": [
                {
                    "url": url,
                    "title": "Titan Investor Presentation Q4",
                    "published_date": "2024-05-10",
                    "raw_content": "Titan Company network expansion reached 3,000 stores across all brands. Jewellery network expanded with 86 net additions in FY24.",
                }
            ]
        }

    fetcher = PageFetcher(tavily_api_key="mock_key", tavily_extract_fn=mock_tavily_extract)
    page = await fetcher.fetch_page("https://titancompany.in/investors")

    assert page.success is True
    assert page.method_used == "tavily_extract"
    assert page.publisher == "titancompany.in"
    assert "86 net additions" in page.text
    assert page.published_date == "2024-05-10"


@pytest.mark.asyncio
async def test_fetcher_fallback_to_httpx_bs4():
    """Test PageFetcher falls back to httpx + BeautifulSoup when Tavily Extract fails."""
    async def failing_tavily_extract(url: str):
        raise RuntimeError("Tavily Extract API error")

    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kalyan Jewellers Footprint Update</title>
        <meta name="pubdate" content="2024-04-15" />
    </head>
    <body>
        <nav><a href="/">Home</a><a href="/about">About</a></nav>
        <h1>Showroom Expansion</h1>
        <p>Kalyan Jewellers added 71 net new showrooms in India and Middle East during FY24.</p>
        <footer><p>Copyright 2024. All rights reserved.</p></footer>
    </body>
    </html>
    """

    async def mock_httpx_fetch(url: str):
        return sample_html, 200

    fetcher = PageFetcher(
        tavily_api_key="mock_key",
        tavily_extract_fn=failing_tavily_extract,
        httpx_fetch_fn=mock_httpx_fetch,
    )
    page = await fetcher.fetch_page("https://kalyanjewellers.net/press-release")

    assert page.success is True
    assert page.method_used == "httpx_bs4"
    assert page.title == "Kalyan Jewellers Footprint Update"
    assert "71 net new showrooms" in page.text
    # Nav and footer tags must be stripped
    assert "Home" not in page.text
    assert "All rights reserved" not in page.text


@pytest.mark.asyncio
async def test_fetcher_http_error_handling():
    """Test that HTTP errors (like 404) are handled cleanly without fabricating text."""
    async def mock_404_fetch(url: str):
        return "Not Found", 404

    fetcher = PageFetcher(
        tavily_api_key="mock_key",
        tavily_extract_fn=None,
        httpx_fetch_fn=mock_404_fetch,
    )
    page = await fetcher.fetch_page("https://example.com/nonexistent")

    assert page.success is False
    assert page.status_code == 404
    assert page.text == ""
    assert "HTTP 404" in page.error


# =====================================================================
# 4. Evidence Extractor Tests
# =====================================================================

def test_evidence_extractor_relevant_passages():
    """Test EvidenceExtractor extracts relevant passages and ignores boilerplate."""
    page_text = (
        "Subscribe to our newsletter for daily market updates!\n\n"
        "Titan Company Limited reported its financial performance for FY2024. "
        "The Jewellery division recorded total network of 850 Tanishq and Mia stores, "
        "opening 60 new stores in India during the fiscal year.\n\n"
        "Click here to accept all cookies and read our privacy policy.\n\n"
        "The watches and wearables division opened 30 stores, while eyewear added 15 stores.\n\n"
        "All rights reserved. Unauthorized reproduction is prohibited."
    )

    page = FetchedPage(
        url="https://titancompany.in/news/fy24-results",
        title="Titan FY24 Annual Report Summary",
        publisher="titancompany.in",
        published_date="2024-05-08",
        text=page_text,
        status_code=200,
        method_used="httpx_bs4",
        latency_ms=120.0,
        success=True,
    )

    extractor = EvidenceExtractor(max_passages_per_page=2)
    evidences = extractor.extract_passages(
        page=page,
        question="How many jewellery stores did Titan open in FY2024?",
        entities=["Titan Company", "Tanishq"],
    )

    assert len(evidences) > 0
    top_evidence = evidences[0]
    assert isinstance(top_evidence, Evidence)
    assert top_evidence.url == page.url
    assert top_evidence.publisher == "titancompany.in"
    assert "Titan Company Limited" in top_evidence.passage
    assert "60 new stores" in top_evidence.passage
    # Boilerplate should not be present
    assert "cookies" not in top_evidence.passage.lower()
    assert "newsletter" not in top_evidence.passage.lower()
