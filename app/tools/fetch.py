"""Page fetching tool: Tavily Extract as primary with httpx + BeautifulSoup fallback."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.models.schemas import SourceStatus

logger = logging.getLogger(__name__)

GENERIC_INDEX_URL_PATTERNS = [
    r"^https?://(?:www\.)?bseindia\.com/corporates/ann\.html(?:\?.*)?$",
    r"^https?://(?:www\.)?nseindia\.com/companies-listing(?:\?.*)?$",
    r"^https?://(?:www\.)?bseindia\.com/?$",
    r"^https?://(?:www\.)?nseindia\.com/?$",
    r"^https?://(?:www\.)?bseindia\.com/index\.html?$",
    r"^https?://(?:www\.)?nseindia\.com/index\.html?$",
]


def is_generic_index_url(url: str) -> bool:
    """Check if a URL is a generic landing or index page rather than a specific disclosure document."""
    norm = url.strip().rstrip("/")
    for pat in GENERIC_INDEX_URL_PATTERNS:
        if re.match(pat, norm, re.IGNORECASE):
            # If it has a specific parameter indicating a specific announcement or filing, allow
            if any(p in url.lower() for p in ["id=", "doc=", "file=", "report=", "ann_id="]):
                return False
            return True
    return False


@dataclass
class FetchedPage:
    """Container for webpage content and extraction metadata."""

    url: str
    title: str
    publisher: str
    published_date: Optional[str]
    text: str
    status_code: int
    method_used: str  # "tavily_extract" or "httpx_bs4"
    latency_ms: float
    success: bool
    error: Optional[str] = None
    source_status: SourceStatus = SourceStatus.FETCHED
    source_type: str = "LIVE"


DEMO_PAGES: Dict[str, Dict[str, Any]] = {
    "https://demo.research/titan-fy24-annual-report": {
        "title": "Titan Company Limited Annual Report FY24",
        "publisher": "titancompany.in",
        "published_date": "2024-05-15",
        "text": (
            "During FY2023-24, Titan Company Limited added 86 net stores across all its business segments, "
            "including Watches & Wearables, Eyecare, and Jewellery. Within the Jewellery business segment, "
            "Tanishq, Mia, and Zoya added net new stores in India. Separately, in Q4 FY24, the jewellery segment "
            "opened 29 new stores."
        ),
    },
    "https://demo.research/kalyan-fy24-presentation": {
        "title": "Kalyan Jewellers India Limited Investor Presentation FY24",
        "publisher": "kalyanjewellers.net",
        "published_date": "2024-05-10",
        "text": (
            "In FY24, Kalyan Jewellers added 71 net new showrooms globally, including 58 showrooms in non-South "
            "markets in India and 13 showrooms in the Middle East. Total showroom network expanded to 217 showrooms."
        ),
    },
    "https://demo.research/senco-fy24-disclosure": {
        "title": "Senco Gold Limited BSE Disclosure FY24",
        "publisher": "bseindia.com",
        "published_date": "2024-05-08",
        "text": (
            "Senco Gold Limited reported 23 gross store openings during FY24 across eastern and northern India, "
            "with total operational showrooms reaching 159."
        ),
    },
}


class PageFetcher:
    """Asynchronous webpage content fetcher with Tavily Extract primary and httpx fallback."""

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        tavily_extract_fn: Optional[Callable[..., Any]] = None,
        httpx_fetch_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize PageFetcher.

        Args:
            tavily_api_key: Optional Tavily API key override.
            timeout: Optional per-fetch timeout in seconds.
            tavily_extract_fn: Optional mock for Tavily Extract during testing.
            httpx_fetch_fn: Optional mock for httpx fetch during testing.
        """
        self.tavily_api_key = tavily_api_key or settings.tavily_api_key
        self.timeout = timeout or settings.fetch_timeout_seconds
        self._tavily_extract_fn = tavily_extract_fn
        self._httpx_fetch_fn = httpx_fetch_fn
        self._tavily_client = None

    def _get_tavily_client(self) -> Any:
        """Lazily initialize Tavily client."""
        if self._tavily_client is None:
            from tavily import AsyncTavilyClient
            self._tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
        return self._tavily_client

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract clean domain/publisher name from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    async def fetch_page(self, url: str) -> FetchedPage:
        """Fetch and extract webpage content, trying Tavily Extract first then httpx + BS4.

        Args:
            url: Target webpage URL.

        Returns:
            FetchedPage containing extracted text, metadata, and status.
        """
        start_time = time.perf_counter()
        domain = self._extract_domain(url)

        # Check demo pages registry for offline/demo mode testing
        if url in DEMO_PAGES:
            demo_data = DEMO_PAGES[url]
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(f"Serving registered DEMO page: {url}")
            return FetchedPage(
                url=url,
                title=demo_data["title"],
                publisher=demo_data["publisher"],
                published_date=demo_data["published_date"],
                text=demo_data["text"],
                status_code=200,
                method_used="demo_registry",
                latency_ms=round(latency_ms, 2),
                success=True,
                source_status=SourceStatus.FETCHED,
                source_type="DEMO",
            )

        # Reject generic index / portal pages that lack specific disclosure text
        if is_generic_index_url(url):
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(f"Rejected generic index page citation: {url}")
            return FetchedPage(
                url=url,
                title=f"Generic Portal: {domain}",
                publisher=domain,
                published_date=None,
                text="",
                status_code=200,
                method_used="rejected_generic",
                latency_ms=round(latency_ms, 2),
                success=False,
                error="Generic index/portal page without specific corporate disclosure document",
                source_status=SourceStatus.REJECTED,
                source_type="LIVE",
            )

        logger.info(f"Fetching webpage: {url}")

        # 1. Attempt Primary: Tavily Extract
        tavily_page = await self._fetch_via_tavily(url, domain, start_time)
        if tavily_page and tavily_page.success and len(tavily_page.text.strip()) > 100:
            logger.info(f"Successfully fetched {url} via Tavily Extract ({len(tavily_page.text)} chars)")
            return tavily_page

        logger.info(f"Tavily Extract unavailable or insufficient for {url}. Falling back to httpx + BS4.")

        # 2. Attempt Fallback: httpx + BeautifulSoup
        fallback_page = await self._fetch_via_httpx(url, domain, start_time)
        if fallback_page.success:
            logger.info(f"Successfully fetched {url} via httpx+BS4 fallback ({len(fallback_page.text)} chars)")
        else:
            logger.warning(f"Failed to fetch {url}: {fallback_page.error} - Source marked as unavailable.")

        return fallback_page

    async def _fetch_via_tavily(
        self, url: str, domain: str, start_time: float
    ) -> Optional[FetchedPage]:
        """Attempt page content extraction using Tavily Extract."""
        try:
            if self._tavily_extract_fn:
                res = await asyncio.wait_for(
                    self._tavily_extract_fn(url),
                    timeout=self.timeout,
                )
            else:
                if not self.tavily_api_key or self.tavily_api_key.strip() in ("", "your_tavily_api_key_here"):
                    return None
                client = self._get_tavily_client()
                res = await asyncio.wait_for(
                    client.extract(urls=[url]),
                    timeout=self.timeout,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if isinstance(res, dict):
                results = res.get("results", [])
                if results and isinstance(results, list):
                    item = results[0]
                    raw_content = item.get("raw_content", "") or ""
                    if raw_content and len(raw_content.strip()) > 50:
                        return FetchedPage(
                            url=url,
                            title=item.get("title", f"Source: {domain}"),
                            publisher=domain,
                            published_date=item.get("published_date"),
                            text=raw_content.strip(),
                            status_code=200,
                            method_used="tavily_extract",
                            latency_ms=round(latency_ms, 2),
                            success=True,
                        )
        except Exception as exc:
            logger.debug(f"Tavily Extract attempt failed for {url}: {exc}")

        return None

    async def _fetch_via_httpx(
        self, url: str, domain: str, start_time: float
    ) -> FetchedPage:
        """Fallback: Fetch HTML directly via httpx and parse with BeautifulSoup."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Mock injection for testing
        if self._httpx_fetch_fn:
            try:
                html_content, status_code = await asyncio.wait_for(
                    self._httpx_fetch_fn(url),
                    timeout=self.timeout,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                if status_code >= 400:
                    return FetchedPage(
                        url=url,
                        title=f"Source: {domain}",
                        publisher=domain,
                        published_date=None,
                        text="",
                        status_code=status_code,
                        method_used="httpx_bs4",
                        latency_ms=round(latency_ms, 2),
                        success=False,
                        error=f"HTTP {status_code}",
                        source_status=SourceStatus.FETCH_FAILED,
                    )
                return self._parse_html(url, domain, html_content, status_code, latency_ms)
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return FetchedPage(
                    url=url,
                    title=f"Source: {domain}",
                    publisher=domain,
                    published_date=None,
                    text="",
                    status_code=0,
                    method_used="httpx_bs4",
                    latency_ms=round(latency_ms, 2),
                    success=False,
                    error=str(exc),
                    source_status=SourceStatus.FETCH_FAILED,
                )

        # Real httpx client
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=self.timeout,
            verify=False,  # Avoid SSL failures on government or older corporate portals
        ) as client:
            last_error = None
            for attempt in range(1, 3):
                try:
                    response = await client.get(url)
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

                    if response.status_code >= 400:
                        return FetchedPage(
                            url=url,
                            title=f"Source: {domain}",
                            publisher=domain,
                            published_date=None,
                            text="",
                            status_code=response.status_code,
                            method_used="httpx_bs4",
                            latency_ms=round(latency_ms, 2),
                            success=False,
                            error=f"HTTP {response.status_code}",
                            source_status=SourceStatus.FETCH_FAILED,
                        )

                    return self._parse_html(
                        url=url,
                        domain=domain,
                        html=response.text,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
                except httpx.TimeoutException:
                    last_error = f"httpx request timed out after {self.timeout}s"
                    logger.warning(f"{last_error} on attempt {attempt}/2 for {url}")
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(f"httpx fetch failed for {url} (attempt {attempt}/2): {last_error}")
                    if attempt < 2:
                        await asyncio.sleep(0.5)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return FetchedPage(
                url=url,
                title=f"Source: {domain}",
                publisher=domain,
                published_date=None,
                text="",
                status_code=0,
                method_used="httpx_bs4",
                latency_ms=round(latency_ms, 2),
                success=False,
                error=last_error or "Direct fetch failed",
                source_status=SourceStatus.FETCH_FAILED,
            )

    @staticmethod
    def _parse_html(
        url: str,
        domain: str,
        html: str,
        status_code: int,
        latency_ms: float,
    ) -> FetchedPage:
        """Parse raw HTML with BeautifulSoup, stripping scripts and boilerplate."""
        soup = BeautifulSoup(html, "html.parser")

        # Strip uninformative elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        else:
            title = f"Document from {domain}"

        # Extract published date if present in meta tags
        published_date = None
        date_meta = soup.find(
            "meta",
            attrs={"property": re.compile(r"article:published_time|published_time|date", re.I)},
        ) or soup.find(
            "meta",
            attrs={"name": re.compile(r"pubdate|publishdate|date", re.I)},
        )
        if date_meta and date_meta.get("content"):
            published_date = date_meta["content"].strip()

        # Clean text
        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        return FetchedPage(
            url=url,
            title=title,
            publisher=domain,
            published_date=published_date,
            text=cleaned_text,
            status_code=status_code,
            method_used="httpx_bs4",
            latency_ms=round(latency_ms, 2),
            success=True,
            source_status=SourceStatus.FETCHED,
        )
