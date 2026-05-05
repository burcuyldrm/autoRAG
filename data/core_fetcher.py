import os
import logging
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from data.base_fetcher import PaperFetcher, PaperMetadata

logger = logging.getLogger(__name__)

CORE_API_BASE = "https://api.core.ac.uk/v3"


class CoreFetcher(PaperFetcher):
    def __init__(self, api_key: Optional[str] = None, download_dir: str = "data/raw"):
        self.api_key = api_key or os.environ.get("CORE_API_KEY", "")
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        response = self._session.post(
            f"{CORE_API_BASE}/search/works",
            json={"q": query, "limit": max_results},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            pdf_url = item.get("downloadUrl") or item.get("links", [{}])[0].get("url", "")
            results.append(
                PaperMetadata(
                    paper_id=str(item.get("id", "")),
                    title=item.get("title", ""),
                    abstract=item.get("abstract", ""),
                    pdf_url=pdf_url,
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    published=item.get("publishedDate", "") or item.get("yearPublished", ""),
                    source="core",
                )
            )
        logger.info("CORE search '%s' returned %d results", query, len(results))
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def download_pdf(self, paper: PaperMetadata) -> Optional[str]:
        if not paper.pdf_url:
            logger.warning("No PDF URL for paper %s", paper.paper_id)
            return None
        dest_path = os.path.join(self.download_dir, f"core_{paper.paper_id}.pdf")
        if os.path.exists(dest_path):
            logger.info("PDF already exists: %s", dest_path)
            return dest_path
        response = self._session.get(paper.pdf_url, timeout=60, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded PDF: %s", dest_path)
        return dest_path


class UnifiedFetcher:
    """Fetches from both ArXiv and CORE, merges and deduplicates by title."""

    def __init__(self, arxiv_fetcher: PaperFetcher, core_fetcher: PaperFetcher):
        self._fetchers = [arxiv_fetcher, core_fetcher]

    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        seen_titles: set[str] = set()
        merged: list[PaperMetadata] = []
        per_source = max(max_results, 5)
        for fetcher in self._fetchers:
            try:
                papers = fetcher.search(query, max_results=per_source)
            except Exception as exc:
                logger.error("Fetcher %s failed: %s", type(fetcher).__name__, exc)
                papers = []
            for paper in papers:
                key = paper.title.lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    merged.append(paper)
        return merged[:max_results]
