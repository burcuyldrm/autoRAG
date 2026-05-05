import os
import time
import logging
from typing import Optional

import arxiv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from data.base_fetcher import PaperFetcher, PaperMetadata

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_SECOND = 3
_last_request_times: list[float] = []


def _rate_limit() -> None:
    now = time.time()
    cutoff = now - 1.0
    _last_request_times[:] = [t for t in _last_request_times if t > cutoff]
    if len(_last_request_times) >= MAX_REQUESTS_PER_SECOND:
        sleep_for = 1.0 - (now - _last_request_times[0])
        if sleep_for > 0:
            time.sleep(sleep_for)
    _last_request_times.append(time.time())


class ArXivFetcher(PaperFetcher):
    def __init__(self, download_dir: str = "data/raw"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        self.client = arxiv.Client(num_retries=3, delay_seconds=2.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def search(self, query: str, max_results: int = 10) -> list[PaperMetadata]:
        _rate_limit()
        search = arxiv.Search(query=query, max_results=max_results)
        results = []
        for result in self.client.results(search):
            results.append(
                PaperMetadata(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    abstract=result.summary,
                    pdf_url=result.pdf_url,
                    authors=[a.name for a in result.authors],
                    published=result.published.isoformat() if result.published else "",
                    source="arxiv",
                )
            )
        logger.info("ArXiv search '%s' returned %d results", query, len(results))
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def download_pdf(self, paper: PaperMetadata) -> Optional[str]:
        _rate_limit()
        dest_path = os.path.join(self.download_dir, f"{paper.paper_id}.pdf")
        if os.path.exists(dest_path):
            logger.info("PDF already exists: %s", dest_path)
            return dest_path
        search = arxiv.Search(id_list=[paper.paper_id])
        result = next(self.client.results(search))
        result.download_pdf(dirpath=self.download_dir, filename=f"{paper.paper_id}.pdf")
        logger.info("Downloaded PDF: %s", dest_path)
        return dest_path
