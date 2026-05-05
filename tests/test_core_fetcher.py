from unittest.mock import MagicMock, patch
import pytest

from data.base_fetcher import PaperMetadata
from data.core_fetcher import CoreFetcher, UnifiedFetcher


def _mock_response(json_data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


SAMPLE_CORE_RESPONSE = {
    "results": [
        {
            "id": 12345,
            "title": "Deep Learning Survey",
            "abstract": "A survey of deep learning.",
            "downloadUrl": "http://core.ac.uk/download/12345.pdf",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "publishedDate": "2023-03-15",
        }
    ]
}


def test_search_returns_metadata():
    fetcher = CoreFetcher(api_key="test-key", download_dir="/tmp/test_raw")
    with patch.object(fetcher._session, "post", return_value=_mock_response(SAMPLE_CORE_RESPONSE)):
        papers = fetcher.search("deep learning", max_results=1)

    assert len(papers) == 1
    paper = papers[0]
    assert isinstance(paper, PaperMetadata)
    assert paper.paper_id == "12345"
    assert paper.title == "Deep Learning Survey"
    assert paper.source == "core"
    assert paper.pdf_url == "http://core.ac.uk/download/12345.pdf"


def test_search_empty_results():
    fetcher = CoreFetcher(api_key="test-key", download_dir="/tmp/test_raw")
    with patch.object(fetcher._session, "post", return_value=_mock_response({"results": []})):
        papers = fetcher.search("nonexistent query xyz", max_results=1)
    assert papers == []


def test_download_pdf_skips_existing(tmp_path):
    fetcher = CoreFetcher(api_key="test-key", download_dir=str(tmp_path))
    paper = PaperMetadata(
        paper_id="99999",
        title="T",
        abstract="A",
        pdf_url="http://example.com/pdf",
        authors=[],
        published="",
        source="core",
    )
    existing = tmp_path / "core_99999.pdf"
    existing.write_bytes(b"fake pdf")
    path = fetcher.download_pdf(paper)
    assert path == str(existing)


def test_download_pdf_no_url_returns_none(tmp_path):
    fetcher = CoreFetcher(api_key="test-key", download_dir=str(tmp_path))
    paper = PaperMetadata(
        paper_id="00000", title="T", abstract="A", pdf_url="",
        authors=[], published="", source="core",
    )
    assert fetcher.download_pdf(paper) is None


def test_unified_fetcher_merges_and_deduplicates():
    fetcher1 = MagicMock()
    fetcher2 = MagicMock()

    shared = PaperMetadata(
        paper_id="1", title="Shared Paper", abstract="", pdf_url="",
        authors=[], published="", source="arxiv",
    )
    unique = PaperMetadata(
        paper_id="2", title="Unique Paper", abstract="", pdf_url="",
        authors=[], published="", source="core",
    )
    fetcher1.search.return_value = [shared]
    fetcher2.search.return_value = [shared, unique]  # shared appears in both

    unified = UnifiedFetcher(arxiv_fetcher=fetcher1, core_fetcher=fetcher2)
    results = unified.search("query", max_results=10)

    titles = [p.title for p in results]
    assert titles.count("Shared Paper") == 1
    assert "Unique Paper" in titles


def test_unified_fetcher_handles_fetcher_error():
    fetcher1 = MagicMock()
    fetcher2 = MagicMock()
    fetcher1.search.side_effect = Exception("network error")
    fetcher2.search.return_value = [
        PaperMetadata(
            paper_id="1", title="Paper", abstract="", pdf_url="",
            authors=[], published="", source="core",
        )
    ]
    unified = UnifiedFetcher(arxiv_fetcher=fetcher1, core_fetcher=fetcher2)
    results = unified.search("query")
    assert len(results) == 1
