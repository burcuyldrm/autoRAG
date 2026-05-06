import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from app.arxiv_fetcher import fetch_papers


def make_mock_result():
    result = MagicMock()
    result.title = "Test Paper"
    result.summary = "Test abstract"
    result.entry_id = "http://arxiv.org/abs/1234.5678"
    result.pdf_url = "http://arxiv.org/pdf/1234.5678"
    return result


@patch("app.arxiv_fetcher.time.sleep")
@patch("app.arxiv_fetcher.arxiv.Search")
def test_fetch_papers_success(mock_search, mock_sleep):
    mock_instance = MagicMock()
    mock_instance.results.return_value = [make_mock_result()]
    mock_search.return_value = mock_instance

    papers = fetch_papers("machine learning", max_results=1)

    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper"


@patch("app.arxiv_fetcher.time.sleep")
@patch("app.arxiv_fetcher.arxiv.Search")
def test_fetch_papers_empty(mock_search, mock_sleep):
    mock_instance = MagicMock()
    mock_instance.results.return_value = []
    mock_search.return_value = mock_instance

    papers = fetch_papers("nothing", max_results=1)

    assert papers == []


@patch("app.arxiv_fetcher.time.sleep")
@patch("app.arxiv_fetcher.arxiv.Search")
def test_fetch_papers_output_fields(mock_search, mock_sleep):
    mock_instance = MagicMock()
    mock_instance.results.return_value = [make_mock_result()]
    mock_search.return_value = mock_instance

    papers = fetch_papers("ai", max_results=1)

    assert "title" in papers[0]
    assert "abstract" in papers[0]
    assert "arxiv_id" in papers[0]
    assert "pdf_url" in papers[0]


@patch("app.arxiv_fetcher.time.sleep")
@patch("app.arxiv_fetcher.arxiv.Search")
def test_fetch_papers_failure_returns_empty(mock_search, mock_sleep):
    mock_search.side_effect = Exception("API error")

    papers = fetch_papers("ai", max_results=1, retries=1)

    assert papers == []


@patch("app.arxiv_fetcher.time.sleep")
@patch("app.arxiv_fetcher.arxiv.Search")
def test_fetch_papers_uses_query_and_max_results(mock_search, mock_sleep):
    mock_instance = MagicMock()
    mock_instance.results.return_value = []
    mock_search.return_value = mock_instance

    fetch_papers("deep learning", max_results=3)

    mock_search.assert_called_once()
    kwargs = mock_search.call_args.kwargs

    assert kwargs["query"] == "deep learning"
    assert kwargs["max_results"] == 3