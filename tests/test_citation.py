from graph.citation import format_citation, format_sources


def test_format_citation_full():
    result = format_citation(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        page=5,
        url="https://arxiv.org/abs/1706.03762",
    )
    assert "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" in result
    assert "Vaswani, A." in result
    assert "Page 5" in result


def test_format_citation_no_page():
    result = format_citation(
        title="Test Paper",
        authors=["Author One"],
        page=None,
        url="http://example.com",
    )
    assert "Page" not in result
    assert "[Test Paper](http://example.com)" in result


def test_format_citation_many_authors():
    result = format_citation(
        title="Big Paper",
        authors=["A", "B", "C", "D", "E"],
        page=1,
        url="http://example.com",
    )
    assert "et al." in result


def test_format_sources_list():
    sources = [
        {
            "id": "s1",
            "text": "some text",
            "metadata": {
                "title": "My Paper",
                "authors": ["Alice"],
                "page": 3,
                "source_url": "http://arxiv.org/pdf/1234",
            },
        }
    ]
    citations = format_sources(sources)
    assert len(citations) == 1
    assert "My Paper" in citations[0]
    assert "Alice" in citations[0]
    assert "Page 3" in citations[0]


def test_format_sources_missing_metadata():
    sources = [{"id": "x1", "text": "text", "metadata": {}}]
    citations = format_sources(sources)
    assert len(citations) == 1
    assert "x1" in citations[0]


def test_format_sources_empty():
    assert format_sources([]) == []
