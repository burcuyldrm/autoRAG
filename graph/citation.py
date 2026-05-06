from __future__ import annotations

from typing import Any


def format_citation(
    title: str,
    authors: list[str],
    page: int | None,
    url: str,
) -> str:
    """Returns formatted citation."""

    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += " et al."

    page_str = f", Page {page}" if page is not None else ""

    if url:
        return f"[{title}]({url}) — {author_str}{page_str}"

    return f"{title} — {author_str}{page_str}"

def format_sources(sources: list[dict[str, Any]]) -> list[str]:
    citations = []
    for source in sources:
        meta = source.get("metadata", {})
        title = meta.get("title", source.get("id", "Unknown"))
        authors = meta.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        page = meta.get("page")
        url = meta.get("source_url", meta.get("pdf_url", ""))
        citations.append(format_citation(title=title, authors=authors, page=page, url=url))
    return citations
