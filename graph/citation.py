from __future__ import annotations

from typing import Any


def format_citation(
    title: str,
    authors: list[str],
    page: int | None,
    url: str,
    source_file: str = "",
) -> str:
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    page_str   = f", s. {page}" if page is not None else ""
    file_str   = f" `{source_file}`" if source_file and not url else ""
    label      = f"{title}{page_str}{file_str}"
    if url:
        return f"[{label}]({url})" + (f" — {author_str}" if author_str else "")
    return f"**{label}**" + (f" — {author_str}" if author_str else "")


def format_sources(sources: list[dict[str, Any]]) -> list[str]:
    citations = []
    for source in sources:
        meta = source.get("metadata", {})
        title       = meta.get("title") or meta.get("paper_id") or source.get("id", "Kaynak")
        authors     = meta.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",")]
        page        = meta.get("page")
        url         = meta.get("source_url") or meta.get("pdf_url") or ""
        source_file = meta.get("source_file", "")
        citations.append(format_citation(
            title=str(title), authors=authors,
            page=page, url=url, source_file=source_file,
        ))
    return citations
