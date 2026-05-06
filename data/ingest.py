"""
Ingest chunks from JSON files into ChromaDB.

Usage:
    python -m data.ingest                          # loads data/chunks/sample_chunks.json
    python -m data.ingest --chunks path/to/file.json
    python -m data.ingest --arxiv "RAG retrieval" --max 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from vectordb.vectorstore import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNKS = os.path.join(os.path.dirname(__file__), "chunks", "sample_chunks.json")


def ingest_from_json(path: str, backend: str = "chroma") -> int:
    with open(path) as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list of chunks, got {type(chunks)}")
    store = get_vectorstore(backend=backend)
    existing = store.count()
    store.ingest(chunks)
    added = store.count() - existing
    logger.info("Ingested %d chunks (store total: %d)", added, store.count())
    return store.count()


def ingest_from_arxiv(query: str, max_results: int = 5, backend: str = "chroma") -> int:
    from data.arxiv_fetcher import ArXivFetcher
    from app.pdf_extractor import extract_pages
    from app.chunker import create_chunks

    fetcher = ArXivFetcher()
    papers = fetcher.search(query, max_results=max_results)
    logger.info("Found %d papers for query: %r", len(papers), query)

    all_chunks = []
    for paper in papers:
        try:
            pdf_path = fetcher.download_pdf(paper)
            if not pdf_path:
                continue
            pages = extract_pages(pdf_path, paper_id=paper.paper_id)
            chunks = create_chunks(pages)
            for chunk in chunks:
                chunk["metadata"]["title"] = paper.title
                chunk["metadata"]["authors"] = ", ".join(paper.authors[:3])
                chunk["metadata"]["published"] = paper.published
            all_chunks.extend(chunks)
            logger.info("  %s: %d chunks", paper.title[:60], len(chunks))
        except Exception as exc:
            logger.warning("  Skipping %s: %s", paper.paper_id, exc)

    if all_chunks:
        store = get_vectorstore(backend=backend)
        store.ingest(all_chunks)
        logger.info("Total chunks in store: %d", store.count())
        return store.count()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="Path to chunks JSON file")
    parser.add_argument("--arxiv", default=None, help="ArXiv search query")
    parser.add_argument("--max", type=int, default=5, help="Max ArXiv results")
    parser.add_argument("--backend", default="chroma", choices=["chroma", "faiss"])
    args = parser.parse_args()

    if args.arxiv:
        count = ingest_from_arxiv(args.arxiv, args.max, args.backend)
    else:
        count = ingest_from_json(args.chunks, args.backend)

    print(f"Vector store now contains {count} chunks.")


if __name__ == "__main__":
    main()
