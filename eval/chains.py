"""
Chain factory functions used by CLI entry points.

All chains are backed by a local Ollama model — no API key required.
This makes every experiment fully reproducible on any machine with Ollama installed.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_standard_chain(
    vectorstore: Any = None,
    bm25_retriever: Any = None,
) -> Any:
    """Return a StandardRAGChain backed by a local Ollama LLM."""
    from app.llm_factory import get_llm
    from app.standard_rag import StandardRAGChain

    llm = get_llm(fast=False)
    return StandardRAGChain(vectorstore=vectorstore, bm25_retriever=bm25_retriever, llm=llm)


def get_autorag_chain(
    vectorstore: Any = None,
    bm25_retriever: Any = None,
    grade_threshold: float = 0.60,
    max_rewrites: int = 2,
) -> Any:
    """Return an AutoRAGChain backed by a local Ollama LLM."""
    from app.llm_factory import get_llm
    from graph.autorag_chain import AutoRAGChain

    llm = get_llm(fast=False)
    return AutoRAGChain(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        llm=llm,
        grade_threshold=grade_threshold,
        max_rewrites=max_rewrites,
    )


def get_retrieval_chain(
    retrieval_mode: str = "hybrid",
    vectorstore: Any = None,
    bm25_retriever: Any = None,
) -> Any:
    """Return a StandardRAGChain pinned to retrieval_mode, backed by Ollama."""
    from app.llm_factory import get_llm
    from app.standard_rag import StandardRAGChain
    from eval.benchmark_runner import RetrieverWrapper

    llm = get_llm(fast=False)
    chain = StandardRAGChain(vectorstore=vectorstore, bm25_retriever=bm25_retriever, llm=llm)
    return RetrieverWrapper(chain, retrieval_mode=retrieval_mode)
