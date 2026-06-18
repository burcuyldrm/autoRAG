"""
Chain factory functions used by CLI entry points.

When OPENAI_API_KEY is not set, all factories return a MockChain so that
the CLI commands produce valid JSON output without requiring an API key.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from eval.benchmark_runner import MockChain

logger = logging.getLogger(__name__)


def get_standard_chain(
    vectorstore: Any = None,
    bm25_retriever: Any = None,
) -> Any:
    """StandardRAGChain if API key is present, else MockChain."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set — using MockChain for standard_chain.")
        return MockChain(answer="[Mock] Standard RAG answer.", rewrite_count=0)
    from langchain_openai import ChatOpenAI
    from app.standard_rag import StandardRAGChain

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return StandardRAGChain(vectorstore=vectorstore, bm25_retriever=bm25_retriever, llm=llm)


def get_autorag_chain(
    vectorstore: Any = None,
    bm25_retriever: Any = None,
    grade_threshold: float = 0.60,
    max_rewrites: int = 2,
) -> Any:
    """AutoRAGChain if API key is present, else MockChain."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set — using MockChain for autorag_chain.")
        return MockChain(answer="[Mock] Auto-RAG answer.", rewrite_count=1)
    from langchain_openai import ChatOpenAI
    from graph.autorag_chain import AutoRAGChain

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
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
    """StandardRAGChain (pinned to retrieval_mode) if API key present, else MockChain."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set — using MockChain for retrieval_chain(%s).", retrieval_mode)
        return MockChain()
    from langchain_openai import ChatOpenAI
    from app.standard_rag import StandardRAGChain

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = StandardRAGChain(vectorstore=vectorstore, bm25_retriever=bm25_retriever, llm=llm)

    # Pin retrieval_mode via wrapper so run_chain_on_dataset() sees it
    from eval.benchmark_runner import RetrieverWrapper
    return RetrieverWrapper(chain, retrieval_mode=retrieval_mode)
