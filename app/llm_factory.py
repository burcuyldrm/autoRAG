"""
LLM factory — returns a locally-running Ollama model.

All LLM-dependent components (grade, generate, rewrite, faithfulness, RAGAS)
use this factory so that the entire pipeline runs without any API key.

Environment variables (all optional):
    OLLAMA_MODEL       — model for generation          (default: deepseek-r1:7b)
    OLLAMA_FAST_MODEL  — model for grading/rewriting   (default: qwen2.5:3b)
    OLLAMA_BASE_URL    — Ollama server URL             (default: http://localhost:11434)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")
_FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:3b")


def get_llm(fast: bool = False):
    """Return a ChatOllama instance.

    Parameters
    ----------
    fast : bool
        If True, use OLLAMA_FAST_MODEL (smaller, quicker).
        If False, use OLLAMA_MODEL (larger, better quality).
    """
    from langchain_ollama import ChatOllama

    model = _FAST_MODEL if fast else _MODEL
    logger.info("LLM factory: ChatOllama(model=%r, base_url=%r)", model, _BASE_URL)
    return ChatOllama(model=model, base_url=_BASE_URL, temperature=0)


def get_ragas_llm(fast: bool = True):
    """Return a RAGAS-compatible LLM wrapper backed by Ollama."""
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(get_llm(fast=fast))


def get_ragas_embeddings():
    """Return a RAGAS-compatible embedding wrapper backed by sentence-transformers."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf)
