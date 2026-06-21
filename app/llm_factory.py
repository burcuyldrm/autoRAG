from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Priority order: OpenAI → Gemini → Ollama (local, no API key)
# Set OLLAMA_MODEL in .env to activate local DeepSeek/any Ollama model.
# Set OLLAMA_FAST_MODEL for lightweight grade/rewrite nodes (default: qwen2.5:3b).


def get_llm(temperature: float = 0):
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI: %s", model)
        return ChatOpenAI(model=model, temperature=temperature)

    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info("Using Gemini: %s", model)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )

    ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    from langchain_ollama import ChatOllama
    logger.info("Using Ollama (local): %s @ %s", ollama_model, ollama_base)
    return ChatOllama(model=ollama_model, base_url=ollama_base, temperature=temperature)


def get_fast_llm(temperature: float = 0):
    """Lightweight LLM for grade/rewrite nodes."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return get_llm(temperature)

    fast_model = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:3b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    from langchain_ollama import ChatOllama
    logger.info("Using fast Ollama (local): %s @ %s", fast_model, ollama_base)
    return ChatOllama(
        model=fast_model,
        base_url=ollama_base,
        temperature=temperature,
        num_predict=200,
    )


def get_ragas_llm(fast: bool = True):
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(get_fast_llm() if fast else get_llm())


def get_ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf)
