from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Priority order: OpenAI → Gemini → Ollama (local, no API key)
# Set OLLAMA_MODEL in .env to activate local DeepSeek/any Ollama model.


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

    # Local Ollama — no API key needed
    ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    from langchain_ollama import ChatOllama
    logger.info("Using Ollama (local): %s @ %s", ollama_model, ollama_base)
    return ChatOllama(model=ollama_model, base_url=ollama_base, temperature=temperature)
