from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0):
    """Returns the best available LLM: OpenAI first, Gemini fallback."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI model: %s", model)
        return ChatOpenAI(model=model, temperature=temperature)

    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info("Using Gemini model: %s", model)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )

    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in your .env file."
    )
