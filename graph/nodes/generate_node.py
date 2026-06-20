from __future__ import annotations

import logging
import os
from typing import Any

from graph.state import GraphState

logger = logging.getLogger(__name__)

_GENERATE_PROMPT = """\
You are a scientific research assistant. Using ONLY the provided document chunks, \
answer the user's question accurately and concisely. Cite the source for each claim.

Question: {query}

Document chunks:
{chunks}

Provide a clear, factual answer based solely on the above information.
"""


def generate_node(state: GraphState, llm: object | None = None) -> GraphState:
    """LangGraph node: generates final answer from graded chunks."""
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        state["final_answer"] = "No relevant information found to answer your question."
        state["sources"] = []
        return state

    chunk_texts = "\n\n".join(
        f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks)
    )
    prompt = _GENERATE_PROMPT.format(query=query, chunks=chunk_texts)

    if llm is None:
        fallback_parts = []

        for chunk in chunks[:3]:
            text = chunk.get("text", "").strip()
            
            if text:
                sentences = text.split(". ")
                fallback_parts.append(". ".join(sentences[:2]))

        fallback_answer = "\n\n".join(fallback_parts).strip()

        if not fallback_answer:
            fallback_answer = "Relevant chunks were retrieved but no answer could be generated."
        
        state["final_answer"] = fallback_answer
        state["sources"] = [
            {
                "id": c.get("id", f"source-{i}"),
                "text": c.get("text", "")[:200],
                "metadata": c.get("metadata", {}),
            }
            for i, c in enumerate(chunks)
        ]
        return state      


    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        state["final_answer"] = answer.strip()
    except Exception as exc:
        logger.error("generate_node LLM error: %s", exc)
        state["final_answer"] = f"Generation failed: {exc}"

    state["sources"] = [
        {
            "id": c.get("id", f"source-{i}"),
            "text": c.get("text", "")[:200],
            "metadata": c.get("metadata", {}),
        }
        for i, c in enumerate(chunks)
    ]
    return state


def _default_llm():
    from app.llm_factory import get_llm

    return get_llm(fast=False)
