from __future__ import annotations

import logging
import re
from typing import Any

from graph.state import GraphState

logger = logging.getLogger(__name__)

_GENERATE_PROMPT = """\
You are a scientific research assistant. Using ONLY the provided document chunks below, \
answer the user's question accurately and concisely. Cite chunk numbers like [1], [2].

Question: {query}

Document chunks:
{chunks}

Answer:"""


def _clean_response(text: str) -> str:
    """Remove DeepSeek-R1 <think> blocks from the final answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_node(state: GraphState, llm: object | None = None) -> GraphState:
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        state["final_answer"] = "İlgili bilgi bulunamadı."
        state["sources"] = []
        return state

    chunk_texts = "\n\n".join(
        f"[{i+1}] (Kaynak: {c.get('metadata', {}).get('paper_id', '?')}, "
        f"sayfa {c.get('metadata', {}).get('page', '?')})\n{c.get('text', '')}"
        for i, c in enumerate(chunks)
    )
    prompt = _GENERATE_PROMPT.format(query=query, chunks=chunk_texts)

    if llm is None:
        llm = _default_llm()

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        state["final_answer"] = _clean_response(raw)
    except Exception as exc:
        logger.error("generate_node error: %s", exc)
        state["final_answer"] = f"Cevap üretilemedi: {exc}"

    state["sources"] = [
        {
            "id": c.get("id", f"source-{i}"),
            "text": c.get("text", "")[:300],
            "metadata": c.get("metadata", {}),
        }
        for i, c in enumerate(chunks)
    ]
    return state


def _default_llm():
    from app.llm_factory import get_llm
    return get_llm()
