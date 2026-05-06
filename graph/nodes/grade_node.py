from __future__ import annotations

import json
import logging
import os

from graph.state import GraphState, GradeResult

logger = logging.getLogger(__name__)

_GRADE_PROMPT = """\
You are a relevance grader. Given the user's question and a set of retrieved document chunks, \
decide whether the chunks contain information that is relevant to answering the question.

Question: {query}

Retrieved chunks:
{chunks}

Respond ONLY with valid JSON in this exact format:
{{"relevant": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence explanation"}}
"""


def grade_node(state: GraphState, llm: object | None = None) -> GraphState:
    """LangGraph node: grades retrieved chunks for relevance."""
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        state["grade_result"] = GradeResult(
            relevant=False, confidence=1.0, reasoning="No chunks retrieved."
        )
        return state

    chunk_texts = "\n\n".join(
        f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks[:5])
    )
    prompt = _GRADE_PROMPT.format(query=query, chunks=chunk_texts)

    if llm is None:
        llm = _default_llm()

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(content.strip())
        state["grade_result"] = GradeResult(
            relevant=bool(result.get("relevant", False)),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=str(result.get("reasoning", "")),
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("grade_node LLM error: %s", exc)
        state["grade_result"] = GradeResult(
            relevant=False, confidence=0.0, reasoning=f"Grading failed: {exc}"
        )
    return state


def _default_llm():
    from app.llm_factory import get_llm
    return get_llm()
