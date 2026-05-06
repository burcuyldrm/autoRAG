from __future__ import annotations

import json
import logging

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
            relevant=False,
            confidence=1.0,
            reasoning="No chunks retrieved.",
        )
        return state

    if llm is None:
        state["grade_result"] = GradeResult(
            relevant=True,
            confidence=0.65,
            reasoning="Fallback grading: chunks were retrieved, assumed moderately relevant.",
        )
        return state

    chunk_texts = "\n\n".join(
        f"[{i + 1}] {chunk.get('text', '')}" for i, chunk in enumerate(chunks[:5])
    )
    prompt = _GRADE_PROMPT.format(query=query, chunks=chunk_texts)

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(content.strip())

        state["grade_result"] = GradeResult(
            relevant=bool(result.get("relevant", False)),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=str(result.get("reasoning", "")),
        )

    except Exception as exc:
        logger.error("grade_node LLM error: %s", exc)
        state["grade_result"] = GradeResult(
            relevant=False,
            confidence=0.0,
            reasoning=f"Grading failed: {exc}",
        )

    return state