from __future__ import annotations

import json
import logging
import re

from graph.state import GraphState, GradeResult

logger = logging.getLogger(__name__)

_GRADE_PROMPT = """\
You are a relevance grader. Given the user's question and retrieved document chunks, \
decide if the chunks contain information relevant to answering the question.

Question: {query}

Retrieved chunks:
{chunks}

Reply with ONLY a JSON object — no markdown, no explanation outside the JSON:
{{"relevant": true, "confidence": 0.85, "reasoning": "one sentence"}}
"""


def _extract_json(text: str) -> dict:
    """Strip <think> tags and extract the first JSON object from LLM output."""
    # Remove DeepSeek-R1 style <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON found in: {text[:200]!r}")


def grade_node(state: GraphState, llm: object | None = None) -> GraphState:
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        state["grade_result"] = GradeResult(
            relevant=False, confidence=1.0, reasoning="No chunks retrieved."
        )
        return state

    chunk_texts = "\n\n".join(
        f"[{i+1}] {c.get('text', '')[:300]}" for i, c in enumerate(chunks[:5])
    )
    prompt = _GRADE_PROMPT.format(query=query, chunks=chunk_texts)

    if llm is None:
        llm = _default_llm()

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        result = _extract_json(content)
        state["grade_result"] = GradeResult(
            relevant=bool(result.get("relevant", True)),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=str(result.get("reasoning", "")),
        )
    except Exception as exc:
        logger.error("grade_node error: %s", exc)
        # default to relevant so pipeline continues
        state["grade_result"] = GradeResult(
            relevant=True, confidence=0.5,
            reasoning="Grading skipped — chunks assumed relevant."
        )
    return state


def _default_llm():
    from app.llm_factory import get_llm
    return get_llm()
