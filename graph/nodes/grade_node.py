from __future__ import annotations

import json
import logging
import re

from graph.state import GraphState, GradeResult

logger = logging.getLogger(__name__)

_GRADE_PROMPT = """\
You are a strict relevance grader. Given a question and retrieved document chunks, \
score how well the chunks answer the SPECIFIC question asked.

Question: {query}

Retrieved chunks:
{chunks}

Scoring guide for confidence (be discriminating — most retrievals are imperfect):
  1.0 — chunks directly and completely answer the question with specific facts
  0.8 — chunks contain most of the needed information but miss some details
  0.6 — chunks are topically related but don't directly address the question
  0.4 — chunks are only tangentially relevant; key information is missing
  0.2 — chunks are about the same broad topic but don't help answer this question
  0.0 — chunks are off-topic or irrelevant

Set relevant=false if the chunks would NOT allow someone to answer the question correctly.

Reply with ONLY a JSON object — no markdown, no explanation outside the JSON:
{{"relevant": true, "confidence": 0.73, "reasoning": "one sentence explaining the score"}}
"""


def _extract_json(text: str) -> dict:
    """Strip <think> tags and extract the first JSON object from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON found in: {text[:200]!r}")


def grade_node(state: GraphState, llm: object | None = None) -> GraphState:
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        state["grade_result"] = GradeResult(
            relevant=False, confidence=1.0, reasoning="No chunks retrieved.",
        )
        return state

    if llm is None:
        state["grade_result"] = GradeResult(
            relevant=True, confidence=0.65,
            reasoning="Fallback grading: chunks were retrieved, assumed moderately relevant.",
        )
        return state

    chunk_texts = "\n\n".join(
        f"[{i+1}] {c.get('text', '')[:300]}" for i, c in enumerate(chunks[:5])
    )
    prompt = _GRADE_PROMPT.format(query=query, chunks=chunk_texts)

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
        state["grade_result"] = GradeResult(
            relevant=True, confidence=0.5,
            reasoning="Grading skipped — chunks assumed relevant."
        )

    return state


def _default_llm():
    from app.llm_factory import get_fast_llm
    return get_fast_llm()
