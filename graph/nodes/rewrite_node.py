"""
Query Rewrite Node

Standalone function rewrite_query() + LangGraph node rewrite_node().
"""
from __future__ import annotations

import logging
import os

from graph.state import GraphState

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query optimizer for academic literature retrieval.

The following query did not return sufficiently relevant results:
Query: {query}
Reason results were insufficient: {reason}

Rewrite the query to be more specific and likely to match academic papers.
Return ONLY the rewritten query, nothing else.
"""


def rewrite_query(
    original_query: str,
    reason: str | None = None,
    llm=None,
) -> str:
    """
    Rewrite a query to improve retrieval quality.

    Parameters
    ----------
    original_query : str
        The query that failed to return relevant results.
    reason : str | None
        Grade node reasoning explaining why results were insufficient.
    llm : Any | None
        LangChain-compatible LLM. Falls back to rule-based rewrite if None.
    """
    if llm is None:
        return _rule_based_rewrite(original_query, reason)

    reason_text = reason or "results were not relevant"
    prompt = _REWRITE_PROMPT.format(query=original_query, reason=reason_text)
    try:
        response = llm.invoke(prompt)
        rewritten = response.content if hasattr(response, "content") else str(response)
        return rewritten.strip()
    except Exception as exc:
        logger.warning("rewrite_query LLM error: %s — using rule-based fallback", exc)
        return _rule_based_rewrite(original_query, reason)


def _rule_based_rewrite(query: str, reason: str | None) -> str:
    """Simple heuristic rewrite when no LLM is available."""
    query = query.strip().rstrip("?")
    return f"{query} academic research review"


def rewrite_node(state: GraphState, llm=None) -> GraphState:
    """LangGraph node: rewrites query based on grade feedback."""
    original = state.get("query", "")
    reason = None
    grade = state.get("grade_result")
    if grade:
        reason = grade.get("reasoning", "")

    new_query = rewrite_query(original, reason=reason, llm=llm)

    state["rewritten_query"] = new_query
    state["query"] = new_query
    state["rewrite_count"] = state.get("rewrite_count", 0) + 1
    logger.info("Query rewritten: %r → %r", original, new_query)
    return state
