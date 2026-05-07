from __future__ import annotations

import logging
import re

from graph.state import GraphState

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
The following search query returned documents that were NOT relevant to the user's question.
Rewrite the query to be more specific, use different keywords, and improve recall.

Original query: {query}
Why it failed: {reasoning}

Reply with ONLY the improved query text — no quotes, no explanation.
"""


def rewrite_node(state: GraphState, llm=None) -> GraphState:
    query    = state.get("query", "")
    grade    = state.get("grade_result", {})
    reasoning = grade.get("reasoning", "") if isinstance(grade, dict) else ""

    if llm is None:
        llm = _default_llm()

    try:
        prompt   = _REWRITE_PROMPT.format(query=query, reasoning=reasoning)
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        content  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        rewritten = content.strip('"').strip("'").strip()
        state["rewritten_query"] = rewritten if rewritten else query
    except Exception as exc:
        logger.error("rewrite_node error: %s", exc)
        state["rewritten_query"] = query  # fallback — keep original

    state["iteration"] = state.get("iteration", 0) + 1
    return state


def _default_llm():
    from app.llm_factory import get_fast_llm
    return get_fast_llm()
