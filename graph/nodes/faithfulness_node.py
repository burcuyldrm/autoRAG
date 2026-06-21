"""
Faithfulness Node

Evaluates whether the generated answer is supported by the retrieved contexts.

Provides:
  - check_faithfulness(question, answer, contexts, llm=None) -> dict
  - faithfulness_node(state, llm=None) -> GraphState
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from graph.state import GraphState

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """\
You are a faithfulness evaluator. Given a question, an answer, and a set of retrieved \
context passages, determine whether EVERY claim in the answer is supported by the contexts.

Question: {question}

Answer: {answer}

Context passages:
{contexts}

Respond ONLY with valid JSON in this exact format:
{{"faithful": true/false, "confidence": 0.0-1.0, "unsupported_claims": ["claim1", ...], \
"reasoning": "one sentence explanation"}}
"""


def check_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    llm: object | None = None,
) -> dict[str, Any]:
    """Check whether the answer is supported by the provided contexts.

    Parameters
    ----------
    question : str
        The original user question.
    answer : str
        The generated answer to evaluate.
    contexts : list[str]
        Retrieved context passages.
    llm : optional
        LangChain-compatible chat model. If None, a heuristic fallback is used.

    Returns
    -------
    dict with keys:
        - faithful (bool)
        - confidence (float, 0–1)
        - unsupported_claims (list[str])
        - reasoning (str)
    """
    # --- Fast-path: trivially unfaithful ---
    if not answer or not answer.strip():
        return {
            "faithful": False,
            "confidence": 1.0,
            "unsupported_claims": [],
            "reasoning": "Answer is empty.",
        }

    if not contexts:
        return {
            "faithful": False,
            "confidence": 1.0,
            "unsupported_claims": [],
            "reasoning": "No context provided; faithfulness cannot be determined.",
        }

    # --- LLM path ---
    if llm is not None:
        context_text = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts[:5])
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            contexts=context_text,
        )
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            result = json.loads(content.strip())
            return {
                "faithful": bool(result.get("faithful", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "unsupported_claims": list(result.get("unsupported_claims", [])),
                "reasoning": str(result.get("reasoning", "")),
            }
        except Exception as exc:
            logger.error("faithfulness LLM call failed: %s", exc)
            # Fall through to heuristic

    # --- Heuristic fallback ---
    return _heuristic_faithfulness(answer, contexts)


def _heuristic_faithfulness(answer: str, contexts: list[str]) -> dict[str, Any]:
    """Token-overlap heuristic when LLM is unavailable."""
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(
        tok
        for ctx in contexts
        for tok in re.findall(r"\w+", ctx.lower())
    )

    # Remove very common stop words
    stopwords = {
        "the", "a", "an", "is", "it", "in", "of", "and", "or", "to", "that",
        "this", "are", "was", "with", "for", "on", "as", "be", "by", "at",
        "from", "its", "not", "but", "which", "have", "has", "been", "they",
        "their", "what", "how", "when", "where", "who", "can", "do", "does",
    }
    meaningful_answer = answer_tokens - stopwords
    meaningful_context = context_tokens - stopwords

    if not meaningful_answer:
        return {
            "faithful": True,
            "confidence": 0.5,
            "unsupported_claims": [],
            "reasoning": "Heuristic: answer has no meaningful tokens to evaluate.",
        }

    overlap = meaningful_answer & meaningful_context
    overlap_ratio = len(overlap) / len(meaningful_answer)

    if overlap_ratio >= 0.5:
        faithful = True
        confidence = min(0.4 + overlap_ratio * 0.4, 0.8)
        reasoning = (
            f"Heuristic: {overlap_ratio:.0%} of answer tokens found in context "
            f"(threshold ≥ 50%)."
        )
    else:
        faithful = False
        confidence = min(0.4 + (1.0 - overlap_ratio) * 0.4, 0.8)
        reasoning = (
            f"Heuristic: only {overlap_ratio:.0%} of answer tokens found in context "
            f"(threshold ≥ 50%)."
        )

    return {
        "faithful": faithful,
        "confidence": round(confidence, 4),
        "unsupported_claims": [],
        "reasoning": reasoning,
    }


def faithfulness_node(
    state: GraphState,
    llm: object | None = None,
) -> GraphState:
    """LangGraph node: evaluate faithfulness of the generated answer.

    Reads ``query``, ``final_answer``, and ``chunks`` from state.
    Writes ``faithfulness_result``, ``faithfulness_score``, and
    ``unsupported_claims`` back to state.
    """
    question = state.get("query", "")
    answer = state.get("final_answer", "")
    chunks = state.get("chunks", [])
    contexts = [c.get("text", "") for c in chunks if c.get("text")]

    result = check_faithfulness(question, answer, contexts, llm=llm)

    state["faithfulness_result"] = result
    state["faithfulness_score"] = result["confidence"] if result["faithful"] else 0.0
    state["unsupported_claims"] = result.get("unsupported_claims", [])

    return state
