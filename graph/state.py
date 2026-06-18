from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    original_query: str
    chunks: list[dict[str, Any]]
    grade_result: "GradeResult"
    rewritten_query: str
    rewrite_count: int
    retrieved_scores: list[float]
    final_answer: str
    sources: list[dict[str, Any]]
    iteration: int


class GradeResult(TypedDict):
    relevant: bool
    confidence: float
    reasoning: str
