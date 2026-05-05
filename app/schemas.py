from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    graded_chunks: List[Dict[str, Any]]
    answer: str
    mode: str
    confidence: float
    iteration: int