from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    graded_chunks: List[Dict[str, Any]]
    answer: str