from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict, total=False):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]