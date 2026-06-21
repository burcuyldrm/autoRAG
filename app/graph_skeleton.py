from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    graded_chunks: List[Dict[str, Any]]
    answer: str
    grade_result: Dict[str, Any]
    retry_count: int


def create_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "grade_result": {},
        "retry_count": 0,
    }


def retrieve_node(state: GraphState) -> GraphState:
    query = state.get("rewritten_query") or state.get("query", "")

    state["retrieved_chunks"] = [
        {
            "id": "mock_chunk_1",
            "text": f"Retrieved context for query: {query}",
            "metadata": {"source": "mock"},
        }
    ]

    return state


def grade_node(state: GraphState) -> GraphState:
    chunks = state.get("retrieved_chunks", [])
    is_relevant = len(chunks) > 0

    state["grade_result"] = {
        "relevant": is_relevant,
        "confidence": 0.8 if is_relevant else 0.0,
        "reasoning": "Chunks found." if is_relevant else "No chunks found.",
    }

    state["graded_chunks"] = chunks if is_relevant else []
    return state


def rewrite_node(state: GraphState) -> GraphState:
    query = state.get("query", "")

    state["rewritten_query"] = f"{query} academic paper context"
    state["iteration"] = state.get("iteration", 0) + 1
    state["retry_count"] = state.get("retry_count", 0) + 1

    return state


def generate_node(state: GraphState) -> GraphState:
    state["answer"] = "placeholder answer"
    return state

def route_node(state: GraphState) -> str:
    if state.get("confidence", 0) >= 0.7:
        return "generate"
    
    return "rewrite"


def build_graph():
    class SimpleGraph:
        def run(self, state: GraphState) -> GraphState:
            state = retrieve_node(state)
            state = grade_node(state)

            route = route_node(state)

            if route == "rewrite":
                state = rewrite_node(state)
                state = retrieve_node(state)
                state = grade_node(state)

            state = generate_node(state)
            return state

    return SimpleGraph()