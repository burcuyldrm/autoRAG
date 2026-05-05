from app.schemas import GraphState


def create_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
    }


def retrieve_node(state: GraphState) -> GraphState:
    state["retrieved_chunks"] = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {
                "paper_id": "dummy-paper",
                "page": 1,
                "source_url": "https://example.com/paper"
            }
        }
    ]

    return state