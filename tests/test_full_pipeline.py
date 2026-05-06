import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_skeleton import create_initial_state, retrieve_node


def dummy_generate_node(state):
    state["answer"] = "This is a test answer"
    state["sources"] = ["dummy_source"]
    return state


def run_pipeline(query: str):
    state = create_initial_state(query)

    state = retrieve_node(state)
    state = dummy_generate_node(state)

    return state


def test_pipeline_basic():
    state = run_pipeline("machine learning")

    assert state["answer"] is not None
    assert state["answer"] != ""
    assert state["sources"] != []


def test_pipeline_multiple_queries():
    queries = [
        "machine learning",
        "deep learning",
        "neural networks"
    ]

    for query in queries:
        state = run_pipeline(query)

        assert state["answer"] is not None
        assert state["sources"] != []


def test_pipeline_state_structure():
    state = run_pipeline("artificial intelligence")

    assert "query" in state
    assert "retrieved_chunks" in state
    assert "answer" in state
    assert "sources" in state