import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_skeleton import retrieve_node, create_initial_state


def test_retrieve_node_updates_state():
    state = create_initial_state("test query")

    result = retrieve_node(state)

    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0


def test_retrieve_node_hybrid_mode():
    state = create_initial_state("test query")
    state["mode"] = "hybrid"

    result = retrieve_node(state)

    assert len(result["retrieved_chunks"]) > 0