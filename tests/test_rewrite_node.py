import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_skeleton import rewrite_node


def test_rewrite_adds_query():
    state = {
        "query": "What is AI?",
        "iteration": 0
    }

    new_state = rewrite_node(state)

    assert "rewritten_query" in new_state
    assert new_state["iteration"] == 1


def test_query_is_modified():
    state = {
        "query": "machine learning",
        "iteration": 0
    }

    new_state = rewrite_node(state)

    assert new_state["rewritten_query"] != state["query"]