import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_skeleton import route_node


def test_route_to_rewrite():
    state = {
        "confidence": 0.5,
        "iteration": 1
    }

    result = route_node(state)

    assert result == "rewrite"


def test_route_to_generate():
    state = {
        "confidence": 0.9,
        "iteration": 1
    }

    result = route_node(state)

    assert result == "generate"