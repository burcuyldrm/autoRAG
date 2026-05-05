import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph_skeleton import build_graph, create_initial_state


def test_graph_runs():
    graph = build_graph()
    state = create_initial_state("What is AI?")

    result = graph.run(state)

    assert "answer" in result
    assert result["answer"] == "placeholder answer"