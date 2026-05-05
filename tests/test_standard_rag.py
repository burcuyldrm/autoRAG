import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.standard_rag import StandardRAGChain


def test_invoke_returns_answer():
    rag = StandardRAGChain()

    result = rag.invoke("What is AI?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_retrieve_then_generate():
    rag = StandardRAGChain()

    result = rag.invoke("AI")

    assert "AI" in result or len(result) > 0