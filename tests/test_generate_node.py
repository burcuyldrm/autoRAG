from unittest.mock import MagicMock

from graph.nodes.generate_node import generate_node
from graph.state import GraphState


def _mock_llm(answer: str) -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.content = answer
    llm.invoke.return_value = response
    return llm


def test_generate_node_produces_answer():
    state: GraphState = {
        "query": "What is RAG?",
        "chunks": [
            {"id": "c1", "text": "RAG stands for Retrieval Augmented Generation.", "metadata": {"page": 1}},
        ],
    }
    llm = _mock_llm("RAG stands for Retrieval Augmented Generation.")
    result = generate_node(state, llm=llm)
    assert result["final_answer"] == "RAG stands for Retrieval Augmented Generation."


def test_generate_node_populates_sources():
    state: GraphState = {
        "query": "Explain attention",
        "chunks": [
            {"id": "c1", "text": "Attention is all you need.", "metadata": {"paper_id": "1706.03762"}},
            {"id": "c2", "text": "Self-attention computes queries, keys, values.", "metadata": {}},
        ],
    }
    llm = _mock_llm("Attention is a mechanism...")
    result = generate_node(state, llm=llm)
    assert len(result["sources"]) == 2
    assert result["sources"][0]["id"] == "c1"


def test_generate_node_no_chunks():
    state: GraphState = {"query": "some query", "chunks": []}
    result = generate_node(state, llm=None)
    assert "No relevant information" in result["final_answer"]
    assert result["sources"] == []


def test_generate_node_llm_error():
    state: GraphState = {
        "query": "question",
        "chunks": [{"id": "c1", "text": "text", "metadata": {}}],
    }
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM down")
    result = generate_node(state, llm=llm)
    assert "Generation failed" in result["final_answer"]


def test_generate_node_sources_truncated():
    long_text = "x" * 500
    state: GraphState = {
        "query": "test",
        "chunks": [{"id": "c1", "text": long_text, "metadata": {}}],
    }
    llm = _mock_llm("answer")
    result = generate_node(state, llm=llm)
    assert len(result["sources"][0]["text"]) <= 200
