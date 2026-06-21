from unittest.mock import MagicMock

import pytest

from graph.nodes.rewrite_node import rewrite_node, rewrite_query
from graph.state import GraphState


def test_rewrite_query_no_llm_returns_string():
    result = rewrite_query("What is RAG?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_query_no_llm_differs_from_original():
    original = "What is RAG?"
    result = rewrite_query(original)
    assert result != original


def test_rewrite_query_no_llm_with_reason():
    result = rewrite_query("attention mechanism", reason="too vague")
    assert isinstance(result, str)


def test_rewrite_query_with_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Rewritten: attention mechanism in transformers")
    result = rewrite_query("attention", llm=llm)
    llm.invoke.assert_called_once()
    assert "Rewritten" in result


def test_rewrite_query_llm_error_falls_back():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM down")
    result = rewrite_query("What is RAG?", llm=llm)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_node_updates_query():
    state: GraphState = {"query": "What is attention?", "chunks": []}
    result = rewrite_node(state, llm=None)
    assert result["query"] != "What is attention?"


def test_rewrite_node_sets_rewritten_query():
    state: GraphState = {"query": "What is attention?", "chunks": []}
    result = rewrite_node(state, llm=None)
    assert "rewritten_query" in result
    assert isinstance(result["rewritten_query"], str)


def test_rewrite_node_increments_rewrite_count():
    state: GraphState = {"query": "test query", "rewrite_count": 1}
    result = rewrite_node(state, llm=None)
    assert result["rewrite_count"] == 2


def test_rewrite_node_initializes_rewrite_count():
    state: GraphState = {"query": "test query"}
    result = rewrite_node(state, llm=None)
    assert result["rewrite_count"] == 1


def test_rewrite_node_uses_grade_reasoning():
    state: GraphState = {
        "query": "What is RAG?",
        "grade_result": {
            "relevant": False,
            "confidence": 0.3,
            "reasoning": "chunks are about image recognition, not text retrieval",
        },
    }
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="RAG retrieval augmented generation survey")
    result = rewrite_node(state, llm=llm)
    prompt_arg = llm.invoke.call_args[0][0]
    assert "image recognition" in prompt_arg


def test_rewrite_node_query_and_rewritten_query_match():
    state: GraphState = {"query": "test"}
    result = rewrite_node(state, llm=None)
    assert result["query"] == result["rewritten_query"]
