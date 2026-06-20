"""Tests for graph.nodes.faithfulness_node."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from graph.nodes.faithfulness_node import check_faithfulness, faithfulness_node
from graph.state import GraphState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(faithful: bool, confidence: float = 0.9,
                   unsupported: list | None = None, reasoning: str = "ok") -> MagicMock:
    llm = MagicMock()
    payload = {
        "faithful": faithful,
        "confidence": confidence,
        "unsupported_claims": unsupported or [],
        "reasoning": reasoning,
    }
    llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    return llm


# ---------------------------------------------------------------------------
# check_faithfulness — fallback (no LLM)
# ---------------------------------------------------------------------------

def test_empty_answer_is_unfaithful():
    result = check_faithfulness("What is RAG?", "", ["RAG is a technique."])
    assert result["faithful"] is False
    assert result["confidence"] == 1.0


def test_no_contexts_is_unfaithful():
    result = check_faithfulness("What is RAG?", "RAG uses retrieval.", [])
    assert result["faithful"] is False
    assert result["confidence"] == 1.0


def test_heuristic_faithful_high_overlap():
    contexts = [
        "Retrieval Augmented Generation combines retrieval and language model generation "
        "to produce accurate grounded responses."
    ]
    answer = "RAG combines retrieval and language model generation for accurate responses."
    result = check_faithfulness("What is RAG?", answer, contexts)
    assert result["faithful"] is True
    assert 0.0 < result["confidence"] <= 1.0


def test_heuristic_unfaithful_low_overlap():
    contexts = ["The capital of France is Paris."]
    answer = "Quantum entanglement enables faster than light communication via photons."
    result = check_faithfulness("What is quantum entanglement?", answer, contexts)
    assert result["faithful"] is False


def test_heuristic_returns_required_keys():
    result = check_faithfulness("q", "answer text here", ["some context text"])
    assert "faithful" in result
    assert "confidence" in result
    assert "unsupported_claims" in result
    assert "reasoning" in result


def test_heuristic_confidence_in_range():
    result = check_faithfulness("q", "answer text about science research", ["science research methods"])
    assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# check_faithfulness — with LLM
# ---------------------------------------------------------------------------

def test_llm_faithful_result():
    llm = _llm_response(faithful=True, confidence=0.95)
    result = check_faithfulness("q", "answer", ["context"], llm=llm)
    assert result["faithful"] is True
    assert result["confidence"] == pytest.approx(0.95)


def test_llm_unfaithful_with_claims():
    llm = _llm_response(faithful=False, confidence=0.8,
                         unsupported=["claim A", "claim B"])
    result = check_faithfulness("q", "answer", ["context"], llm=llm)
    assert result["faithful"] is False
    assert result["unsupported_claims"] == ["claim A", "claim B"]


def test_llm_error_falls_back_to_heuristic():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("API unavailable")
    contexts = ["Retrieval augmented generation uses dense and sparse methods."]
    answer = "Dense and sparse retrieval methods are used in RAG."
    result = check_faithfulness("What is RAG?", answer, contexts, llm=llm)
    # Should not raise and should return a valid dict
    assert "faithful" in result
    assert "confidence" in result


def test_llm_invalid_json_falls_back():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="not json at all!")
    result = check_faithfulness("q", "answer", ["context"], llm=llm)
    assert "faithful" in result


# ---------------------------------------------------------------------------
# faithfulness_node
# ---------------------------------------------------------------------------

def test_faithfulness_node_writes_to_state():
    state: GraphState = {
        "query": "What is BM25?",
        "final_answer": "BM25 is a ranking function based on term frequency.",
        "chunks": [{"id": "c1", "text": "BM25 is a term frequency ranking function.", "metadata": {}}],
    }
    state = faithfulness_node(state)
    assert "faithfulness_result" in state
    assert "faithfulness_score" in state
    assert "unsupported_claims" in state


def test_faithfulness_node_score_nonnegative():
    state: GraphState = {
        "query": "q",
        "final_answer": "answer",
        "chunks": [{"id": "c1", "text": "context text", "metadata": {}}],
    }
    state = faithfulness_node(state)
    assert state["faithfulness_score"] >= 0.0


def test_faithfulness_node_empty_answer():
    state: GraphState = {
        "query": "q",
        "final_answer": "",
        "chunks": [{"id": "c1", "text": "context", "metadata": {}}],
    }
    state = faithfulness_node(state)
    assert state["faithfulness_result"]["faithful"] is False
    assert state["faithfulness_score"] == 0.0


def test_faithfulness_node_no_chunks():
    state: GraphState = {
        "query": "q",
        "final_answer": "some answer",
        "chunks": [],
    }
    state = faithfulness_node(state)
    assert state["faithfulness_result"]["faithful"] is False


def test_faithfulness_node_with_llm():
    llm = _llm_response(faithful=True, confidence=0.88)
    state: GraphState = {
        "query": "What is RAG?",
        "final_answer": "RAG combines retrieval and generation.",
        "chunks": [{"id": "c1", "text": "RAG context passage.", "metadata": {}}],
    }
    state = faithfulness_node(state, llm=llm)
    assert state["faithfulness_result"]["faithful"] is True
    assert state["faithfulness_score"] == pytest.approx(0.88)


def test_faithfulness_node_unsupported_claims_list():
    state: GraphState = {
        "query": "q",
        "final_answer": "answer",
        "chunks": [{"id": "c1", "text": "ctx", "metadata": {}}],
    }
    state = faithfulness_node(state)
    assert isinstance(state["unsupported_claims"], list)
