from unittest.mock import MagicMock

from graph.nodes.grade_node import grade_node
from graph.state import GraphState


def _mock_llm(json_str: str) -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.content = json_str
    llm.invoke.return_value = response
    return llm


def test_grade_node_relevant():
    state: GraphState = {
        "query": "What is retrieval augmented generation?",
        "chunks": [{"text": "RAG combines retrieval with LLMs to produce accurate answers."}],
    }
    llm = _mock_llm('{"relevant": true, "confidence": 0.95, "reasoning": "Chunk directly answers the question."}')
    result = grade_node(state, llm=llm)
    assert result["grade_result"]["relevant"] is True
    assert result["grade_result"]["confidence"] == 0.95


def test_grade_node_not_relevant():
    state: GraphState = {
        "query": "What is the capital of France?",
        "chunks": [{"text": "This paper is about neural network training algorithms."}],
    }
    llm = _mock_llm('{"relevant": false, "confidence": 0.9, "reasoning": "Chunks do not answer the query."}')
    result = grade_node(state, llm=llm)
    assert result["grade_result"]["relevant"] is False


def test_grade_node_no_chunks():
    state: GraphState = {"query": "some query", "chunks": []}
    result = grade_node(state, llm=None)
    assert result["grade_result"]["relevant"] is False
    assert result["grade_result"]["confidence"] == 1.0


def test_grade_node_llm_error():
    state: GraphState = {
        "query": "question",
        "chunks": [{"text": "some text"}],
    }
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM unavailable")
    result = grade_node(state, llm=llm)
    assert result["grade_result"]["relevant"] is False
    assert "Grading failed" in result["grade_result"]["reasoning"]


def test_grade_node_invalid_json():
    state: GraphState = {
        "query": "question",
        "chunks": [{"text": "some text"}],
    }
    llm = _mock_llm("not valid json")
    result = grade_node(state, llm=llm)
    assert result["grade_result"]["relevant"] is False


def test_grade_node_state_updated():
    state: GraphState = {
        "query": "test",
        "chunks": [{"text": "relevant info"}],
    }
    llm = _mock_llm('{"relevant": true, "confidence": 0.8, "reasoning": "matches"}')
    result = grade_node(state, llm=llm)
    assert "grade_result" in result
    assert "relevant" in result["grade_result"]
    assert "confidence" in result["grade_result"]
    assert "reasoning" in result["grade_result"]
