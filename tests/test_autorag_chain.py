from unittest.mock import MagicMock, patch

import pytest

from graph.autorag_chain import AutoRAGChain, _rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "metadata": {"title": "Paper"}}


def _search_result(cid: str, text: str, score: float = 0.9) -> dict:
    return {"chunk": _chunk(cid, text), "score": score}


def _ranked_chunk(cid: str, text: str, score: float = 3.0) -> dict:
    return {"chunk": _chunk(cid, text), "score": score}


def _make_vs(results=None) -> MagicMock:
    vs = MagicMock()
    vs.search.return_value = results if results is not None else [
        _search_result("c1", "Dense result one"),
        _search_result("c2", "Dense result two"),
    ]
    return vs


def _make_bm25(results=None) -> MagicMock:
    bm25 = MagicMock()
    bm25.retrieve.return_value = results if results is not None else [
        _ranked_chunk("c2", "BM25 result two"),
        _ranked_chunk("c3", "BM25 result three"),
    ]
    return bm25


def _grade_llm(relevant: bool, confidence: float = 0.9) -> MagicMock:
    llm = MagicMock()
    import json
    llm.invoke.return_value = MagicMock(
        content=json.dumps({"relevant": relevant, "confidence": confidence, "reasoning": "test"})
    )
    return llm


def _generate_llm(answer: str = "Generated answer") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=answer)
    return llm


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_run_returns_all_required_keys():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("What is RAG?")
    expected = {
        "original_query", "final_query", "rewritten_query", "rewrite_count",
        "grade_confidence", "grade_relevant", "retrieval_mode", "top_k",
        "latency_seconds", "retrieved_scores", "answer", "contexts", "sources", "chunks",
    }
    assert expected.issubset(result.keys())


def test_original_query_preserved():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("My original question")
    assert result["original_query"] == "My original question"


def test_latency_is_nonnegative_float():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("q")
    assert isinstance(result["latency_seconds"], float)
    assert result["latency_seconds"] >= 0.0


def test_retrieval_mode_echoed():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("q", retrieval_mode="dense")
    assert result["retrieval_mode"] == "dense"


def test_top_k_echoed():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("q", top_k=3)
    assert result["top_k"] == 3


# ---------------------------------------------------------------------------
# Grade threshold — no rewrite when grade is accepted
# ---------------------------------------------------------------------------

def test_no_rewrite_when_grade_relevant():
    chain = AutoRAGChain(
        vectorstore=_make_vs(), llm=_grade_llm(relevant=True, confidence=0.95)
    )
    result = chain.run("q")
    assert result["rewrite_count"] == 0
    assert result["rewritten_query"] is None
    assert result["final_query"] == "q"


def test_no_rewrite_when_confidence_above_threshold():
    chain = AutoRAGChain(
        vectorstore=_make_vs(),
        llm=_grade_llm(relevant=False, confidence=0.70),
        grade_threshold=0.60,
    )
    result = chain.run("q")
    assert result["rewrite_count"] == 0


def test_rewrite_triggered_when_confidence_below_threshold():
    # grade returns low confidence → rewrite should happen
    import json
    llm = MagicMock()
    # First call: grade (low confidence) → invoke returns grade JSON
    # Subsequent calls (generate + possibly rewrite) return text
    responses = [
        MagicMock(content=json.dumps({"relevant": False, "confidence": 0.2, "reasoning": "not relevant"})),
        MagicMock(content=json.dumps({"relevant": True, "confidence": 0.9, "reasoning": "relevant now"})),
        MagicMock(content="Final answer"),
    ]
    llm.invoke.side_effect = responses
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=llm, grade_threshold=0.60, max_rewrites=1)
    result = chain.run("q")
    assert result["rewrite_count"] == 1
    assert result["rewritten_query"] is not None


# ---------------------------------------------------------------------------
# max_rewrites enforcement
# ---------------------------------------------------------------------------

def test_max_rewrites_respected():
    import json
    llm = MagicMock()
    # Always return low confidence → should stop after max_rewrites
    low_grade = json.dumps({"relevant": False, "confidence": 0.1, "reasoning": "bad"})
    llm.invoke.return_value = MagicMock(content=low_grade)

    chain = AutoRAGChain(
        vectorstore=_make_vs(), llm=llm, grade_threshold=0.60, max_rewrites=2
    )
    result = chain.run("q")
    assert result["rewrite_count"] <= 2


def test_max_rewrites_zero_means_no_rewrite():
    chain = AutoRAGChain(
        vectorstore=_make_vs(),
        llm=_grade_llm(relevant=False, confidence=0.1),
        max_rewrites=0,
    )
    result = chain.run("q")
    assert result["rewrite_count"] == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_contexts_populated_from_chunks():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("q", retrieval_mode="dense")
    assert isinstance(result["contexts"], list)
    assert all(isinstance(c, str) for c in result["contexts"])


def test_retrieved_scores_populated():
    chain = AutoRAGChain(vectorstore=_make_vs(), llm=_grade_llm(True))
    result = chain.run("q", retrieval_mode="dense")
    assert isinstance(result["retrieved_scores"], list)
    assert all(isinstance(s, float) for s in result["retrieved_scores"])


def test_no_retriever_returns_gracefully():
    chain = AutoRAGChain(llm=_grade_llm(False))
    result = chain.run("q")
    assert result["contexts"] == []
    assert result["retrieved_scores"] == []


# ---------------------------------------------------------------------------
# _rrf_fuse utility
# ---------------------------------------------------------------------------

def test_rrf_fuse_returns_correct_count():
    dense = [_search_result(f"d{i}", f"dense {i}") for i in range(3)]
    sparse = [_ranked_chunk(f"s{i}", f"sparse {i}") for i in range(3)]
    chunks, scores = _rrf_fuse(dense, sparse, top_k=4)
    assert len(chunks) == 4
    assert len(scores) == 4


def test_rrf_fuse_deduplicates():
    shared = _chunk("shared", "Shared text")
    dense = [{"chunk": shared, "score": 0.9}]
    sparse = [{"chunk": shared, "score": 3.0}]
    chunks, scores = _rrf_fuse(dense, sparse, top_k=5)
    ids = [c["id"] for c in chunks]
    assert ids.count("shared") == 1


def test_rrf_fuse_scores_descending():
    dense = [_search_result("d1", "text d1"), _search_result("d2", "text d2")]
    sparse = [_ranked_chunk("d1", "text d1")]  # d1 appears in both → higher score
    chunks, scores = _rrf_fuse(dense, sparse, top_k=2)
    assert chunks[0]["id"] == "d1"
    assert scores[0] >= scores[1]
