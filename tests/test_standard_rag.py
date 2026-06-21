from unittest.mock import MagicMock

import pytest

from app.standard_rag import StandardRAGChain


def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "metadata": {"title": "Test Paper"}}


def _search_result(cid: str, text: str, score: float = 0.9) -> dict:
    return {"chunk": _chunk(cid, text), "score": score}


def _ranked_chunk(cid: str, text: str, score: float = 5.0) -> dict:
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


def _make_llm(answer: str = "LLM answer") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=answer)
    return llm


def test_run_returns_all_required_keys():
    chain = StandardRAGChain(vectorstore=_make_vs())
    result = chain.run("What is RAG?", retrieval_mode="dense")
    assert {"answer", "contexts", "sources", "retriever_type", "top_k", "latency_seconds"}.issubset(
        result.keys()
    )


def test_retriever_type_echoes_mode():
    chain = StandardRAGChain(vectorstore=_make_vs())
    assert chain.run("q", retrieval_mode="dense")["retriever_type"] == "dense"
    assert chain.run("q", retrieval_mode="hybrid")["retriever_type"] == "hybrid"


def test_top_k_echoed_in_output():
    chain = StandardRAGChain(vectorstore=_make_vs())
    assert chain.run("q", top_k=3)["top_k"] == 3


def test_latency_is_nonnegative_float():
    chain = StandardRAGChain(vectorstore=_make_vs())
    lat = chain.run("q")["latency_seconds"]
    assert isinstance(lat, float)
    assert lat >= 0.0


def test_dense_mode_calls_vectorstore():
    vs = _make_vs([_search_result("c1", "Dense text")])
    chain = StandardRAGChain(vectorstore=vs)
    result = chain.run("query", retrieval_mode="dense", top_k=5)
    vs.search.assert_called_once_with("query", k=5)
    assert "Dense text" in result["contexts"]


def test_sources_text_truncated_to_200():
    long_text = "a" * 500
    vs = _make_vs([_search_result("c1", long_text)])
    chain = StandardRAGChain(vectorstore=vs)
    result = chain.run("q", retrieval_mode="dense")
    assert len(result["sources"][0]["text"]) <= 200


def test_sources_carry_metadata():
    vs = _make_vs([_search_result("c1", "text")])
    chain = StandardRAGChain(vectorstore=vs)
    result = chain.run("q", retrieval_mode="dense")
    assert result["sources"][0]["metadata"] == {"title": "Test Paper"}


def test_hybrid_calls_both_retrievers():
    vs = _make_vs()
    bm25 = _make_bm25()
    chain = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25)
    chain.run("query", retrieval_mode="hybrid")
    vs.search.assert_called_once()
    bm25.retrieve.assert_called_once()


def test_hybrid_rrf_deduplicates_shared_chunk():
    shared_chunk = _chunk("shared", "Shared text")
    vs = _make_vs([{"chunk": shared_chunk, "score": 0.9}])
    bm25 = _make_bm25([{"chunk": shared_chunk, "score": 3.0}])
    chain = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25)
    result = chain.run("q", retrieval_mode="hybrid")
    ids = [s["id"] for s in result["sources"]]
    assert ids.count("shared") == 1


def test_hybrid_falls_back_to_dense_without_bm25():
    vs = _make_vs()
    chain = StandardRAGChain(vectorstore=vs, bm25_retriever=None)
    result = chain.run("q", retrieval_mode="hybrid")
    vs.search.assert_called_once()
    assert len(result["contexts"]) > 0


def test_hybrid_falls_back_to_bm25_without_dense():
    bm25 = _make_bm25()
    chain = StandardRAGChain(vectorstore=None, bm25_retriever=bm25)
    result = chain.run("q", retrieval_mode="hybrid")
    bm25.retrieve.assert_called_once()
    assert len(result["contexts"]) > 0


def test_llm_is_invoked_when_provided():
    llm = _make_llm("Generated answer from LLM")
    chain = StandardRAGChain(vectorstore=_make_vs(), llm=llm)
    result = chain.run("What is attention?")
    llm.invoke.assert_called_once()
    assert result["answer"] == "Generated answer from LLM"


def test_fallback_answer_when_no_llm():
    chain = StandardRAGChain(vectorstore=_make_vs(), llm=None)
    result = chain.run("q")
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_fallback_answer_contains_context_text():
    vs = _make_vs([_search_result("c1", "Transformer architecture")])
    chain = StandardRAGChain(vectorstore=vs, llm=None)
    result = chain.run("q")
    assert "Transformer architecture" in result["answer"]


def test_empty_retrieval_gives_no_context_message():
    vs = _make_vs([])
    chain = StandardRAGChain(vectorstore=vs)
    result = chain.run("q")
    assert result["contexts"] == []
    assert result["sources"] == []
    assert "No relevant" in result["answer"]


def test_no_retriever_returns_gracefully():
    chain = StandardRAGChain()
    result = chain.run("q")
    assert result["answer"] == "No relevant context found for the given query."
    assert result["contexts"] == []
    assert result["sources"] == []
