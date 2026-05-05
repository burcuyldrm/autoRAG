import pytest
from retrieval.bm25_retriever import BM25Retriever, Chunk


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", text="neural networks deep learning classification", metadata={"page": 1}),
        Chunk(id="c2", text="retrieval augmented generation language models", metadata={"page": 2}),
        Chunk(id="c3", text="transformer attention mechanism self attention", metadata={"page": 3}),
        Chunk(id="c4", text="graph neural network node classification", metadata={"page": 4}),
        Chunk(id="c5", text="knowledge graph embedding entity relation", metadata={"page": 5}),
    ]


def test_retrieve_returns_ranked_results():
    retriever = BM25Retriever(_make_chunks())
    results = retriever.retrieve("deep learning neural", k=3)
    assert len(results) == 3
    assert results[0]["chunk"]["id"] == "c1"


def test_retrieve_top_k_limit():
    retriever = BM25Retriever(_make_chunks())
    results = retriever.retrieve("language model", k=2)
    assert len(results) == 2


def test_retrieve_scores_descending():
    retriever = BM25Retriever(_make_chunks())
    results = retriever.retrieve("retrieval augmented generation", k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_index():
    retriever = BM25Retriever()
    results = retriever.retrieve("query")
    assert results == []


def test_retrieve_k_larger_than_corpus():
    retriever = BM25Retriever(_make_chunks())
    results = retriever.retrieve("neural", k=100)
    assert len(results) == len(_make_chunks())


def test_index_reindex():
    retriever = BM25Retriever(_make_chunks())
    new_chunks = [Chunk(id="new", text="brand new document about RAG", metadata={})]
    retriever.index(new_chunks)
    results = retriever.retrieve("RAG document", k=1)
    assert results[0]["chunk"]["id"] == "new"


def test_result_structure():
    retriever = BM25Retriever(_make_chunks())
    results = retriever.retrieve("knowledge graph", k=1)
    r = results[0]
    assert "chunk" in r
    assert "score" in r
    assert isinstance(r["score"], float)
    assert "id" in r["chunk"]
    assert "text" in r["chunk"]
    assert "metadata" in r["chunk"]
