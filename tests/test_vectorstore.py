import os
import pytest

from vectordb.vectorstore import Chunk, FAISSVectorStore, get_vectorstore


def _make_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i}",
            text=f"This is test document number {i} about machine learning.",
            metadata={"paper_id": f"p{i}", "page": i},
        )
        for i in range(n)
    ]


# ---- FAISS (no external service required) ----

def test_faiss_ingest_and_count(tmp_path):
    store = FAISSVectorStore(index_dir=str(tmp_path))
    assert store.count() == 0
    store.ingest(_make_chunks(3))
    assert store.count() == 3


def test_faiss_search_returns_results(tmp_path):
    store = FAISSVectorStore(index_dir=str(tmp_path))
    store.ingest(_make_chunks(5))
    results = store.search("machine learning document", k=3)
    assert len(results) == 3
    for r in results:
        assert "text" in r["chunk"]
        assert "score" in r
        assert isinstance(r["score"], float)


def test_faiss_search_empty_store(tmp_path):
    store = FAISSVectorStore(index_dir=str(tmp_path))
    results = store.search("anything", k=5)
    assert results == []


def test_faiss_metadata_preserved(tmp_path):
    store = FAISSVectorStore(index_dir=str(tmp_path))
    chunks = _make_chunks(2)
    store.ingest(chunks)
    results = store.search("test document", k=1)
    assert results[0]["chunk"]["metadata"]["paper_id"].startswith("p")


def test_faiss_persistence(tmp_path):
    store1 = FAISSVectorStore(index_dir=str(tmp_path))
    store1.ingest(_make_chunks(4))
    store2 = FAISSVectorStore(index_dir=str(tmp_path))
    assert store2.count() == 4


def test_get_vectorstore_factory_faiss(tmp_path):
    store = get_vectorstore("faiss", index_dir=str(tmp_path))
    assert isinstance(store, FAISSVectorStore)


def test_get_vectorstore_invalid_backend():
    with pytest.raises(ValueError):
        get_vectorstore("invalid")
