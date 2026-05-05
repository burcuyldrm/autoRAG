from __future__ import annotations

import os
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Chunk(TypedDict):
    id: str
    text: str
    metadata: dict[str, Any]


class SearchResult(TypedDict):
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def ingest(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str = "vectordb/chroma.db", collection_name: str = "papers"):
        import chromadb
        from chromadb.utils import embedding_functions

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaVectorStore initialised (collection=%s)", collection_name)

    def ingest(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Ingested %d chunks into ChromaDB", len(chunks))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, max(self._collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        output: list[SearchResult] = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            output.append(
                SearchResult(
                    chunk=Chunk(id=cid, text=doc, metadata=meta),
                    score=1.0 - dist,
                )
            )
        return output

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FAISSVectorStore(VectorStore):
    def __init__(self, index_dir: str = "vectordb/faiss"):
        import faiss
        from sentence_transformers import SentenceTransformer

        os.makedirs(index_dir, exist_ok=True)
        self._index_dir = index_dir
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._dim = self._model.get_sentence_embedding_dimension()

        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "chunks.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self._chunks: list[Chunk] = pickle.load(f)
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._chunks = []
        logger.info("FAISSVectorStore initialised (dim=%d)", self._dim)

    def _save(self) -> None:
        import faiss

        faiss.write_index(self._index, os.path.join(self._index_dir, "index.faiss"))
        with open(os.path.join(self._index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    def ingest(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        self._index.add(np.array(embeddings, dtype=np.float32))
        self._chunks.extend(chunks)
        self._save()
        logger.info("Ingested %d chunks into FAISS", len(chunks))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []
        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(q_emb, dtype=np.float32), min(k, self._index.ntotal))
        output: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            output.append(SearchResult(chunk=self._chunks[idx], score=float(score)))
        return output

    def count(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vectorstore(backend: str = "chroma", **kwargs: Any) -> VectorStore:
    """backend: 'chroma' | 'faiss'"""
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    if backend == "faiss":
        return FAISSVectorStore(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")
