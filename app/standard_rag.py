"""
Standard (non-reflective) RAG baseline chain.

Retrieve → Generate, no self-evaluation loop.
Used as the baseline in eval comparisons against Auto-RAG.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from graph.nodes.faithfulness_node import check_faithfulness

logger = logging.getLogger(__name__)


class StandardRAGChain:

    def __init__(
        self,
        vectorstore=None,
        bm25_retriever=None,
        llm=None,
    ) -> None:
        self._vs = vectorstore
        self._bm25 = bm25_retriever
        self._llm = llm

    def run(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        t0 = time.monotonic()

        chunks = self._retrieve(query, retrieval_mode, top_k)
        contexts = [c["text"] for c in chunks]
        sources = [
            {
                "id": c["id"],
                "text": c["text"][:200],
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
        ]
        answer = self._generate(query, chunks)

        # Use heuristic faithfulness (no extra LLM call) to keep generation LLM usage isolated.
        faithfulness_result = check_faithfulness(query, answer, contexts, llm=None)

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "retriever_type": retrieval_mode,
            "top_k": top_k,
            "latency_seconds": time.monotonic() - t0,
            "faithfulness_result": faithfulness_result,
            "faithfulness_score": faithfulness_result["confidence"] if faithfulness_result["faithful"] else 0.0,
            "unsupported_claims": faithfulness_result.get("unsupported_claims", []),
        }

    def _retrieve(
        self, query: str, mode: str, top_k: int
    ) -> list[dict[str, Any]]:
        has_dense = self._vs is not None
        has_sparse = self._bm25 is not None

        if mode == "hybrid" and has_dense and has_sparse:
            dense = self._vs.search(query, k=top_k)
            sparse = self._bm25.retrieve(query, k=top_k)
            return self._rrf_fuse(dense, sparse, top_k)

        if has_dense:
            return [r["chunk"] for r in self._vs.search(query, k=top_k)]

        if has_sparse:
            return [r["chunk"] for r in self._bm25.retrieve(query, k=top_k)]

        logger.warning("StandardRAGChain: no retriever configured, returning empty context.")
        return []

    @staticmethod
    def _rrf_fuse(
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        for rank, item in enumerate(dense):
            chunk = item["chunk"]
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks_by_id.setdefault(cid, chunk)

        for rank, item in enumerate(sparse):
            chunk = item["chunk"]
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks_by_id.setdefault(cid, chunk)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [chunks_by_id[cid] for cid in sorted_ids]

    def _generate(self, query: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant context found for the given query."

        if self._llm is None:
            parts = [c["text"][:200] for c in chunks[:3]]
            return "Retrieved context: " + " ... ".join(parts)

        context = "\n\n".join(
            f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)
        )
        prompt = (
            "Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        response = self._llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
