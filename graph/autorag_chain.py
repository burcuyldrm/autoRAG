"""
Auto-RAG Chain

Orchestrates: retrieve → grade → [rewrite → retrieve → grade]* → generate

All loop metrics are recorded in the output dict for eval and UI consumption.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from graph.nodes.grade_node import grade_node
from graph.nodes.generate_node import generate_node
from graph.nodes.rewrite_node import rewrite_query
from graph.nodes.faithfulness_node import faithfulness_node

logger = logging.getLogger(__name__)

_RRF_K = 60


class AutoRAGChain:
    """
    Self-reflective RAG chain.

    Parameters
    ----------
    vectorstore : VectorStore | None
        Dense retrieval backend (FAISS or ChromaDB).
    bm25_retriever : BM25Retriever | None
        Sparse BM25 retriever.
    llm : Any | None
        LangChain-compatible chat model used by grade, generate, and rewrite nodes.
    grade_threshold : float
        Minimum grade confidence to accept retrieval without rewriting (default 0.75).
        Rewrite triggers when grade_confidence < grade_threshold.
    retrieval_threshold : float
        Minimum average retrieval score to accept without rewriting (default 0.0 = disabled).
        When > 0, rewrite also triggers if avg_retrieval_score < retrieval_threshold.
    max_rewrites : int
        Maximum number of query rewrites before forcing generation (default 2).
    """

    def __init__(
        self,
        vectorstore=None,
        bm25_retriever=None,
        llm=None,
        grade_threshold: float = 0.75,
        retrieval_threshold: float = 0.0,
        max_rewrites: int = 2,
    ) -> None:
        self._vs = vectorstore
        self._bm25 = bm25_retriever
        self._llm = llm
        self._grade_threshold = grade_threshold
        self._retrieval_threshold = retrieval_threshold
        self._max_rewrites = max_rewrites

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Run the Auto-RAG self-reflection loop.

        Returns
        -------
        dict
            original_query    : str
            final_query       : str   — query used for the final generate call
            rewritten_query   : str | None — last rewrite (None if no rewrite happened)
            rewrite_count     : int
            grade_confidence  : float
            grade_relevant    : bool
            retrieval_mode    : str
            top_k             : int
            latency_seconds   : float
            retrieved_scores  : list[float]
            answer            : str
            contexts          : list[str]   — chunk texts (for RAGAS)
            sources           : list[dict]
            chunks            : list[dict]  — full chunk dicts (for UI trace)
        """
        t0 = time.monotonic()

        original_query = query
        current_query = query
        rewrite_count = 0
        last_rewritten: str | None = None
        last_grade: dict | None = None
        last_chunks: list[dict] = []
        last_scores: list[float] = []
        rewrite_trace: list[dict[str, Any]] = []
        initial_scores: list[float] = []

        for attempt in range(self._max_rewrites + 1):
            t_attempt = time.monotonic()
            chunks, scores = self._retrieve(current_query, retrieval_mode, top_k)

            if attempt == 0:
                initial_scores = scores

            avg_score = sum(scores) / len(scores) if scores else 0.0

            state: dict[str, Any] = {"query": current_query, "chunks": chunks}
            state = grade_node(state, llm=self._llm)
            grade = state["grade_result"]

            # Acceptance condition: grade confidence AND retrieval score both above thresholds.
            # Using confidence as SOLE gate (not `relevant`) prevents the model from
            # bypassing the threshold by always returning relevant=True.
            grade_ok = grade["confidence"] >= self._grade_threshold
            score_ok = (self._retrieval_threshold <= 0.0) or (avg_score >= self._retrieval_threshold)
            accepted = grade_ok and score_ok

            trace_entry: dict[str, Any] = {
                "attempt": attempt,
                "query": current_query,
                "avg_retrieval_score": round(avg_score, 4),
                "grade_relevant": grade["relevant"],
                "grade_confidence": grade["confidence"],
                "grade_reasoning": grade.get("reasoning", ""),
                "grade_ok": grade_ok,
                "score_ok": score_ok,
                "accepted": accepted,
                "latency_seconds": round(time.monotonic() - t_attempt, 3),
            }

            last_chunks = chunks
            last_scores = scores
            last_grade = grade

            if accepted:
                logger.info(
                    "Grade accepted at attempt %d (confidence=%.2f >= %.2f, avg_score=%.3f)",
                    attempt, grade["confidence"], self._grade_threshold, avg_score,
                )
                rewrite_trace.append(trace_entry)
                break

            if rewrite_count >= self._max_rewrites:
                logger.info("Max rewrites (%d) reached; proceeding to generate.", self._max_rewrites)
                rewrite_trace.append(trace_entry)
                break

            logger.info(
                "Rewrite triggered at attempt %d (confidence=%.2f < %.2f, avg_score=%.3f)",
                attempt, grade["confidence"], self._grade_threshold, avg_score,
            )

            reason = grade.get("reasoning", "")
            rewritten = rewrite_query(current_query, reason=reason, llm=self._llm)
            trace_entry["rewritten_to"] = rewritten
            rewrite_trace.append(trace_entry)

            current_query = rewritten
            last_rewritten = current_query
            rewrite_count += 1
            logger.info("Rewrite #%d: %r", rewrite_count, current_query)

        gen_state: dict[str, Any] = {"query": current_query, "chunks": last_chunks}
        gen_state = generate_node(gen_state, llm=self._llm)

        # Faithfulness evaluation
        gen_state["query"] = current_query
        gen_state = faithfulness_node(gen_state, llm=self._llm)

        final_avg_score = sum(last_scores) / len(last_scores) if last_scores else 0.0

        return {
            "original_query": original_query,
            "final_query": current_query,
            "rewritten_query": last_rewritten,
            "rewrite_count": rewrite_count,
            "rewrite_triggered": rewrite_count > 0,
            "grade_confidence": last_grade["confidence"] if last_grade else 0.0,
            "grade_relevant": last_grade["relevant"] if last_grade else False,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "latency_seconds": time.monotonic() - t0,
            "retrieved_scores": last_scores,
            "retrieved_scores_before": initial_scores,
            "retrieved_scores_after": last_scores,
            "avg_retrieval_score_initial": round(sum(initial_scores) / len(initial_scores), 4) if initial_scores else 0.0,
            "avg_retrieval_score_final": round(final_avg_score, 4),
            "rewrite_trace": rewrite_trace,
            "answer": gen_state.get("final_answer", ""),
            "contexts": [c["text"] for c in last_chunks],
            "sources": gen_state.get("sources", []),
            "chunks": last_chunks,
            "faithfulness_result": gen_state.get("faithfulness_result", {}),
            "faithfulness_score": gen_state.get("faithfulness_score", 0.0),
            "unsupported_claims": gen_state.get("unsupported_claims", []),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self, query: str, mode: str, top_k: int
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Returns (chunks, scores)."""
        has_dense = self._vs is not None
        has_sparse = self._bm25 is not None

        if mode == "hybrid" and has_dense and has_sparse:
            dense = self._vs.search(query, k=top_k)
            sparse = self._bm25.retrieve(query, k=top_k)
            return _rrf_fuse(dense, sparse, top_k)

        if has_dense:
            results = self._vs.search(query, k=top_k)
            return [r["chunk"] for r in results], [r["score"] for r in results]

        if has_sparse:
            results = self._bm25.retrieve(query, k=top_k)
            return [r["chunk"] for r in results], [r["score"] for r in results]

        logger.warning("AutoRAGChain: no retriever configured.")
        return [], []


# ------------------------------------------------------------------
# Module-level RRF helper
# ------------------------------------------------------------------

def _rrf_fuse(
    dense: list[dict],
    sparse: list[dict],
    top_k: int,
    rrf_k: int = _RRF_K,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Reciprocal Rank Fusion. Returns (chunks, rrf_scores)."""
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
    return (
        [chunks_by_id[cid] for cid in sorted_ids],
        [scores[cid] for cid in sorted_ids],
    )
