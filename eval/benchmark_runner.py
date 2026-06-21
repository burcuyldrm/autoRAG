"""
Core benchmark infrastructure.

Provides:
  - MockChain          : API-less dummy chain for testing
  - RetrieverWrapper   : fixes retrieval_mode on run()
  - TopKWrapper        : fixes top_k on run()
  - compute_ragas_metrics_safe : RAGAS with graceful fallback
  - run_single_experiment      : chain → ExperimentResult
  - save_benchmark_results     : list[ExperimentResult] → JSON
  - run_rag_comparison         : Benchmark 1 — Standard RAG vs Auto-RAG
  - run_retriever_comparison   : Benchmark 2 — bm25 / dense / hybrid
  - run_chunk_comparison       : Benchmark 3 — chunk sizes 256/512/1024/2048
  - run_full_suite             : runs benchmarks 1–3 in sequence

CLI:
    python -m eval.benchmark_runner --dataset data/qa.json [--benchmark all|rag|retriever|chunk]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Any

from eval.eval_runner import run_chain_on_dataset
from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)

_GLOBAL_SEED = 42


def _set_seed(seed: int = _GLOBAL_SEED) -> None:
    """Fix random seeds for reproducible experiment results."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

RETRIEVER_TYPES: tuple[str, ...] = ("bm25", "dense", "hybrid")
CHUNK_SIZES: tuple[int, ...] = (256, 512, 1024, 2048)


# ---------------------------------------------------------------------------
# Mock chain — no API required
# ---------------------------------------------------------------------------

class MockChain:
    """Deterministic stand-in chain. Use when LLM/vectorstore are unavailable."""

    def __init__(
        self,
        answer: str = "Mock answer about the queried topic.",
        rewrite_count: int = 0,
        latency: float = 0.05,
    ) -> None:
        self._answer = answer
        self._rewrite_count = rewrite_count
        self._latency = latency

    def run(self, query: str, **_: Any) -> dict[str, Any]:
        return {
            "answer": self._answer,
            "contexts": [f"Context passage related to: {query[:80]}"],
            "sources": [],
            "rewrite_count": self._rewrite_count,
            "latency_seconds": self._latency,
            "grade_confidence": 0.85,
            "grade_relevant": True,
        }


# ---------------------------------------------------------------------------
# Chain wrappers — pin retrieval_mode or top_k across run_chain_on_dataset
# ---------------------------------------------------------------------------

class RetrieverWrapper:
    """Injects a fixed retrieval_mode into every run() call."""

    def __init__(self, chain: Any, retrieval_mode: str) -> None:
        self._chain = chain
        self._mode = retrieval_mode

    def run(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self._chain.run(query, retrieval_mode=self._mode, **kwargs)


class TopKWrapper:
    """Injects a fixed top_k into every run() call."""

    def __init__(self, chain: Any, top_k: int) -> None:
        self._chain = chain
        self._top_k = top_k

    def run(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self._chain.run(query, top_k=self._top_k, **kwargs)


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------

_EMPTY_METRICS: dict[str, Any] = {
    "faithfulness": None,
    "answer_relevancy": None,
    "context_precision": None,
    "context_recall": None,
}


def compute_ragas_metrics_safe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute all four RAGAS metrics using a local Ollama LLM and sentence-transformers
    embeddings — no API key required, fully reproducible.

    Falls back to None values only if Ollama is unreachable or records are empty.
    """
    if not records:
        return dict(_EMPTY_METRICS)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from app.llm_factory import get_ragas_embeddings, get_ragas_llm

        ragas_llm = get_ragas_llm(fast=True)
        ragas_embeddings = get_ragas_embeddings()

        ds = Dataset.from_list(records)
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }
    except Exception as exc:
        logger.warning("RAGAS evaluation failed (%s) — metrics set to None.", exc)
        return dict(_EMPTY_METRICS)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _avg(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def run_single_experiment(
    chain: Any,
    dataset: list[dict[str, Any]],
    experiment_name: str,
    experiment_type: str,
    model_name: str | None = None,
    retriever_type: str | None = None,
    chunk_size: int | None = None,
    top_k: int | None = None,
) -> ExperimentResult:
    """Run chain on dataset, compute RAGAS metrics, return ExperimentResult."""
    _set_seed()
    records = run_chain_on_dataset(chain, dataset)
    metrics = compute_ragas_metrics_safe(records)

    return ExperimentResult(
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        model_name=model_name,
        retriever_type=retriever_type,
        chunk_size=chunk_size,
        top_k=top_k,
        n_questions=len(dataset),
        avg_latency_seconds=_avg(records, "latency_seconds"),
        avg_rewrite_count=_avg(records, "rewrite_count"),
        **metrics,
    )


def save_benchmark_results(
    results: list[ExperimentResult],
    output_path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save list of ExperimentResult to a JSON file."""
    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_experiments": len(results),
        **(extra or {}),
        "experiments": [r.to_dict() for r in results],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %d experiment(s) to %s", len(results), output_path)


# ---------------------------------------------------------------------------
# Benchmark 1 — Standard RAG vs Auto-RAG
# ---------------------------------------------------------------------------

def run_rag_comparison(
    dataset: list[dict[str, Any]],
    standard_chain: Any | None = None,
    autorag_chain: Any | None = None,
    output_path: str = "results/benchmark_rag_comparison.json",
    n_questions: int | None = None,
    dataset_path: str | None = None,
) -> list[ExperimentResult]:
    """Benchmark 1: run Standard RAG and Auto-RAG side by side."""
    from eval.chains import get_standard_chain, get_autorag_chain

    if standard_chain is None:
        standard_chain = get_standard_chain()
    
    if autorag_chain is None:
        autorag_chain = get_autorag_chain()


    n = n_questions if n_questions is not None else len(dataset)
    logger.info("Benchmark 1 — Standard RAG vs Auto-RAG (%d questions)", n)
    results = [
        run_single_experiment(
            standard_chain, dataset,
            experiment_name="standard_rag",
            experiment_type="comparison",
        ),
        run_single_experiment(
            autorag_chain, dataset,
            experiment_name="auto_rag",
            experiment_type="comparison",
        ),
    ]
    extra: dict[str, Any] = {
        "benchmark_name": "standard_vs_autorag",
        "n_questions": n,
    }
    if dataset_path:
        extra["dataset_path"] = dataset_path
    save_benchmark_results(results, output_path, extra=extra)
    return results


# ---------------------------------------------------------------------------
# Benchmark 2 — Retriever type comparison
# ---------------------------------------------------------------------------

def run_retriever_comparison(
    dataset: list[dict[str, Any]],
    base_chain: Any | None = None,
    retriever_types: tuple[str, ...] = RETRIEVER_TYPES,
    output_path: str = "results/benchmark_retriever.json",
    n_questions: int | None = None,
    dataset_path: str | None = None,
) -> list[ExperimentResult]:
    """Benchmark 2: compare bm25, dense, and hybrid retrieval."""
    from eval.chains import get_standard_chain

    if base_chain is None:
        base_chain = get_standard_chain()

    n = n_questions if n_questions is not None else len(dataset)
    logger.info("Benchmark 2 — Retriever comparison: %s", retriever_types)
    results = []
    for mode in retriever_types:
        wrapped = RetrieverWrapper(base_chain, retrieval_mode=mode)
        result = run_single_experiment(
            wrapped, dataset,
            experiment_name=f"retriever_{mode}",
            experiment_type="retrieval",
            retriever_type=mode,
        )
        results.append(result)
        logger.info("  %-8s answer_relevancy=%s", mode, result.answer_relevancy)

    extra: dict[str, Any] = {
        "benchmark_name": "retriever_comparison",
        "n_questions": n,
    }
    if dataset_path:
        extra["dataset_path"] = dataset_path
    save_benchmark_results(results, output_path, extra=extra)
    return results


# ---------------------------------------------------------------------------
# Benchmark 3 — Chunk size comparison
# ---------------------------------------------------------------------------

def run_chunk_comparison(
    dataset: list[dict[str, Any]],
    base_chain: Any | None = None,
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES,
    retriever_type: str = "hybrid",
    output_path: str = "results/benchmark_chunk_size.json",
) -> list[ExperimentResult]:
    """Benchmark 3: compare chunk sizes 256 / 512 / 1024 / 2048."""
    from eval.chains import get_standard_chain

    if base_chain is None:
        base_chain = get_standard_chain()

    logger.info("Benchmark 3 — Chunk size comparison: %s", chunk_sizes)
    results = []
    for size in chunk_sizes:
        result = run_single_experiment(
            base_chain, dataset,
            experiment_name=f"chunk_{size}",
            experiment_type="chunking",
            retriever_type=retriever_type,
            chunk_size=size,
        )
        results.append(result)
        logger.info("  chunk_size=%-5d answer_relevancy=%s", size, result.answer_relevancy)

    save_benchmark_results(results, output_path, extra={"benchmark": "chunk_comparison"})
    return results


# ---------------------------------------------------------------------------
# Full suite (benchmarks 1–3)
# ---------------------------------------------------------------------------

def run_full_suite(
    dataset: list[dict[str, Any]],
    output_dir: str = "results",
    standard_chain: Any | None = None,
    autorag_chain: Any | None = None,
    base_chain: Any | None = None,
) -> dict[str, list[ExperimentResult]]:
    """Run benchmarks 1, 2, and 3 in sequence."""
    return {
        "rag_comparison": run_rag_comparison(
            dataset, standard_chain, autorag_chain,
            output_path=os.path.join(output_dir, "benchmark_rag_comparison.json"),
        ),
        "retriever": run_retriever_comparison(
            dataset, base_chain,
            output_path=os.path.join(output_dir, "benchmark_retriever.json"),
        ),
        "chunk": run_chunk_comparison(
            dataset, base_chain,
            output_path=os.path.join(output_dir, "benchmark_chunk_size.json"),
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Auto-RAG Benchmark Suite")
    parser.add_argument("--dataset", required=True, help="Path to Q&A JSON dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--benchmark",
        choices=("all", "rag", "retriever", "chunk"),
        default="all",
        help="Which benchmark to run",
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    n = len(dataset)
    output_dir = args.output_dir

    if args.benchmark in ("all", "rag"):
        run_rag_comparison(
            dataset,
            output_path=os.path.join(output_dir, "benchmark_rag_comparison.json"),
            n_questions=n,
            dataset_path=args.dataset,
        )
    if args.benchmark in ("all", "retriever"):
        run_retriever_comparison(
            dataset,
            output_path=os.path.join(output_dir, "benchmark_retriever.json"),
            n_questions=n,
            dataset_path=args.dataset,
        )
    if args.benchmark in ("all", "chunk"):
        run_chunk_comparison(
            dataset,
            output_path=os.path.join(output_dir, "benchmark_chunk_size.json"),
        )


if __name__ == "__main__":
    main()
