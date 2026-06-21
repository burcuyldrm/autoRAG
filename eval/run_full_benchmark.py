"""
Full benchmark suite for publication.

Runs all experiments using real Ollama LLMs and custom RAGAS-style metrics.
Results are saved to results/benchmark_*.json.

Usage:
    python -m eval.run_full_benchmark
    python -m eval.run_full_benchmark --n-main 10 --n-ablation 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup_retrievers():
    from vectordb.vectorstore import ChromaVectorStore
    from retrieval.bm25_retriever import BM25Retriever

    with open("data/chunks/amnesty_chunks.json") as f:
        chunks = json.load(f)

    vs = ChromaVectorStore()
    bm25 = BM25Retriever(chunks=chunks)
    logger.info("Retrievers ready: %d chunks in ChromaDB, %d in BM25", vs.count(), len(bm25._chunks))
    return vs, bm25


def _setup_chains(vs, bm25):
    from app.standard_rag import StandardRAGChain
    from graph.autorag_chain import AutoRAGChain
    from app.llm_factory import get_llm, get_fast_llm

    llm = get_llm()
    standard = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25, llm=llm)
    autorag = AutoRAGChain(vectorstore=vs, bm25_retriever=bm25, llm=llm, grade_threshold=0.60, max_rewrites=2)
    return standard, autorag


def _load_dataset(path: str, n: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:n] if n else data


def _run_chain(chain, dataset: list[dict], mode: str = "hybrid", top_k: int = 5) -> list[dict]:
    records = []
    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item.get("ground_truth", "")
        logger.info("  [%d/%d] %s", i + 1, len(dataset), q[:70])
        try:
            out = chain.run(q, retrieval_mode=mode, top_k=top_k)
            records.append({
                "question": q,
                "answer": out.get("answer", ""),
                "contexts": out.get("contexts", []),
                "ground_truth": gt,
                "rewrite_count": out.get("rewrite_count", 0),
                "latency_seconds": out.get("latency_seconds", 0.0),
            })
        except Exception as exc:
            logger.warning("  FAILED: %s", exc)
            records.append({
                "question": q, "answer": "", "contexts": [], "ground_truth": gt,
                "rewrite_count": 0, "latency_seconds": 0.0,
            })
    return records


def _avg(lst: list, default: float = 0.0) -> float:
    vals = [v for v in lst if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else default


def _save(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Benchmark 1: Standard RAG vs Auto-RAG
# ---------------------------------------------------------------------------

def run_rag_comparison(dataset, standard_chain, autorag_chain) -> dict:
    from eval.custom_metrics import compute_all_metrics

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {"timestamp": ts, "benchmark": "rag_comparison", "n_questions": len(dataset), "experiments": []}

    for name, chain in [("standard_rag", standard_chain), ("auto_rag", autorag_chain)]:
        logger.info("=== %s (%d questions) ===", name, len(dataset))
        records = _run_chain(chain, dataset)
        logger.info("Computing metrics for %s...", name)
        metrics = compute_all_metrics(records)
        exp = {
            "experiment_name": name,
            "experiment_type": "rag_comparison",
            "timestamp": ts,
            "n_questions": len(records),
            "retriever_type": "hybrid",
            "top_k": 5,
            "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records]), 3),
            "avg_rewrite_count": round(_avg([r["rewrite_count"] for r in records]), 2),
            **metrics,
        }
        results["experiments"].append(exp)
        results[name] = {**metrics, "avg_latency_seconds": exp["avg_latency_seconds"],
                         "avg_rewrite_count": exp.get("avg_rewrite_count", 0)}
        logger.info("  %s: F=%.3f AR=%.3f CP=%.3f CR=%.3f",
                    name, metrics["faithfulness"], metrics["answer_relevancy"],
                    metrics["context_precision"], metrics["context_recall"])

    results["retrieval_mode"] = "hybrid"
    results["chunk_size"] = 1024
    results["rewrite"] = True
    _save(results, "results/benchmark_rag_comparison.json")
    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Retriever comparison (BM25 / Dense / Hybrid)
# ---------------------------------------------------------------------------

def run_retriever_comparison(dataset, vs, bm25) -> dict:
    from app.standard_rag import StandardRAGChain
    from app.llm_factory import get_llm
    from eval.custom_metrics import compute_all_metrics

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {"timestamp": ts, "benchmark": "retriever_comparison", "n_questions": len(dataset), "experiments": []}

    llm = get_llm()
    chain = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25, llm=llm)

    for mode in ("bm25", "dense", "hybrid"):
        logger.info("=== retriever=%s (%d questions) ===", mode, len(dataset))
        records = _run_chain(chain, dataset, mode=mode)
        logger.info("Computing metrics for %s...", mode)
        metrics = compute_all_metrics(records)
        exp = {
            "experiment_name": f"retriever_{mode}",
            "experiment_type": "retriever",
            "timestamp": ts,
            "n_questions": len(records),
            "retriever_type": mode,
            "top_k": 5,
            "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records]), 3),
            **metrics,
        }
        results["experiments"].append(exp)
        logger.info("  %s: F=%.3f AR=%.3f CP=%.3f CR=%.3f",
                    mode, metrics["faithfulness"], metrics["answer_relevancy"],
                    metrics["context_precision"], metrics["context_recall"])

    _save(results, "results/benchmark_retriever.json")
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Top-K comparison
# ---------------------------------------------------------------------------

def run_topk_comparison(dataset, standard_chain) -> dict:
    from eval.custom_metrics import compute_all_metrics

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    top_k_values = [3, 5, 10, 15]
    results = {
        "timestamp": ts, "benchmark": "topk_comparison",
        "n_questions": len(dataset), "top_k_values": top_k_values,
        "retriever_type": "hybrid", "experiments": [],
    }

    for k in top_k_values:
        logger.info("=== top_k=%d (%d questions) ===", k, len(dataset))
        records = _run_chain(standard_chain, dataset, top_k=k)
        logger.info("Computing metrics for top_k=%d...", k)
        metrics = compute_all_metrics(records)
        exp = {
            "experiment_name": f"topk_{k}",
            "experiment_type": "topk",
            "timestamp": ts,
            "n_questions": len(records),
            "retriever_type": "hybrid",
            "top_k": k,
            "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records]), 3),
            **metrics,
        }
        results["experiments"].append(exp)
        logger.info("  top_k=%d: F=%.3f AR=%.3f CP=%.3f CR=%.3f",
                    k, metrics["faithfulness"], metrics["answer_relevancy"],
                    metrics["context_precision"], metrics["context_recall"])

    _save(results, "results/benchmark_topk.json")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full benchmark suite for publication")
    parser.add_argument("--dataset", default="data/amnesty_qa.json")
    parser.add_argument("--n-main", type=int, default=10,
                        help="Questions for main RAG comparison (default: 10)")
    parser.add_argument("--n-ablation", type=int, default=5,
                        help="Questions for ablation experiments (default: 5)")
    parser.add_argument("--only", choices=["rag", "retriever", "topk", "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading dataset from %s", args.dataset)
    dataset_main = _load_dataset(args.dataset, args.n_main)
    dataset_abl = _load_dataset(args.dataset, args.n_ablation)
    logger.info("Main: %d questions, Ablation: %d questions", len(dataset_main), len(dataset_abl))

    logger.info("Setting up retrievers...")
    vs, bm25 = _setup_retrievers()

    t_total = time.monotonic()

    if args.only in ("rag", "all"):
        logger.info("=== Benchmark 1: Standard RAG vs Auto-RAG ===")
        standard, autorag = _setup_chains(vs, bm25)
        run_rag_comparison(dataset_main, standard, autorag)

    if args.only in ("retriever", "all"):
        logger.info("=== Benchmark 2: Retriever Comparison ===")
        run_retriever_comparison(dataset_abl, vs, bm25)

    if args.only in ("topk", "all"):
        logger.info("=== Benchmark 3: Top-K Comparison ===")
        if args.only == "all":
            standard, _ = _setup_chains(vs, bm25)
        run_topk_comparison(dataset_abl, standard)

    elapsed = time.monotonic() - t_total
    logger.info("All benchmarks complete in %.1f minutes", elapsed / 60)
    logger.info("Results saved to results/")


if __name__ == "__main__":
    main()
