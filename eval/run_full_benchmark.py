"""
Full benchmark suite for publication.

Runs all experiments using real Ollama LLMs and custom RAGAS-style metrics.
Results are saved to results/benchmark_*.json.

Usage:
    python -m eval.run_full_benchmark
    python -m eval.run_full_benchmark --n-main 10 --n-ablation 5
    python -m eval.run_full_benchmark --only thresholds --dataset data/challenging_qa_dataset.json
    python -m eval.run_full_benchmark --only ablation --dataset data/challenging_qa_dataset.json
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


def _setup_chains(vs, bm25, grade_threshold: float = 0.75):
    from app.standard_rag import StandardRAGChain
    from graph.autorag_chain import AutoRAGChain
    from app.llm_factory import get_llm

    llm = get_llm()
    standard = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25, llm=llm)
    autorag = AutoRAGChain(
        vectorstore=vs, bm25_retriever=bm25, llm=llm,
        grade_threshold=grade_threshold, max_rewrites=2,
    )
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
                "rewrite_triggered": out.get("rewrite_triggered", False),
                "original_query": out.get("original_query", q),
                "final_query": out.get("final_query", q),
                "rewritten_query": out.get("rewritten_query"),
                "grade_confidence": out.get("grade_confidence", 0.0),
                "grade_relevant": out.get("grade_relevant", False),
                "avg_retrieval_score_initial": out.get("avg_retrieval_score_initial", 0.0),
                "avg_retrieval_score_final": out.get("avg_retrieval_score_final", 0.0),
                "rewrite_trace": out.get("rewrite_trace", []),
                "latency_seconds": out.get("latency_seconds", 0.0),
            })
        except Exception as exc:
            logger.warning("  FAILED: %s", exc)
            records.append({
                "question": q, "answer": "", "contexts": [], "ground_truth": gt,
                "rewrite_count": 0, "rewrite_triggered": False,
                "original_query": q, "final_query": q, "rewritten_query": None,
                "grade_confidence": 0.0, "grade_relevant": False,
                "avg_retrieval_score_initial": 0.0, "avg_retrieval_score_final": 0.0,
                "rewrite_trace": [], "latency_seconds": 0.0,
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
# Benchmark 4: Threshold Experiment
# ---------------------------------------------------------------------------

def run_threshold_experiment(dataset, vs, bm25) -> dict:
    from graph.autorag_chain import AutoRAGChain
    from app.llm_factory import get_llm
    from eval.custom_metrics import compute_all_metrics

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    thresholds = [0.60, 0.75, 0.85]
    results = {
        "timestamp": ts, "benchmark": "threshold_experiment",
        "n_questions": len(dataset), "thresholds_tested": thresholds, "experiments": [],
    }

    llm = get_llm()
    for threshold in thresholds:
        logger.info("=== threshold=%.2f (%d questions) ===", threshold, len(dataset))
        chain = AutoRAGChain(
            vectorstore=vs, bm25_retriever=bm25, llm=llm,
            grade_threshold=threshold, max_rewrites=2,
        )
        records = _run_chain(chain, dataset)
        logger.info("Computing metrics for threshold=%.2f...", threshold)
        metrics = compute_all_metrics(records)
        rewrite_rate = sum(1 for r in records if r.get("rewrite_triggered")) / len(records)
        exp = {
            "experiment_name": f"threshold_{threshold}",
            "experiment_type": "threshold",
            "timestamp": ts,
            "n_questions": len(records),
            "grade_threshold": threshold,
            "retriever_type": "hybrid",
            "top_k": 5,
            "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records]), 3),
            "avg_rewrite_count": round(_avg([r["rewrite_count"] for r in records]), 3),
            "rewrite_rate": round(rewrite_rate, 3),
            "avg_grade_confidence": round(_avg([r["grade_confidence"] for r in records]), 3),
            **metrics,
        }
        results["experiments"].append(exp)
        logger.info(
            "  threshold=%.2f: F=%.3f AR=%.3f rewrites=%.2f rate=%.0f%%",
            threshold, metrics["faithfulness"], metrics["answer_relevancy"],
            exp["avg_rewrite_count"], rewrite_rate * 100,
        )

    _save(results, "results/benchmark_thresholds.json")
    return results


# ---------------------------------------------------------------------------
# Benchmark 5: Ablation Study
# ---------------------------------------------------------------------------

def run_ablation_study(dataset, vs, bm25) -> dict:
    from app.standard_rag import StandardRAGChain
    from graph.autorag_chain import AutoRAGChain
    from app.llm_factory import get_llm
    from eval.custom_metrics import compute_all_metrics

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {
        "timestamp": ts, "benchmark": "ablation_study",
        "n_questions": len(dataset), "experiments": [],
    }

    llm = get_llm()

    # Condition 1: Standard RAG (no reflection)
    logger.info("=== Ablation: standard_rag ===")
    standard = StandardRAGChain(vectorstore=vs, bm25_retriever=bm25, llm=llm)
    records_std = _run_chain(standard, dataset)
    metrics_std = compute_all_metrics(records_std)
    results["experiments"].append({
        "experiment_name": "standard_rag",
        "description": "Baseline: retrieve + generate, no reflection",
        "grade_threshold": None, "rewrite_enabled": False,
        "retriever_type": "hybrid", "top_k": 5,
        "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records_std]), 3),
        "avg_rewrite_count": 0.0, "rewrite_rate": 0.0,
        **metrics_std,
    })

    # Condition 2: Auto-RAG rewrite disabled (threshold=1.01, never triggers)
    logger.info("=== Ablation: autorag_no_rewrite ===")
    chain_norewrite = AutoRAGChain(
        vectorstore=vs, bm25_retriever=bm25, llm=llm,
        grade_threshold=1.01, max_rewrites=0,
    )
    records_norewrite = _run_chain(chain_norewrite, dataset)
    metrics_norewrite = compute_all_metrics(records_norewrite)
    results["experiments"].append({
        "experiment_name": "autorag_no_rewrite",
        "description": "Auto-RAG pipeline with rewrite disabled (threshold=1.01)",
        "grade_threshold": 1.01, "rewrite_enabled": False,
        "retriever_type": "hybrid", "top_k": 5,
        "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records_norewrite]), 3),
        "avg_rewrite_count": 0.0, "rewrite_rate": 0.0,
        **metrics_norewrite,
    })

    # Condition 3: Auto-RAG rewrite enabled (threshold=0.75)
    logger.info("=== Ablation: autorag_with_rewrite ===")
    chain_rewrite = AutoRAGChain(
        vectorstore=vs, bm25_retriever=bm25, llm=llm,
        grade_threshold=0.75, max_rewrites=2,
    )
    records_rewrite = _run_chain(chain_rewrite, dataset)
    metrics_rewrite = compute_all_metrics(records_rewrite)
    rewrite_rate = sum(1 for r in records_rewrite if r.get("rewrite_triggered")) / len(records_rewrite)
    results["experiments"].append({
        "experiment_name": "autorag_with_rewrite",
        "description": "Full Auto-RAG with self-reflection (threshold=0.75)",
        "grade_threshold": 0.75, "rewrite_enabled": True,
        "retriever_type": "hybrid", "top_k": 5,
        "avg_latency_seconds": round(_avg([r["latency_seconds"] for r in records_rewrite]), 3),
        "avg_rewrite_count": round(_avg([r["rewrite_count"] for r in records_rewrite]), 3),
        "rewrite_rate": round(rewrite_rate, 3),
        **metrics_rewrite,
    })

    logger.info(
        "Ablation: std F=%.3f | no-rewrite F=%.3f | rewrite F=%.3f (rate=%.0f%%)",
        metrics_std["faithfulness"], metrics_norewrite["faithfulness"],
        metrics_rewrite["faithfulness"], rewrite_rate * 100,
    )

    _save(results, "results/ablation_study.json")
    return results


# ---------------------------------------------------------------------------
# Benchmark 6: Rewrite Effectiveness Analysis
# ---------------------------------------------------------------------------

def run_rewrite_effectiveness(dataset, vs, bm25) -> dict:
    from graph.autorag_chain import AutoRAGChain
    from app.llm_factory import get_llm
    from eval.custom_metrics import compute_faithfulness, compute_context_precision

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    results: dict[str, Any] = {
        "timestamp": ts, "benchmark": "rewrite_effectiveness",
        "n_questions": len(dataset), "records": [],
    }

    llm = get_llm()
    chain = AutoRAGChain(
        vectorstore=vs, bm25_retriever=bm25, llm=llm,
        grade_threshold=0.75, max_rewrites=2,
    )

    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item.get("ground_truth", "")
        logger.info("  [%d/%d] %s", i + 1, len(dataset), q[:70])
        try:
            out = chain.run(q, retrieval_mode="hybrid", top_k=5)
            trace = out.get("rewrite_trace", [])

            record: dict[str, Any] = {
                "question": q,
                "ground_truth": gt,
                "rewrite_triggered": out.get("rewrite_triggered", False),
                "rewrite_count": out.get("rewrite_count", 0),
                "original_query": out.get("original_query", q),
                "final_query": out.get("final_query", q),
                "rewritten_query": out.get("rewritten_query"),
                "before_grade_confidence": trace[0]["grade_confidence"] if trace else 0.0,
                "after_grade_confidence": out.get("grade_confidence", 0.0),
                "before_avg_score": out.get("avg_retrieval_score_initial", 0.0),
                "after_avg_score": out.get("avg_retrieval_score_final", 0.0),
                "final_answer": out.get("answer", ""),
                "latency_seconds": out.get("latency_seconds", 0.0),
                "trace": trace,
            }

            # Compute before/after faithfulness only for rewritten questions
            if out.get("rewrite_triggered") and len(trace) >= 2:
                record["confidence_improved"] = (
                    record["after_grade_confidence"] > record["before_grade_confidence"]
                )
                record["score_improved"] = (
                    record["after_avg_score"] > record["before_avg_score"]
                )
            else:
                record["confidence_improved"] = None
                record["score_improved"] = None

            results["records"].append(record)
        except Exception as exc:
            logger.warning("  FAILED: %s", exc)

    rewrites = [r for r in results["records"] if r["rewrite_triggered"]]
    results["summary"] = {
        "total_questions": len(dataset),
        "rewrite_triggered_count": len(rewrites),
        "rewrite_rate": round(len(rewrites) / len(dataset), 3) if dataset else 0.0,
        "avg_confidence_before": round(
            _avg([r["before_grade_confidence"] for r in rewrites]), 3
        ) if rewrites else 0.0,
        "avg_confidence_after": round(
            _avg([r["after_grade_confidence"] for r in rewrites]), 3
        ) if rewrites else 0.0,
        "confidence_improved_count": sum(
            1 for r in rewrites if r.get("confidence_improved")
        ),
    }
    logger.info(
        "Rewrite effectiveness: %d/%d rewrites, confidence %.3f → %.3f",
        results["summary"]["rewrite_triggered_count"],
        results["summary"]["total_questions"],
        results["summary"]["avg_confidence_before"],
        results["summary"]["avg_confidence_after"],
    )

    _save(results, "results/rewrite_effectiveness.json")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full benchmark suite for publication")
    parser.add_argument("--dataset", default="data/amnesty_qa.json")
    parser.add_argument("--challenging-dataset", default="data/challenging_qa_dataset.json")
    parser.add_argument("--n-main", type=int, default=10,
                        help="Questions for main RAG comparison (default: 10)")
    parser.add_argument("--n-ablation", type=int, default=5,
                        help="Questions for ablation experiments (default: 5)")
    parser.add_argument("--only",
                        choices=["rag", "retriever", "topk", "thresholds", "ablation", "rewrite", "all"],
                        default="all")
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

    # Challenging dataset benchmarks — use challenging_qa_dataset by default
    challenging_path = args.challenging_dataset
    try:
        dataset_challenging = _load_dataset(challenging_path, args.n_ablation)
        logger.info("Challenging dataset: %d questions from %s", len(dataset_challenging), challenging_path)
    except FileNotFoundError:
        dataset_challenging = dataset_abl
        logger.warning("Challenging dataset not found at %s, falling back to ablation dataset", challenging_path)

    if args.only in ("thresholds", "all"):
        logger.info("=== Benchmark 4: Threshold Experiment ===")
        run_threshold_experiment(dataset_challenging, vs, bm25)

    if args.only in ("ablation", "all"):
        logger.info("=== Benchmark 5: Ablation Study ===")
        run_ablation_study(dataset_challenging, vs, bm25)

    if args.only in ("rewrite", "all"):
        logger.info("=== Benchmark 6: Rewrite Effectiveness ===")
        run_rewrite_effectiveness(dataset_challenging, vs, bm25)

    elapsed = time.monotonic() - t_total
    logger.info("All benchmarks complete in %.1f minutes", elapsed / 60)
    logger.info("Results saved to results/")


if __name__ == "__main__":
    main()
