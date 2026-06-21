"""
Top-k Comparison Experiment  (Benchmark 4)

Measures how retrieval quality changes across top-k values: 3, 5, 10, 15.

CLI:
    python -m eval.topk_experiment --dataset data/qa.json
    python -m eval.topk_experiment --dataset data/qa.json --top-k-values 3 5 10 --retriever-type dense
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from eval.benchmark_runner import MockChain, TopKWrapper, run_single_experiment, save_benchmark_results
from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K_VALUES: tuple[int, ...] = (3, 5, 10, 15)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_topk_experiment(
    dataset: list[dict[str, Any]],
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
    retriever_type: str = "hybrid",
    base_chain: Any | None = None,
    output_path: str = "results/benchmark_topk.json",
) -> list[ExperimentResult]:
    """
    Run the dataset with each top-k value and compare RAGAS metrics.

    Parameters
    ----------
    dataset : list[dict]
        Q&A dataset with 'question' and 'ground_truth' keys.
    top_k_values : tuple[int, ...]
        Top-k values to compare (default: 3, 5, 10, 15).
    retriever_type : str
        Retrieval mode applied to all runs.
    base_chain : Any | None
        Chain to wrap; defaults to MockChain.
    """
    if base_chain is None:
        base_chain = MockChain()

    logger.info("Benchmark 4 — Top-k comparison: %s (retriever=%s)", top_k_values, retriever_type)
    results: list[ExperimentResult] = []

    for k in top_k_values:
        wrapped = TopKWrapper(base_chain, top_k=k)
        result = run_single_experiment(
            wrapped, dataset,
            experiment_name=f"topk_{k}",
            experiment_type="topk",
            retriever_type=retriever_type,
            top_k=k,
        )
        results.append(result)
        logger.info(
            "  top_k=%-3d  answer_relevancy=%-6s  latency=%.2fs",
            k, result.answer_relevancy, result.avg_latency_seconds or 0,
        )

    save_benchmark_results(
        results, output_path,
        extra={
            "benchmark": "topk_comparison",
            "retriever_type": retriever_type,
            "top_k_values": list(top_k_values),
        },
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Top-k Retrieval Experiment")
    parser.add_argument("--dataset", required=True, help="Path to Q&A JSON dataset")
    parser.add_argument(
        "--top-k-values", nargs="+", type=int,
        default=list(DEFAULT_TOP_K_VALUES),
        metavar="K",
    )
    parser.add_argument("--retriever-type", default="hybrid", choices=("bm25", "dense", "hybrid"))
    parser.add_argument("--output", default="results/benchmark_topk.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    results = run_topk_experiment(
        dataset,
        top_k_values=tuple(args.top_k_values),
        retriever_type=args.retriever_type,
        output_path=args.output,
    )
    print(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()
