"""
T22 - Retrieval Karşılaştırma Deneyi

CLI: python -m eval.retrieval_experiment --retrieval-mode dense --output results/retrieval_experiment.json

Runs 50 test queries with dense-only vs hybrid retrieval and saves RAGAS metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("dense", "hybrid")
DEFAULT_N_QUERIES = 50


def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_retrieval_experiment(
    queries: list[dict[str, Any]],
    retrieval_mode: str,
    retriever,
    chain,
) -> list[dict[str, Any]]:
    records = []
    for item in queries:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        try:
            output = chain.run(question, retrieval_mode=retrieval_mode)
            answer = output.get("answer", "")
            contexts = output.get("contexts", [])
        except Exception as exc:
            logger.warning("Experiment failed for question %r: %s", question[:50], exc)
            answer = ""
            contexts = []
        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
    return records


def compute_answer_relevancy(records: list[dict[str, Any]]) -> float | None:
    from eval.benchmark_runner import compute_ragas_metrics_safe

    metrics = compute_ragas_metrics_safe(records)
    return metrics.get("answer_relevancy")


def save_results(results: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Retrieval Comparison Experiment")
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        required=True,
        help="Retrieval mode: dense or hybrid",
    )
    parser.add_argument("--queries", default="data/test_queries.json", help="Path to test queries JSON")
    parser.add_argument("--output", default="results/retrieval_experiment.json", help="Output JSON path")
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    args = parser.parse_args()

    queries = load_queries(args.queries)[: args.n_queries]
    logger.info("Running %d queries with mode=%s", len(queries), args.retrieval_mode)

    from eval.chains import get_retrieval_chain

    chain = get_retrieval_chain(args.retrieval_mode)
    records = run_retrieval_experiment(
        queries=queries,
        retrieval_mode=args.retrieval_mode,
        retriever=None,
        chain=chain,
    )
    score = compute_answer_relevancy(records)

    result = ExperimentResult(
        experiment_name=f"retrieval_{args.retrieval_mode}",
        experiment_type="retrieval",
        retriever_type=args.retrieval_mode,
        answer_relevancy=score,
        n_questions=len(queries),
    )
    result_dict = result.to_dict()
    # Backward-compat keys: dashboard hâlâ bunlara bakıyor
    result_dict["retrieval_mode"] = args.retrieval_mode
    result_dict["n_queries"] = len(queries)
    save_results(result_dict, args.output)
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
