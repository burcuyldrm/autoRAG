"""
Chunking Boyutu Karşılaştırma Deneyi

Farklı chunk boyutlarının RAGAS answer_relevancy metriğine etkisini ölçer.

CLI:
    python -m eval.chunking_experiment --chunk-size 512 --dataset data/qa.json
    python -m eval.chunking_experiment --chunk-size 1024 --dataset data/qa.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)

VALID_CHUNK_SIZES = (256, 512, 1024, 2048)


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_chunking_experiment(
    queries: list[dict[str, Any]],
    chunk_size: int,
    chain,
) -> list[dict[str, Any]]:
    """Verilen chunk boyutuyla zinciri çalıştırır, ham kayıtları döndürür."""
    records = []
    for item in queries:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        try:
            output = chain.run(question, chunk_size=chunk_size)
            answer = output.get("answer", "")
            contexts = output.get("contexts", [])
        except Exception as exc:
            logger.warning("Chain failed for %r: %s", question[:50], exc)
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
    parser = argparse.ArgumentParser(description="Chunking Size Experiment")
    parser.add_argument("--chunk-size", type=int, required=True, choices=VALID_CHUNK_SIZES)
    parser.add_argument("--dataset", default="data/test_queries.json")
    parser.add_argument("--output", default="results/chunking_experiment.json")
    parser.add_argument("--retrieval-mode", default="hybrid", choices=("dense", "hybrid"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = load_dataset(args.dataset)
    logger.info("Running %d queries with chunk_size=%d", len(queries), args.chunk_size)

    from eval.chains import get_retrieval_chain

    chain = get_retrieval_chain(args.retrieval_mode)
    records = run_chunking_experiment(queries, args.chunk_size, chain)
    score = compute_answer_relevancy(records)

    result = ExperimentResult(
        experiment_name=f"chunking_{args.chunk_size}",
        experiment_type="chunking",
        retriever_type=args.retrieval_mode,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        answer_relevancy=score,
        n_questions=len(queries),
    )
    result_dict = result.to_dict()
    result_dict["chunk_size_label"] = str(args.chunk_size)
    save_results(result_dict, args.output)
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
