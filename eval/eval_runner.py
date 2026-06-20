"""
RAGAS Evaluation Harness

Runs Standard RAG and Auto-RAG (LangGraph) on a Q&A dataset, computes
RAGAS metrics: faithfulness, answer_relevancy, context_precision.

Usage:
    python -m eval.eval_runner --dataset path/to/qa.json --output results/eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Protocol

from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)


class RAGChain(Protocol):
    def run(self, query: str) -> dict[str, Any]: ...


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_chain_on_dataset(
    chain: RAGChain,
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns list of {question, answer, contexts, ground_truth, rewrite_count, latency_seconds, grade_confidence}."""
    records = []
    for item in dataset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        try:
            output = chain.run(question)
            answer = output.get("answer", "")
            contexts = output.get("contexts", [])
            rewrite_count = output.get("rewrite_count", 0)
            latency = output.get("latency_seconds", 0.0)
            grade_confidence = output.get("grade_confidence", None)
        except Exception as exc:
            logger.warning("Chain failed for question %r: %s", question[:50], exc)
            answer = ""
            contexts = []
            rewrite_count = 0
            latency = 0.0
            grade_confidence = None
        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "rewrite_count": rewrite_count,
                "latency_seconds": latency,
                "grade_confidence": grade_confidence,
            }
        )
    return records


def compute_ragas_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute RAGAS metrics using local Ollama LLM + sentence-transformers (no API key)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from app.llm_factory import get_ragas_embeddings, get_ragas_llm

    ragas_llm = get_ragas_llm(fast=True)
    ragas_embeddings = get_ragas_embeddings()

    ds = Dataset.from_list(records)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }


def save_results(results: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def run_evaluation(
    standard_chain: RAGChain,
    autorag_chain: RAGChain,
    dataset: list[dict[str, Any]],
    output_path: str = "results/eval.json",
) -> dict[str, Any]:
    logger.info("Running Standard RAG on %d questions...", len(dataset))
    standard_records = run_chain_on_dataset(standard_chain, dataset)
    standard_metrics = compute_ragas_metrics(standard_records)

    logger.info("Running Auto-RAG on %d questions...", len(dataset))
    autorag_records = run_chain_on_dataset(autorag_chain, dataset)
    autorag_metrics = compute_ragas_metrics(autorag_records)

    def _avg(records: list[dict], key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else default

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    standard_result = ExperimentResult(
        experiment_name="standard_rag",
        experiment_type="comparison",
        timestamp=ts,
        n_questions=len(dataset),
        avg_latency_seconds=_avg(standard_records, "latency_seconds"),
        **standard_metrics,
    )
    autorag_result = ExperimentResult(
        experiment_name="auto_rag",
        experiment_type="comparison",
        timestamp=ts,
        n_questions=len(dataset),
        avg_rewrite_count=_avg(autorag_records, "rewrite_count"),
        avg_latency_seconds=_avg(autorag_records, "latency_seconds"),
        **autorag_metrics,
    )
    results = {
        "timestamp": ts,
        "experiment_type": "comparison",
        "n_questions": len(dataset),
        "standard_rag": standard_metrics,
        "auto_rag": {
            **autorag_metrics,
            "avg_rewrite_count": autorag_result.avg_rewrite_count,
            "avg_latency_seconds": autorag_result.avg_latency_seconds,
        },
        "experiments": [standard_result.to_dict(), autorag_result.to_dict()],
    }
    save_results(results, output_path)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness")
    parser.add_argument("--dataset", required=True, help="Path to Q&A JSON dataset")
    parser.add_argument("--output", default="results/eval.json", help="Output JSON path")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d questions from %s", len(dataset), args.dataset)

    # Chains are injected at runtime; this entry-point shows the wiring
    from eval.chains import get_autorag_chain, get_standard_chain

    results = run_evaluation(
        standard_chain=get_standard_chain(),
        autorag_chain=get_autorag_chain(),
        dataset=dataset,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
