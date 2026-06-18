"""
Model Comparison Experiment  (Benchmark 5)

Compare different LLM models on the same RAG pipeline. Falls back to MockChain
when OPENAI_API_KEY is not set, so the experiment always produces output.

CLI:
    python -m eval.model_comparison --dataset data/qa.json
    python -m eval.model_comparison --dataset data/qa.json --models gpt-4o-mini gpt-4o
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

from eval.benchmark_runner import MockChain, run_single_experiment, save_benchmark_results
from eval.metrics_schema import ExperimentResult

logger = logging.getLogger(__name__)

DEFAULT_MODELS: list[str] = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def _build_chain(
    model_name: str,
    vectorstore: Any = None,
    bm25_retriever: Any = None,
) -> Any:
    """Return a StandardRAGChain for model_name, or MockChain if no API key."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set — using MockChain for model %r.", model_name)
        return MockChain(answer=f"[Mock] Answer from {model_name}.", latency=0.05)

    try:
        from langchain_openai import ChatOpenAI
        from app.standard_rag import StandardRAGChain

        llm = ChatOpenAI(model=model_name, temperature=0)
        return StandardRAGChain(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            llm=llm,
        )
    except Exception as exc:
        logger.warning("Could not build chain for %r (%s) — falling back to MockChain.", model_name, exc)
        return MockChain(answer=f"[Fallback] Answer from {model_name}.", latency=0.05)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_model_comparison(
    dataset: list[dict[str, Any]],
    model_names: list[str] = DEFAULT_MODELS,
    retriever_type: str = "hybrid",
    top_k: int = 5,
    vectorstore: Any = None,
    bm25_retriever: Any = None,
    output_path: str = "results/benchmark_model_comparison.json",
) -> list[ExperimentResult]:
    """
    Run each model on the dataset and compare RAGAS metrics.

    Parameters
    ----------
    dataset : list[dict]
        Q&A dataset with 'question' and 'ground_truth' keys.
    model_names : list[str]
        OpenAI model identifiers (e.g. ["gpt-4o-mini", "gpt-4o"]).
    retriever_type : str
        Retrieval mode used for all models.
    top_k : int
        Number of chunks retrieved per query.
    vectorstore / bm25_retriever : optional
        Injected retrievers; None uses MockChain regardless of API key.
    """
    logger.info("Benchmark 5 — Model comparison: %s", model_names)
    results: list[ExperimentResult] = []

    for model in model_names:
        chain = _build_chain(model, vectorstore=vectorstore, bm25_retriever=bm25_retriever)
        result = run_single_experiment(
            chain, dataset,
            experiment_name=f"model_{model.replace('-', '_').replace('.', '_')}",
            experiment_type="model_comparison",
            model_name=model,
            retriever_type=retriever_type,
            top_k=top_k,
        )
        results.append(result)
        logger.info(
            "  %-20s answer_relevancy=%-6s  latency=%.2fs",
            model, result.answer_relevancy, result.avg_latency_seconds or 0,
        )

    save_benchmark_results(
        results, output_path,
        extra={"benchmark": "model_comparison", "models": model_names},
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Model Comparison Experiment")
    parser.add_argument("--dataset", required=True, help="Path to Q&A JSON dataset")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--retriever-type", default="hybrid", choices=("bm25", "dense", "hybrid"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="results/benchmark_model_comparison.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    results = run_model_comparison(
        dataset,
        model_names=args.models,
        retriever_type=args.retriever_type,
        top_k=args.top_k,
        output_path=args.output,
    )
    print(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()
