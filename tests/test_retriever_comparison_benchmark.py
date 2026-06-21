"""Tests for Retriever Comparison benchmark (Benchmark 2)."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from eval.benchmark_runner import MockChain, RETRIEVER_TYPES, run_retriever_comparison
from eval.metrics_schema import ExperimentResult


_MOCK_METRICS = {
    "faithfulness": 0.78,
    "answer_relevancy": 0.82,
    "context_precision": 0.74,
    "context_recall": 0.68,
}


def _patch_ragas():
    return patch(
        "eval.benchmark_runner.compute_ragas_metrics_safe",
        return_value=dict(_MOCK_METRICS),
    )


def _dataset(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"q{i:03d}",
            "question": f"Question {i}?",
            "ground_truth": f"Answer {i}.",
            "expected_source": "paper",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_retriever_comparison_three_experiments(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    assert len(results) == 3


def test_retriever_comparison_covers_all_modes(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    modes = {r.retriever_type for r in results}
    assert modes == {"bm25", "dense", "hybrid"}


def test_retriever_comparison_json_has_experiments(tmp_path):
    path = str(tmp_path / "ret.json")
    with _patch_ragas():
        run_retriever_comparison(_dataset(), output_path=path)
    with open(path) as f:
        data = json.load(f)
    assert "experiments" in data
    assert len(data["experiments"]) == 3


def test_retriever_comparison_json_benchmark_name(tmp_path):
    path = str(tmp_path / "ret.json")
    with _patch_ragas():
        run_retriever_comparison(_dataset(30), output_path=path, n_questions=30,
                                  dataset_path="data/qa_dataset.json")
    with open(path) as f:
        data = json.load(f)
    assert data.get("benchmark_name") == "retriever_comparison"


def test_retriever_comparison_json_n_questions(tmp_path):
    path = str(tmp_path / "ret.json")
    with _patch_ragas():
        run_retriever_comparison(_dataset(30), output_path=path, n_questions=30)
    with open(path) as f:
        data = json.load(f)
    assert data.get("n_questions") == 30


def test_retriever_comparison_json_dataset_path(tmp_path):
    path = str(tmp_path / "ret.json")
    with _patch_ragas():
        run_retriever_comparison(_dataset(), output_path=path,
                                  dataset_path="data/qa_dataset.json")
    with open(path) as f:
        data = json.load(f)
    assert data.get("dataset_path") == "data/qa_dataset.json"


def test_retriever_comparison_experiment_type(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    assert all(r.experiment_type == "retrieval" for r in results)


def test_retriever_comparison_metrics_set(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    for r in results:
        assert r.faithfulness == pytest.approx(0.78)
        assert r.answer_relevancy == pytest.approx(0.82)


def test_retriever_comparison_experiment_fields_in_json(tmp_path):
    path = str(tmp_path / "ret.json")
    with _patch_ragas():
        run_retriever_comparison(_dataset(), output_path=path)
    with open(path) as f:
        data = json.load(f)
    for exp in data["experiments"]:
        assert "experiment_name" in exp
        assert "retriever_type" in exp
        assert "experiment_type" in exp
        assert "faithfulness" in exp
        assert "answer_relevancy" in exp
        assert "n_questions" in exp


def test_retriever_comparison_bm25_experiment(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    bm25 = next(r for r in results if r.retriever_type == "bm25")
    assert bm25.experiment_name == "retriever_bm25"


def test_retriever_comparison_custom_modes(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(),
            retriever_types=("bm25", "dense"),
            output_path=str(tmp_path / "ret.json"),
        )
    assert len(results) == 2
    modes = {r.retriever_type for r in results}
    assert modes == {"bm25", "dense"}
