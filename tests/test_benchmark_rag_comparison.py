"""Tests for Standard RAG vs Auto-RAG benchmark (Benchmark 1)."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from eval.benchmark_runner import MockChain, run_rag_comparison, save_benchmark_results
from eval.metrics_schema import ExperimentResult


_MOCK_METRICS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.70,
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
            "question": f"What is topic {i}?",
            "ground_truth": f"Answer about topic {i}.",
            "expected_source": "paper",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_rag_comparison_returns_two_experiments(tmp_path):
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(), output_path=str(tmp_path / "rag.json")
        )
    assert len(results) == 2


def test_rag_comparison_experiment_names(tmp_path):
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(), output_path=str(tmp_path / "rag.json")
        )
    names = {r.experiment_name for r in results}
    assert "standard_rag" in names
    assert "auto_rag" in names


def test_rag_comparison_json_has_experiments_key(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(), output_path=path)
    with open(path) as f:
        data = json.load(f)
    assert "experiments" in data
    assert len(data["experiments"]) == 2


def test_rag_comparison_json_has_benchmark_name(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(30), output_path=path, n_questions=30,
                           dataset_path="data/qa_dataset.json")
    with open(path) as f:
        data = json.load(f)
    assert data.get("benchmark_name") == "standard_vs_autorag"


def test_rag_comparison_json_has_n_questions(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(30), output_path=path, n_questions=30)
    with open(path) as f:
        data = json.load(f)
    assert data.get("n_questions") == 30


def test_rag_comparison_json_has_dataset_path(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(), output_path=path,
                           dataset_path="data/qa_dataset.json")
    with open(path) as f:
        data = json.load(f)
    assert data.get("dataset_path") == "data/qa_dataset.json"


def test_rag_comparison_experiment_fields(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(), output_path=path)
    with open(path) as f:
        data = json.load(f)
    exp = data["experiments"][0]
    for field in ("experiment_name", "experiment_type", "faithfulness",
                  "answer_relevancy", "context_precision", "context_recall",
                  "avg_latency_seconds", "avg_rewrite_count", "n_questions"):
        assert field in exp, f"Missing field: {field}"


def test_rag_comparison_standard_rewrite_count_zero(tmp_path):
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(),
            standard_chain=MockChain(rewrite_count=0),
            autorag_chain=MockChain(rewrite_count=1),
            output_path=str(tmp_path / "rag.json"),
        )
    standard = next(r for r in results if r.experiment_name == "standard_rag")
    autorag = next(r for r in results if r.experiment_name == "auto_rag")
    assert standard.avg_rewrite_count == pytest.approx(0.0)
    assert autorag.avg_rewrite_count == pytest.approx(1.0)


def test_rag_comparison_metrics_populated(tmp_path):
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(), output_path=str(tmp_path / "rag.json")
        )
    for r in results:
        assert r.faithfulness == pytest.approx(0.80)
        assert r.answer_relevancy == pytest.approx(0.85)


def test_rag_comparison_custom_chains(tmp_path):
    """Custom chains produce different avg_rewrite_counts."""
    std = MockChain(rewrite_count=0, answer="Standard answer")
    auto = MockChain(rewrite_count=2, answer="Auto answer")
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(), standard_chain=std, autorag_chain=auto,
            output_path=str(tmp_path / "rag.json"),
        )
    auto_result = next(r for r in results if r.experiment_name == "auto_rag")
    assert auto_result.avg_rewrite_count == pytest.approx(2.0)
