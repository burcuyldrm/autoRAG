import json
import os
from unittest.mock import MagicMock, patch

import pytest

from eval.topk_experiment import DEFAULT_TOP_K_VALUES, run_topk_experiment
from eval.benchmark_runner import MockChain, TopKWrapper
from eval.metrics_schema import ExperimentResult


def _dataset(n: int = 2) -> list[dict]:
    return [{"question": f"Q{i}", "ground_truth": f"A{i}"} for i in range(n)]


_MOCK_METRICS = {
    "faithfulness": 0.78,
    "answer_relevancy": 0.82,
    "context_precision": 0.71,
    "context_recall": 0.68,
}


def _patch_ragas():
    return patch(
        "eval.benchmark_runner.compute_ragas_metrics_safe",
        return_value=dict(_MOCK_METRICS),
    )


# ---------------------------------------------------------------------------
# TopKWrapper (used by topk_experiment)
# ---------------------------------------------------------------------------

def test_topk_wrapper_injects_top_k():
    inner = MagicMock()
    inner.run.return_value = {
        "answer": "a", "contexts": [], "sources": [],
        "rewrite_count": 0, "latency_seconds": 0.0,
    }
    wrapped = TopKWrapper(inner, top_k=7)
    wrapped.run("query")
    inner.run.assert_called_once_with("query", top_k=7)


# ---------------------------------------------------------------------------
# run_topk_experiment
# ---------------------------------------------------------------------------

def test_returns_one_result_per_k(tmp_path):
    k_vals = (3, 5, 10)
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=k_vals,
            output_path=str(tmp_path / "topk.json"),
        )
    assert len(results) == 3
    assert {r.top_k for r in results} == set(k_vals)


def test_default_top_k_values(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(),
            output_path=str(tmp_path / "topk.json"),
        )
    assert len(results) == len(DEFAULT_TOP_K_VALUES)
    assert {r.top_k for r in results} == set(DEFAULT_TOP_K_VALUES)


def test_experiment_type_is_topk(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=(5,),
            output_path=str(tmp_path / "topk.json"),
        )
    assert results[0].experiment_type == "topk"


def test_experiment_name_contains_k(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=(3, 10),
            output_path=str(tmp_path / "topk.json"),
        )
    names = {r.experiment_name for r in results}
    assert "topk_3" in names
    assert "topk_10" in names


def test_retriever_type_propagated(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=(5,),
            retriever_type="bm25",
            output_path=str(tmp_path / "topk.json"),
        )
    assert results[0].retriever_type == "bm25"


def test_ragas_metrics_present(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=(5,),
            output_path=str(tmp_path / "topk.json"),
        )
    r = results[0]
    assert r.faithfulness == 0.78
    assert r.answer_relevancy == 0.82
    assert r.context_precision == 0.71
    assert r.context_recall == 0.68


def test_saves_json_file(tmp_path):
    path = str(tmp_path / "topk.json")
    with _patch_ragas():
        run_topk_experiment(_dataset(), top_k_values=(3, 5), output_path=path)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["n_experiments"] == 2
    assert data["benchmark"] == "topk_comparison"
    assert data["top_k_values"] == [3, 5]


def test_uses_mock_chain_by_default(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(), top_k_values=(5,),
            base_chain=None,
            output_path=str(tmp_path / "topk.json"),
        )
    assert len(results) == 1


def test_custom_base_chain_used(tmp_path):
    custom = MagicMock()
    custom.run.return_value = {
        "answer": "custom", "contexts": ["ctx"], "sources": [],
        "rewrite_count": 1, "latency_seconds": 0.2,
    }
    with _patch_ragas():
        run_topk_experiment(
            _dataset(2), top_k_values=(5,),
            base_chain=custom,
            output_path=str(tmp_path / "topk.json"),
        )
    assert custom.run.called


def test_n_questions_correct(tmp_path):
    with _patch_ragas():
        results = run_topk_experiment(
            _dataset(7), top_k_values=(5,),
            output_path=str(tmp_path / "topk.json"),
        )
    assert results[0].n_questions == 7
