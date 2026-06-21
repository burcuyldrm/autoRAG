import json
import os
from unittest.mock import patch

import pytest

from eval.model_comparison import DEFAULT_MODELS, _build_chain, run_model_comparison
from eval.benchmark_runner import MockChain
from eval.metrics_schema import ExperimentResult


def _dataset(n: int = 2) -> list[dict]:
    return [{"question": f"Q{i}", "ground_truth": f"A{i}"} for i in range(n)]


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


# ---------------------------------------------------------------------------
# _build_chain
# ---------------------------------------------------------------------------

def test_build_chain_returns_mock_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chain = _build_chain("gpt-4o-mini")
    assert isinstance(chain, MockChain)


def test_build_chain_mock_answer_contains_model_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chain = _build_chain("gpt-4o")
    result = chain.run("q")
    assert "gpt-4o" in result["answer"]


# ---------------------------------------------------------------------------
# run_model_comparison
# ---------------------------------------------------------------------------

def test_returns_one_result_per_model(tmp_path):
    models = ["gpt-4o-mini", "gpt-4o"]
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(), model_names=models,
            output_path=str(tmp_path / "models.json"),
        )
    assert len(results) == 2
    assert {r.model_name for r in results} == set(models)


def test_experiment_type_is_model_comparison(tmp_path):
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(), model_names=["gpt-4o-mini"],
            output_path=str(tmp_path / "m.json"),
        )
    assert results[0].experiment_type == "model_comparison"


def test_retriever_type_propagated(tmp_path):
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(), model_names=["gpt-4o-mini"],
            retriever_type="dense",
            output_path=str(tmp_path / "m.json"),
        )
    assert results[0].retriever_type == "dense"


def test_top_k_propagated(tmp_path):
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(), model_names=["gpt-4o-mini"],
            top_k=10,
            output_path=str(tmp_path / "m.json"),
        )
    assert results[0].top_k == 10


def test_ragas_metrics_present(tmp_path):
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(), model_names=["gpt-4o-mini"],
            output_path=str(tmp_path / "m.json"),
        )
    r = results[0]
    assert r.faithfulness == 0.80
    assert r.answer_relevancy == 0.85
    assert r.context_precision == 0.75
    assert r.context_recall == 0.70


def test_saves_json_file(tmp_path):
    path = str(tmp_path / "models.json")
    with _patch_ragas():
        run_model_comparison(
            _dataset(), model_names=["gpt-4o-mini"],
            output_path=path,
        )
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert "experiments" in data
    assert data["n_experiments"] == 1


def test_json_contains_model_names(tmp_path):
    models = ["gpt-4o-mini", "gpt-4o"]
    path = str(tmp_path / "models.json")
    with _patch_ragas():
        run_model_comparison(_dataset(), model_names=models, output_path=path)
    with open(path) as f:
        data = json.load(f)
    assert data["models"] == models


def test_n_questions_correct(tmp_path):
    with _patch_ragas():
        results = run_model_comparison(
            _dataset(5), model_names=["gpt-4o-mini"],
            output_path=str(tmp_path / "m.json"),
        )
    assert results[0].n_questions == 5


def test_default_models_list_non_empty():
    assert len(DEFAULT_MODELS) >= 2
