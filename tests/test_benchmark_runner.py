import json
import os
from unittest.mock import patch

import pytest

from eval.benchmark_runner import (
    CHUNK_SIZES,
    RETRIEVER_TYPES,
    MockChain,
    RetrieverWrapper,
    TopKWrapper,
    compute_ragas_metrics_safe,
    run_chunk_comparison,
    run_full_suite,
    run_rag_comparison,
    run_retriever_comparison,
    run_single_experiment,
    save_benchmark_results,
)
from eval.metrics_schema import ExperimentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dataset(n: int = 3) -> list[dict]:
    return [
        {"question": f"Q{i}", "ground_truth": f"A{i}"}
        for i in range(n)
    ]


_MOCK_METRICS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.70,
}


def _patch_ragas(target="eval.benchmark_runner.compute_ragas_metrics_safe"):
    return patch(target, return_value=dict(_MOCK_METRICS))


# ---------------------------------------------------------------------------
# MockChain
# ---------------------------------------------------------------------------

def test_mock_chain_run_returns_required_keys():
    chain = MockChain()
    result = chain.run("What is RAG?")
    assert {"answer", "contexts", "sources", "rewrite_count", "latency_seconds"}.issubset(result)


def test_mock_chain_custom_answer():
    chain = MockChain(answer="Custom answer", rewrite_count=2)
    result = chain.run("q")
    assert result["answer"] == "Custom answer"
    assert result["rewrite_count"] == 2


def test_mock_chain_contexts_contain_query():
    chain = MockChain()
    result = chain.run("attention mechanism")
    assert any("attention" in c for c in result["contexts"])


# ---------------------------------------------------------------------------
# RetrieverWrapper
# ---------------------------------------------------------------------------

def test_retriever_wrapper_passes_mode():
    from unittest.mock import MagicMock
    inner = MagicMock()
    inner.run.return_value = {"answer": "a", "contexts": [], "sources": [],
                              "rewrite_count": 0, "latency_seconds": 0.1}
    wrapped = RetrieverWrapper(inner, retrieval_mode="dense")
    wrapped.run("q")
    inner.run.assert_called_once_with("q", retrieval_mode="dense")


def test_retriever_wrapper_different_modes():
    from unittest.mock import MagicMock
    inner = MagicMock()
    inner.run.return_value = {"answer": "a", "contexts": [], "sources": [],
                              "rewrite_count": 0, "latency_seconds": 0.0}
    for mode in ("bm25", "dense", "hybrid"):
        RetrieverWrapper(inner, retrieval_mode=mode).run("q")
    calls = [call.kwargs["retrieval_mode"] for call in inner.run.call_args_list]
    assert calls == ["bm25", "dense", "hybrid"]


# ---------------------------------------------------------------------------
# TopKWrapper
# ---------------------------------------------------------------------------

def test_topk_wrapper_passes_top_k():
    from unittest.mock import MagicMock
    inner = MagicMock()
    inner.run.return_value = {"answer": "a", "contexts": [], "sources": [],
                              "rewrite_count": 0, "latency_seconds": 0.0}
    wrapped = TopKWrapper(inner, top_k=10)
    wrapped.run("q")
    inner.run.assert_called_once_with("q", top_k=10)


# ---------------------------------------------------------------------------
# compute_ragas_metrics_safe
# ---------------------------------------------------------------------------

def test_safe_metrics_empty_records():
    result = compute_ragas_metrics_safe([])
    assert result == {"faithfulness": None, "answer_relevancy": None,
                      "context_precision": None, "context_recall": None}


def test_safe_metrics_ragas_failure():
    from unittest.mock import MagicMock
    mock_ragas = MagicMock()
    mock_ragas.evaluate.side_effect = Exception("no key")
    with patch.dict("sys.modules", {
        "ragas": mock_ragas,
        "ragas.metrics": MagicMock(),
        "datasets": MagicMock(),
    }):
        result = compute_ragas_metrics_safe([
            {"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}
        ])
    assert result["faithfulness"] is None


# ---------------------------------------------------------------------------
# run_single_experiment
# ---------------------------------------------------------------------------

def test_run_single_experiment_returns_experiment_result():
    with _patch_ragas():
        result = run_single_experiment(
            MockChain(), _dataset(),
            experiment_name="test_exp",
            experiment_type="comparison",
        )
    assert isinstance(result, ExperimentResult)
    assert result.experiment_name == "test_exp"


def test_run_single_experiment_all_schema_fields():
    with _patch_ragas():
        result = run_single_experiment(
            MockChain(), _dataset(5),
            experiment_name="test",
            experiment_type="retrieval",
            model_name="gpt-4o-mini",
            retriever_type="hybrid",
            chunk_size=512,
            top_k=5,
        )
    assert result.model_name == "gpt-4o-mini"
    assert result.retriever_type == "hybrid"
    assert result.chunk_size == 512
    assert result.top_k == 5
    assert result.n_questions == 5
    assert result.faithfulness == 0.80
    assert result.answer_relevancy == 0.85
    assert result.context_precision == 0.75
    assert result.context_recall == 0.70


def test_run_single_experiment_avg_latency():
    with _patch_ragas():
        result = run_single_experiment(
            MockChain(latency=0.1), _dataset(3),
            experiment_name="t", experiment_type="t",
        )
    assert result.avg_latency_seconds == pytest.approx(0.1, abs=1e-3)


def test_run_single_experiment_avg_rewrite_count():
    with _patch_ragas():
        result = run_single_experiment(
            MockChain(rewrite_count=2), _dataset(4),
            experiment_name="t", experiment_type="t",
        )
    assert result.avg_rewrite_count == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# save_benchmark_results
# ---------------------------------------------------------------------------

def test_save_benchmark_results_creates_file(tmp_path):
    results = [
        ExperimentResult(experiment_name="a", experiment_type="comparison"),
        ExperimentResult(experiment_name="b", experiment_type="comparison"),
    ]
    path = str(tmp_path / "out" / "bench.json")
    save_benchmark_results(results, path)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["n_experiments"] == 2
    assert len(data["experiments"]) == 2


def test_save_benchmark_results_extra_fields(tmp_path):
    path = str(tmp_path / "bench.json")
    save_benchmark_results(
        [ExperimentResult(experiment_name="x", experiment_type="t")],
        path,
        extra={"benchmark": "test_bench"},
    )
    with open(path) as f:
        data = json.load(f)
    assert data["benchmark"] == "test_bench"


# ---------------------------------------------------------------------------
# Benchmark 1 — run_rag_comparison
# ---------------------------------------------------------------------------

def test_rag_comparison_returns_two_results(tmp_path):
    with _patch_ragas():
        results = run_rag_comparison(
            _dataset(), output_path=str(tmp_path / "rag.json")
        )
    assert len(results) == 2
    names = {r.experiment_name for r in results}
    assert names == {"standard_rag", "auto_rag"}


def test_rag_comparison_saves_json(tmp_path):
    path = str(tmp_path / "rag.json")
    with _patch_ragas():
        run_rag_comparison(_dataset(), output_path=path)
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Benchmark 2 — run_retriever_comparison
# ---------------------------------------------------------------------------

def test_retriever_comparison_returns_one_per_mode(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    assert len(results) == len(RETRIEVER_TYPES)
    types = {r.retriever_type for r in results}
    assert types == set(RETRIEVER_TYPES)


def test_retriever_comparison_experiment_types(tmp_path):
    with _patch_ragas():
        results = run_retriever_comparison(
            _dataset(), output_path=str(tmp_path / "ret.json")
        )
    assert all(r.experiment_type == "retrieval" for r in results)


# ---------------------------------------------------------------------------
# Benchmark 3 — run_chunk_comparison
# ---------------------------------------------------------------------------

def test_chunk_comparison_returns_one_per_size(tmp_path):
    with _patch_ragas():
        results = run_chunk_comparison(
            _dataset(), output_path=str(tmp_path / "chunk.json")
        )
    assert len(results) == len(CHUNK_SIZES)
    sizes = {r.chunk_size for r in results}
    assert sizes == set(CHUNK_SIZES)


def test_chunk_comparison_experiment_type(tmp_path):
    with _patch_ragas():
        results = run_chunk_comparison(
            _dataset(), output_path=str(tmp_path / "chunk.json")
        )
    assert all(r.experiment_type == "chunking" for r in results)


# ---------------------------------------------------------------------------
# run_full_suite
# ---------------------------------------------------------------------------

def test_full_suite_runs_all_benchmarks(tmp_path):
    with _patch_ragas():
        suite = run_full_suite(_dataset(), output_dir=str(tmp_path))
    assert set(suite.keys()) == {"rag_comparison", "retriever", "chunk"}
    assert len(suite["rag_comparison"]) == 2
    assert len(suite["retriever"]) == 3
    assert len(suite["chunk"]) == 4
