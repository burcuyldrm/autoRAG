import pytest
from eval.metrics_schema import ExperimentResult


def test_required_fields_only():
    r = ExperimentResult(experiment_name="test", experiment_type="comparison")
    assert r.experiment_name == "test"
    assert r.experiment_type == "comparison"
    assert r.faithfulness is None
    assert r.n_questions == 0
    assert r.rewrite_count == 0


def test_optional_fields_accept_none():
    r = ExperimentResult(
        experiment_name="x",
        experiment_type="retrieval",
        model_name=None,
        provider=None,
        retriever_type=None,
        chunk_size=None,
        top_k=None,
        faithfulness=None,
        answer_relevancy=None,
        context_precision=None,
        context_recall=None,
        latency_seconds=None,
        avg_latency_seconds=None,
        token_usage=None,
        cost_estimate=None,
    )
    assert r.faithfulness is None
    assert r.chunk_size is None


def test_to_dict_contains_all_fields():
    r = ExperimentResult(
        experiment_name="eval_run",
        experiment_type="comparison",
        faithfulness=0.85,
        answer_relevancy=0.90,
        n_questions=10,
    )
    d = r.to_dict()
    assert d["experiment_name"] == "eval_run"
    assert d["faithfulness"] == 0.85
    assert d["answer_relevancy"] == 0.90
    assert d["n_questions"] == 10
    assert "context_recall" in d
    assert "rewrite_count" in d


def test_to_dict_has_timestamp():
    r = ExperimentResult(experiment_name="t", experiment_type="retrieval")
    d = r.to_dict()
    assert "timestamp" in d
    assert len(d["timestamp"]) == 19  # "YYYY-MM-DDTHH:MM:SS"


def test_from_dict_roundtrip():
    original = ExperimentResult(
        experiment_name="retrieval_hybrid",
        experiment_type="retrieval",
        retriever_type="hybrid",
        chunk_size=512,
        top_k=5,
        answer_relevancy=0.78,
        n_questions=50,
    )
    d = original.to_dict()
    restored = ExperimentResult.from_dict(d)
    assert restored.experiment_name == original.experiment_name
    assert restored.retriever_type == original.retriever_type
    assert restored.answer_relevancy == original.answer_relevancy
    assert restored.chunk_size == original.chunk_size


def test_from_dict_ignores_unknown_keys():
    data = {
        "experiment_name": "x",
        "experiment_type": "chunking",
        "unknown_key": "should_be_ignored",
        "another_extra": 999,
    }
    r = ExperimentResult.from_dict(data)
    assert r.experiment_name == "x"
    assert not hasattr(r, "unknown_key")


def test_chunking_experiment_type():
    r = ExperimentResult(
        experiment_name="chunking_512",
        experiment_type="chunking",
        chunk_size=512,
        answer_relevancy=0.72,
        n_questions=30,
    )
    assert r.chunk_size == 512
    assert r.experiment_type == "chunking"


def test_rewrite_stats_defaults():
    r = ExperimentResult(experiment_name="x", experiment_type="comparison")
    assert r.rewrite_count == 0
    assert r.avg_rewrite_count == 0.0
