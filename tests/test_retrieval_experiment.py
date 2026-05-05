import json
import os
from unittest.mock import MagicMock, patch

import pytest

from eval.retrieval_experiment import (
    load_queries,
    run_retrieval_experiment,
    save_results,
)


def _make_queries(n: int = 3) -> list[dict]:
    return [
        {"question": f"Query {i}", "ground_truth": f"Answer {i}"}
        for i in range(n)
    ]


def _make_chain(answer: str = "answer", contexts=None):
    chain = MagicMock()
    chain.run.return_value = {"answer": answer, "contexts": contexts or ["ctx"]}
    return chain


def test_load_queries(tmp_path):
    data = [{"question": "q1", "ground_truth": "a1"}]
    p = tmp_path / "queries.json"
    p.write_text(json.dumps(data))
    loaded = load_queries(str(p))
    assert loaded == data


def test_run_retrieval_experiment_dense():
    chain = _make_chain("relevant answer", ["context A"])
    records = run_retrieval_experiment(
        queries=_make_queries(3),
        retrieval_mode="dense",
        retriever=None,
        chain=chain,
    )
    assert len(records) == 3
    assert all(r["answer"] == "relevant answer" for r in records)


def test_run_retrieval_experiment_hybrid():
    chain = _make_chain("hybrid answer", ["ctx1", "ctx2"])
    records = run_retrieval_experiment(
        queries=_make_queries(2),
        retrieval_mode="hybrid",
        retriever=None,
        chain=chain,
    )
    assert len(records) == 2
    assert records[0]["contexts"] == ["ctx1", "ctx2"]


def test_run_retrieval_experiment_chain_error():
    chain = MagicMock()
    chain.run.side_effect = Exception("network error")
    records = run_retrieval_experiment(
        queries=_make_queries(1),
        retrieval_mode="dense",
        retriever=None,
        chain=chain,
    )
    assert records[0]["answer"] == ""
    assert records[0]["contexts"] == []


def test_save_results_creates_file(tmp_path):
    results = {"retrieval_mode": "dense", "answer_relevancy": 0.75}
    output = str(tmp_path / "sub" / "result.json")
    save_results(results, output)
    assert os.path.exists(output)
    with open(output) as f:
        data = json.load(f)
    assert data["answer_relevancy"] == 0.75
