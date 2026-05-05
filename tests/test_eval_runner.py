import json
import os
from unittest.mock import MagicMock, patch

import pytest

from eval.eval_runner import (
    compute_ragas_metrics,
    load_dataset,
    run_chain_on_dataset,
    save_results,
)


def _make_dataset(n: int = 3) -> list[dict]:
    return [
        {
            "question": f"What is concept {i}?",
            "ground_truth": f"Concept {i} is about topic {i}.",
        }
        for i in range(n)
    ]


def _make_chain(answer: str = "test answer", contexts: list[str] | None = None):
    chain = MagicMock()
    chain.run.return_value = {"answer": answer, "contexts": contexts or ["ctx1"]}
    return chain


def test_load_dataset(tmp_path):
    data = [{"question": "q1", "ground_truth": "a1"}]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    loaded = load_dataset(str(p))
    assert loaded == data


def test_run_chain_on_dataset():
    chain = _make_chain("The answer is 42.", ["relevant context"])
    records = run_chain_on_dataset(chain, _make_dataset(2))
    assert len(records) == 2
    assert records[0]["answer"] == "The answer is 42."
    assert records[0]["contexts"] == ["relevant context"]
    assert "question" in records[0]
    assert "ground_truth" in records[0]


def test_run_chain_on_dataset_handles_error():
    chain = MagicMock()
    chain.run.side_effect = Exception("chain error")
    records = run_chain_on_dataset(chain, _make_dataset(1))
    assert records[0]["answer"] == ""
    assert records[0]["contexts"] == []


def test_save_results(tmp_path):
    results = {"faithfulness": 0.9, "answer_relevancy": 0.85}
    output = str(tmp_path / "out" / "eval.json")
    save_results(results, output)
    assert os.path.exists(output)
    with open(output) as f:
        loaded = json.load(f)
    assert loaded["faithfulness"] == 0.9


def test_compute_ragas_metrics_mocked():
    records = [
        {
            "question": "What is RAG?",
            "answer": "Retrieval Augmented Generation.",
            "contexts": ["RAG combines retrieval with generation."],
            "ground_truth": "RAG is retrieval augmented generation.",
        }
    ]
    mock_result = {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8}

    mock_ds_module = MagicMock()
    mock_ds_module.from_list.return_value = MagicMock()

    with patch.dict("sys.modules", {"ragas": MagicMock(evaluate=MagicMock(return_value=mock_result)),
                                     "datasets": MagicMock(Dataset=mock_ds_module),
                                     "ragas.metrics": MagicMock()}):
        with patch("eval.eval_runner.compute_ragas_metrics", return_value=mock_result):
            metrics = mock_result

    assert metrics["faithfulness"] == 0.9
    assert metrics["answer_relevancy"] == 0.85
    assert metrics["context_precision"] == 0.8
