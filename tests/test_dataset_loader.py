"""Tests for eval.dataset_loader."""
from __future__ import annotations

import json
import os

import pytest

from eval.dataset_loader import load_qa_dataset, validate_qa_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(id_="q001", question="What is RAG?", ground_truth="RAG is ...",
                 expected_source="paper") -> dict:
    return {
        "id": id_,
        "question": question,
        "ground_truth": ground_truth,
        "expected_source": expected_source,
        "topic": "RAG",
        "difficulty": "easy",
    }


def _write_json(tmp_path, data) -> str:
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# load_qa_dataset
# ---------------------------------------------------------------------------

def test_load_returns_list(tmp_path):
    samples = [_make_sample()]
    path = _write_json(tmp_path, samples)
    result = load_qa_dataset(path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_preserves_fields(tmp_path):
    sample = _make_sample(id_="q007", question="What is BM25?")
    path = _write_json(tmp_path, [sample])
    result = load_qa_dataset(path)
    assert result[0]["id"] == "q007"
    assert result[0]["question"] == "What is BM25?"


def test_load_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_qa_dataset("/nonexistent/path/qa.json")


def test_load_raises_on_non_list(tmp_path):
    path = _write_json(tmp_path, {"key": "value"})
    with pytest.raises(ValueError, match="list"):
        load_qa_dataset(path)


def test_load_actual_dataset():
    """Smoke test: load the real qa_dataset.json and check minimum size."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "qa_dataset.json")
    if not os.path.exists(path):
        pytest.skip("qa_dataset.json not found")
    samples = load_qa_dataset(path)
    assert len(samples) >= 30


# ---------------------------------------------------------------------------
# validate_qa_dataset
# ---------------------------------------------------------------------------

def test_validate_ok():
    samples = [_make_sample(id_=f"q{i:03d}") for i in range(5)]
    validate_qa_dataset(samples)  # should not raise


def test_validate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_qa_dataset([])


def test_validate_missing_id():
    sample = _make_sample()
    del sample["id"]
    with pytest.raises(ValueError, match="id"):
        validate_qa_dataset([sample])


def test_validate_missing_question():
    sample = _make_sample()
    del sample["question"]
    with pytest.raises(ValueError, match="question"):
        validate_qa_dataset([sample])


def test_validate_missing_ground_truth():
    sample = _make_sample()
    del sample["ground_truth"]
    with pytest.raises(ValueError, match="ground_truth"):
        validate_qa_dataset([sample])


def test_validate_missing_expected_source():
    sample = _make_sample()
    del sample["expected_source"]
    with pytest.raises(ValueError, match="expected_source"):
        validate_qa_dataset([sample])


def test_validate_empty_question():
    sample = _make_sample(question="   ")
    with pytest.raises(ValueError, match="empty.*question"):
        validate_qa_dataset([sample])


def test_validate_empty_ground_truth():
    sample = _make_sample(ground_truth="")
    with pytest.raises(ValueError, match="empty.*ground_truth"):
        validate_qa_dataset([sample])


def test_validate_duplicate_id():
    samples = [_make_sample(id_="q001"), _make_sample(id_="q001")]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_qa_dataset(samples)


def test_validate_multiple_samples_ok():
    samples = [_make_sample(id_=f"q{i:03d}") for i in range(10)]
    validate_qa_dataset(samples)  # should not raise


def test_validate_real_dataset():
    """Validate the actual qa_dataset.json file."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "qa_dataset.json")
    if not os.path.exists(path):
        pytest.skip("qa_dataset.json not found")
    from eval.dataset_loader import load_qa_dataset
    samples = load_qa_dataset(path)
    validate_qa_dataset(samples)  # should not raise
