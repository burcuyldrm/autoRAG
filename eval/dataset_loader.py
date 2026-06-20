"""
Q&A Dataset Loader

Provides:
  - load_qa_dataset(path) -> list[dict]
  - validate_qa_dataset(samples) -> None  (raises ValueError on failure)
"""
from __future__ import annotations

import json
from typing import Any


_REQUIRED_FIELDS = ("id", "question", "ground_truth", "expected_source")


def load_qa_dataset(path: str) -> list[dict[str, Any]]:
    """Load a JSON Q&A dataset from the given file path.

    Parameters
    ----------
    path : str
        Path to a JSON file containing a list of Q&A sample dicts.

    Returns
    -------
    list[dict]
        The parsed list of samples.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain a JSON list.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path!r}, got {type(data).__name__}.")
    return data


def validate_qa_dataset(samples: list[dict[str, Any]]) -> None:
    """Validate the structure of a Q&A dataset.

    Checks:
    - Each sample has the required fields: id, question, ground_truth, expected_source
    - ``question`` is non-empty
    - ``ground_truth`` is non-empty
    - No duplicate ``id`` values

    Parameters
    ----------
    samples : list[dict]
        The dataset to validate.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if not samples:
        raise ValueError("Dataset is empty.")

    seen_ids: set[str] = set()

    for idx, sample in enumerate(samples):
        # Required fields present
        for field in _REQUIRED_FIELDS:
            if field not in sample:
                raise ValueError(
                    f"Sample at index {idx} is missing required field {field!r}."
                )

        sample_id = sample["id"]

        # Non-empty question
        if not str(sample.get("question", "")).strip():
            raise ValueError(
                f"Sample {sample_id!r} (index {idx}) has an empty 'question'."
            )

        # Non-empty ground_truth
        if not str(sample.get("ground_truth", "")).strip():
            raise ValueError(
                f"Sample {sample_id!r} (index {idx}) has an empty 'ground_truth'."
            )

        # Duplicate id check
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate 'id' value found: {sample_id!r} (index {idx})."
            )
        seen_ids.add(sample_id)
