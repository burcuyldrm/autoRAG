import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from app.retriever import VectorRetriever

def test_retriever_returns_top_k():
    chunks = [
        {"text": "apple"},
        {"text": "banana"},
        {"text": "car"},
    ]

    embeddings = [
        np.array([1, 0]),   # apple
        np.array([0.9, 0.1]),  # banana (apple'a yakın)
        np.array([0, 1]),   # car
    ]

    retriever = VectorRetriever(chunks, embeddings)

    query = np.array([1, 0])  # apple'a yakın

    results = retriever.retrieve(query, k=2)

    assert len(results) == 2
    assert results[0]["text"] == "apple"