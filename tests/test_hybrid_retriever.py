import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.hybrid_retriever import HybridRetriever

def test_rrf_fusion_works():
    vector_results = [
        {"text": "apple"},
        {"text": "banana"},
        {"text": "car"},
    ]

    bm25_results = [
        {"text": "banana"},
        {"text": "apple"},
        {"text": "dog"},
    ]

    hybrid = HybridRetriever(vector_results, bm25_results)

    results = hybrid.fuse(top_k=2)

    assert len(results) == 2
    assert results[0]["text"] in ["apple", "banana"]