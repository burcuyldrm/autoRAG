from typing import List, Dict, Any


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridRetriever:
    def __init__(self, vector_results: List[Dict], bm25_results: List[Dict]):
        self.vector_results = vector_results
        self.bm25_results = bm25_results

    def fuse(self, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = {}

        for rank, item in enumerate(self.vector_results):
            key = item["text"]
            scores[key] = scores.get(key, 0) + rrf_score(rank)

        for rank, item in enumerate(self.bm25_results):
            key = item["text"]
            scores[key] = scores.get(key, 0) + rrf_score(rank)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for text, _ in sorted_items[:top_k]:
            for item in self.vector_results + self.bm25_results:
                if item["text"] == text:
                    result.append(item)
                    break

        return result