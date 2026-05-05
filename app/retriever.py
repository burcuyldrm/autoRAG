from typing import List, Dict, Any
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VectorRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        self.chunks = chunks
        self.embeddings = embeddings

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        scores = []

        for i, emb in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, emb)
            scores.append((score, self.chunks[i]))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scores[:k]]