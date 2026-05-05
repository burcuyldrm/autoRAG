from typing import List, Dict, Any
import numpy as np

from app.retriever import VectorRetriever


class StandardRAGChain:
    def __init__(self):
        # dummy data (şimdilik)
        self.chunks = [
            {"text": "Artificial intelligence is a field of study."},
            {"text": "Machine learning is a subset of AI."},
            {"text": "Cars run on fuel."},
        ]

        self.embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
        ]

        self.retriever = VectorRetriever(self.chunks, self.embeddings)

    def generate(self, chunks: List[Dict[str, Any]]) -> str:
        # basit generate (LLM yok)
        texts = [chunk["text"] for chunk in chunks]
        return " ".join(texts)

    def invoke(self, query: str) -> str:
        query_embedding = np.array([1.0, 0.0])

        retrieved = self.retriever.retrieve(query_embedding, k=2)

        answer = self.generate(retrieved)

        return answer