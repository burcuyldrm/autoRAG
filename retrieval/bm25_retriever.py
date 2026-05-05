from __future__ import annotations

import re
import logging
from typing import Any, TypedDict

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    id: str
    text: str
    metadata: dict[str, Any]


class RankedChunk(TypedDict):
    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [_tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d chunks", len(self._chunks))

    def retrieve(self, query: str, k: int = 5) -> list[RankedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k = min(k, len(self._chunks))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RankedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked_indices
        ]
