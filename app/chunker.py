from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


def split_text(text: str, config: ChunkerConfig = None) -> List[str]:
    if config is None:
        config = ChunkerConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )
    return splitter.split_text(text)


def create_chunks(
    pages: List[Dict[str, Any]],
    config: ChunkerConfig = None,
) -> List[Chunk]:
    if config is None:
        config = ChunkerConfig()
    all_chunks: List[Chunk] = []

    for page_data in pages:
        chunks = split_text(page_data["text"], config)

        for i, chunk_text_item in enumerate(chunks):
            chunk: Chunk = {
                "id": f'{page_data.get("paper_id", uuid.uuid4().hex)}_{page_data.get("page", 0)}_{i}',
                "text": chunk_text_item,
                "metadata": {
                    "paper_id": page_data.get("paper_id"),
                    "page": page_data.get("page"),
                    "source_url": page_data.get("source_url"),
                    "source_file": page_data.get("source_file"),
                },
            }
            all_chunks.append(chunk)

    return all_chunks
