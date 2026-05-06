from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass, field

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
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
        text = page_data.get("text", "")
        page_chunks = split_text(text, config)

        for index, chunk_text in enumerate(page_chunks):
            chunk: Chunk = {
                "id": f"{page_data.get('paper_id')}_p{page_data.get('page')}_c{index}_{uuid.uuid4().hex[:8]}",
                "text": chunk_text,
                "metadata": {
                    "paper_id": page_data.get("paper_id"),
                    "page": page_data.get("page"),
                    "source_url": page_data.get("source_url"),
                    "source_file": page_data.get("source_file"),
                },
            }
            all_chunks.append(chunk)

    return all_chunks
