from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 200


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    text: str,
    config: ChunkerConfig | None = None
) -> list[str]:

    if config is None:
        config = ChunkerConfig()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)

def split_text(text: str, chunk_size: int = 512) -> list[str]:
    config = ChunkerConfig(
        chunk_size=chunk_size,
        chunk_overlap=0
    )

    return chunk_text(text, config)


def create_chunks(pages: list[dict]) -> list[dict]:
    results = []

    for page_data in pages:
        chunks = split_text(page_data["text"])

        for i, chunk in enumerate(chunks):
            results.append({
                "id": f'{page_data["paper_id"]}_{page_data["page"]}_{i}',
                "text": chunk,
                "metadata": {
                    "paper_id": page_data["paper_id"],
                    "page": page_data["page"],
                    "source_url": page_data["source_url"],
                    "source_file": page_data["source_file"],
                }
            })

    return results