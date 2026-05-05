from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]



        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


def create_chunks(
    pages: List[Dict[str, Any]],
    config: ChunkerConfig = ChunkerConfig()
) -> List[Chunk]:
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
