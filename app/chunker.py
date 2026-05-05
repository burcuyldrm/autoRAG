from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import json
import os
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:
    CHUNK_SIZE: int = 512
    OVERLAP: int = 50


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


def split_text(text: str, config: ChunkerConfig = ChunkerConfig()) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.OVERLAP,
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


def process_json_file(
    input_path: str,
    output_folder: str = "data/chunks"
) -> List[Chunk]:
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = create_chunks(pages)

    file_name = os.path.basename(input_path).replace(".json", "_chunks.json")
    output_path = os.path.join(output_folder, file_name)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"{len(chunks)} chunk oluşturuldu: {output_path}")

    return chunks


if __name__ == "__main__":
    process_json_file("data/processed/sample.json")