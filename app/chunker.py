from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import json
import os
import uuid


@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    overlap: int = 50


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


def split_text(text: str, config: ChunkerConfig) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + config.chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += config.chunk_size - config.overlap

    return chunks


def create_chunks(pages: List[Dict[str, Any]], config: ChunkerConfig = ChunkerConfig()) -> List[Chunk]:
    all_chunks = []

    for page_data in pages:
        text = page_data.get("text", "")
        page_chunks = split_text(text, config)

        for chunk_text in page_chunks:
            chunk: Chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "paper_id": page_data.get("paper_id"),
                    "page": page_data.get("page"),
                    "source_file": page_data.get("source_file"),
                    "source_url": page_data.get("source_url")
                }
            }

            all_chunks.append(chunk)

    return all_chunks


def process_json_file(input_path: str, output_folder: str = "data/chunks") -> None:
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = create_chunks(pages)

    file_name = os.path.basename(input_path).replace(".json", "_chunks.json")
    output_path = os.path.join(output_folder, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(chunks)} chunk oluşturuldu: {output_path}")


if __name__ == "__main__":
    process_json_file("data/processed/sample.json")