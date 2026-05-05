from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import json
import os
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    overlap: int = 50
    mode: str = "512"


class Chunk(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


class Chunker1024:
    def __init__(self):
        self.config = ChunkerConfig(
            chunk_size=1024,
            overlap=100,
            mode="1024"
        )

    def split(self, text: str) -> List[str]:
        return split_text(text, self.config)

    def create_chunks(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        return create_chunks(pages, self.config)


def split_text(text: str, config: ChunkerConfig = ChunkerConfig()) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
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
                    "chunk_size": config.chunk_size,
                    "overlap": config.overlap,
                    "mode": config.mode,
                },
            }

            all_chunks.append(chunk)

    return all_chunks


def get_config_by_mode(mode: str) -> ChunkerConfig:
    if mode == "1024":
        return ChunkerConfig(chunk_size=1024, overlap=100, mode="1024")

    return ChunkerConfig(chunk_size=512, overlap=50, mode="512")


def process_json_file(
    input_path: str,
    output_folder: str = "data/chunks",
    mode: str = "512"
) -> List[Chunk]:
    os.makedirs(output_folder, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    config = get_config_by_mode(mode)
    chunks = create_chunks(pages, config)

    file_name = os.path.basename(input_path).replace(".json", f"_{mode}_chunks.json")
    output_path = os.path.join(output_folder, file_name)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"{len(chunks)} chunk oluşturuldu: {output_path}")

    return chunks


if __name__ == "__main__":
    process_json_file("data/processed/sample.json", mode="1024")