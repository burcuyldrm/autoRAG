import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunker import create_chunks, ChunkerConfig


def run_experiment(input_path: str, chunk_size: int):
    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    config = ChunkerConfig(
        CHUNK_SIZE=chunk_size,
        OVERLAP=50 if chunk_size == 512 else 100
    )

    chunks = create_chunks(pages, config)

    return chunks


def main():
    input_path = "data/processed/sample.json"
    output_path = "results/chunking_experiment.json"

    os.makedirs("results", exist_ok=True)

    results = {}

    for size in [512, 1024]:
        chunks = run_experiment(input_path, size)

        chunk_count = len(chunks)
        chunk_lengths = [len(c["text"]) for c in chunks]

        avg_chunk_length = sum(chunk_lengths) / chunk_count
        min_chunk_length = min(chunk_lengths)
        max_chunk_length = max(chunk_lengths)

        results[str(size)] = {
            "chunk_size": size,
            "overlap": 50 if size == 512 else 100,
            "chunk_count": chunk_count,
            "avg_chunk_length": round(avg_chunk_length, 2),
            "min_chunk_length": min_chunk_length,
            "max_chunk_length": max_chunk_length
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Experiment tamamlandı: {output_path}")


if __name__ == "__main__":
    main()