import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from vectordb.vectorstore import get_vectorstore

CHUNK_FILE = "data/chunks/sample_chunks.json"


def main():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"{len(chunks)} chunk yüklendi")

    store = get_vectorstore(backend="faiss")

    store.ingest(chunks)

    print("FAISS ingestion tamamlandı")
    print(f"Toplam chunk sayısı: {store.count()}")


if __name__ == "__main__":
    main()