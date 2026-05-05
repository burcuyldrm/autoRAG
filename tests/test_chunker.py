import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunker import ChunkerConfig, Chunker1024, split_text, create_chunks


def test_512_char_chunking_works():
    text = "a" * 1200

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 512 for chunk in chunks)


def test_metadata_is_preserved():
    pages = [
        {
            "paper_id": "paper-1",
            "page": 3,
            "text": "This is a sample academic paper text. " * 100,
            "source_url": "https://example.com/paper.pdf",
            "source_file": "paper.pdf",
        }
    ]

    chunks = create_chunks(pages)

    assert len(chunks) > 0
    assert chunks[0]["metadata"]["paper_id"] == "paper-1"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["source_url"] == "https://example.com/paper.pdf"


def test_1024_char_chunking_works():
    text = "a" * 2500

    config = ChunkerConfig(chunk_size=1024, overlap=100, mode="1024")
    chunks = split_text(text, config)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1024 for chunk in chunks)


def test_chunker1024_class_works():
    text = "a" * 2500

    chunker = Chunker1024()
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1024 for chunk in chunks)


def test_512_and_1024_chunking_are_different():
    text = "This is a sample academic text. " * 200

    chunks_512 = split_text(
        text,
        ChunkerConfig(chunk_size=512, overlap=50, mode="512")
    )

    chunks_1024 = split_text(
        text,
        ChunkerConfig(chunk_size=1024, overlap=100, mode="1024")
    )

    assert len(chunks_1024) < len(chunks_512)