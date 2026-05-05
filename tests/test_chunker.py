import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chunker import ChunkerConfig, split_text, create_chunks


def test_512_char_chunking_works():
    text = "a" * 1200

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 512 for chunk in chunks)


def test_1024_char_chunking_works():
    text = "a" * 2500

    config = ChunkerConfig(CHUNK_SIZE=1024, OVERLAP=100)
    chunks = split_text(text, config)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1024 for chunk in chunks)


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


def test_chunk_has_required_fields():
    pages = [
        {
            "paper_id": "paper-2",
            "page": 1,
            "text": "Machine learning is useful. " * 80,
            "source_url": "https://example.com/ml.pdf",
            "source_file": "ml.pdf",
        }
    ]

    chunks = create_chunks(pages)

    assert "id" in chunks[0]
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]


def test_chunk_lengths_not_zero():
    text = "a" * 2000

    chunks = split_text(text)

    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)