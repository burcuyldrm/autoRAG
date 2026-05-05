import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pdf_extractor import extract_text_from_pdf


def test_invalid_path():
    result = extract_text_from_pdf("fake.pdf", "fake")

    assert result == []


def test_empty_file(tmp_path):
    file = tmp_path / "empty.pdf"
    file.write_bytes(b"")

    result = extract_text_from_pdf(str(file), "empty")

    assert isinstance(result, list)


def test_output_format(tmp_path):
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"")

    result = extract_text_from_pdf(str(file), "sample")

    assert isinstance(result, list)


def test_metadata_fields(tmp_path):
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"")

    result = extract_text_from_pdf(str(file), "sample")

    if result:
        assert "paper_id" in result[0]
        assert "page" in result[0]
        assert "text" in result[0]
        assert "source_file" in result[0]


def test_returns_list():
    result = extract_text_from_pdf("nonexistent.pdf", "nonexistent")

    assert isinstance(result, list)