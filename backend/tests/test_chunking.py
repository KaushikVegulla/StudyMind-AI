import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.chunking import chunk_page_text


def test_chunks_keep_page_numbers():
    chunks = chunk_page_text("word " * 400, page=3)
    assert len(chunks) >= 2
    assert all(c["page"] == 3 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_page_text("   ", page=1) == []


def test_overlap_must_be_smaller_than_chunk():
    import pytest
    with pytest.raises(ValueError):
        chunk_page_text("hello world", page=1, chunk_size=10, overlap=10)
