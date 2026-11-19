import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.repositories.memory import MemoryRepository
from app.services import embeddings, rag
from app.services.chunking import chunk_page_text


# -- chunking ---------------------------------------------------------------

def test_chunks_snap_to_sentence_boundaries():
    text = ("Binary trees store data in nodes. Each node has two children. "
            "Traversal visits nodes in a defined order. "
            "Inorder traversal yields sorted keys for a search tree.")
    chunks = chunk_page_text(text, page=2, chunk_size=90, overlap=20)
    assert len(chunks) >= 2
    assert all(c["page"] == 2 for c in chunks)
    # Every chunk except possibly one carrying overlap tail text ends a
    # sentence or is the final piece; no chunk starts mid-word.
    assert all(not c["text"][0].islower() or c["text"].split()[0].isalpha()
               for c in chunks)


def test_chunk_records_section_and_context_prefix():
    text = "Binary Trees\nA binary tree has at most two children per node. It is used for searching."
    chunks = chunk_page_text(text, page=1, subject="Data Structures",
                             document="dsa.pdf")
    assert chunks
    assert chunks[0]["section"] == "Binary Trees"
    assert chunks[0]["embedding_text"].startswith(
        "Data Structures > dsa.pdf > Binary Trees")
    # Stored text is the original wording, without the context prefix.
    assert not chunks[0]["text"].startswith("Data Structures >")


# -- embeddings ---------------------------------------------------------------

def test_bge_model_and_query_prefix():
    assert embeddings.MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert embeddings.EMBEDDING_DIM == 384
    seen = {}
    original = embeddings.embed_texts
    try:
        embeddings.embed_texts = lambda texts: (
            seen.setdefault("texts", texts), [[0.1] * 384])[1]
        embeddings.embed_query("what is a stack?")
    finally:
        embeddings.embed_texts = original
    assert seen["texts"] == [
        "Represent this sentence for searching relevant passages: "
        "what is a stack?"]


# -- hybrid retrieval -----------------------------------------------------------

def _repo_with_chunks():
    repo = MemoryRepository()
    ws = repo.create_workspace("Data Structures", "test")
    repo.create_document(ws["id"], "dsa.pdf", 2, [
        {"page": 1, "index": 0,
         "text": "A binary search tree keeps keys sorted. Rotations keep it balanced.",
         "embedding": [1.0, 0.0, 0.0]},
        {"page": 2, "index": 0,
         "text": "Photosynthesis in chlorophyll converts sunlight into glucose.",
         "embedding": [0.0, 1.0, 0.0]},
    ])
    return repo, ws


def _patch_embeddings(monkeypatch, vector):
    monkeypatch.setattr(embeddings, "embeddings_available", lambda: True)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: vector)


def test_hybrid_retrieval_fuses_vector_and_keyword(monkeypatch):
    repo, ws = _repo_with_chunks()
    _patch_embeddings(monkeypatch, [1.0, 0.0, 0.0])
    chunks, mode = rag.retrieve("binary search tree rotations",
                                ws["id"], repo)
    assert mode.startswith("hybrid")
    assert chunks[0]["text"].startswith("A binary search tree")
    assert "rrf_score" in chunks[0]


def test_refusal_below_threshold(monkeypatch):
    repo, ws = _repo_with_chunks()
    _patch_embeddings(monkeypatch, [0.0, 0.0, 1.0])
    chunks, mode = rag.retrieve("quantum entanglement tunnelling",
                                ws["id"], repo)
    assert chunks == []
    assert mode == "below_threshold"


def test_source_ids_assigned_by_code():
    chunks = [{"filename": "dsa.pdf", "page": 3, "index": 1,
               "text": "A stack is LIFO."}]
    labelled = rag.assign_source_ids(chunks)
    citation = rag.citation_for(labelled[0])
    assert citation["source_id"] == "S1"
    prompt = rag.build_prompt("What is a stack?", labelled)
    assert "[S1]" in prompt
    assert "Never invent" in prompt
