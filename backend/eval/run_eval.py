"""Golden-set evaluation harness for StudyMind retrieval.

Runs the app's real retrieval pipeline (services/rag.py: hybrid
vector + keyword search, RRF fusion, refusal threshold) over a small
fixture corpus and a golden set of 20 questions, then reports:

- **Recall@5** — the expected document appears among the top 5
  retrieved chunks.
- **Citation correctness** — the top retrieved chunk (the one the
  answer would cite first, [S1]) belongs to the expected document AND
  contains at least one of the question's expected keywords.
- **Context precision** (offline proxy) — averaged over questions,
  the share of the retrieved top-5 chunks that come from the expected
  document. A cheap stand-in for the RAGAS metric of the same name:
  no LLM judge, just document identity.
- **Citation coverage** (offline proxy) — the share of questions
  whose top citation is from the expected document AND contains an
  expected keyword (the same per-question test as citation
  correctness, reported under its proxy name). These proxies are NOT
  RAGAS scores and are labelled as proxies in results.json.

The fixture corpus (fixture_corpus/) is two short sample lecture texts
written for this harness — one on Data Structures (stacks, queues,
binary search trees), one on DBMS (normalization, ACID, keys). Results
are therefore fixture-based: they show the pipeline behaving correctly
on known material, not accuracy on real student notes.

Embeddings: when the app's embedding service is available
(fastembed or sentence-transformers serving BAAI/bge-small-en-v1.5)
it is used for both chunks and queries. Otherwise a deterministic
hashing bag-of-words embedder (same 384 dimensions, L2-normalised)
stands in for BOTH sides, so retrieval still exercises the full
hybrid path offline with no API keys; which embedder ran is recorded
in results.json. Set STUDY_MIND_EVAL_OFFLINE=1 to force the fallback
embedder (CI does this, so CI never downloads a model).

Usage:  python backend/eval/run_eval.py
Writes: backend/eval/results.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.repositories.memory import MemoryRepository
from app.services import embeddings, rag
from app.services.chunking import chunk_page_text

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_DIR = HERE / "fixture_corpus"
TOP_K = 5


def _hash_embed(text: str, dim: int = embeddings.EMBEDDING_DIM) -> list[float]:
    """Deterministic bag-of-words hashing embedder (fallback only)."""
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _use_fallback_embedder() -> None:
    embeddings.embeddings_available = lambda: True  # noqa: E731
    embeddings.embed_query = lambda q: _hash_embed(q)  # noqa: E731


def _load_corpus(repo, workspace_id: str, use_real_embeddings: bool):
    doc_ids = {}
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pages = [p.strip() for p in path.read_text(encoding="utf-8")
                 .split("\n---\n")]
        chunks = []
        for page_no, page_text in enumerate(pages, start=1):
            chunks.extend(chunk_page_text(
                page_text, page=page_no, subject="Golden set fixture",
                document=path.name))
        if use_real_embeddings:
            vectors = embeddings.embed_texts(
                [c["embedding_text"] for c in chunks])
        else:
            vectors = [_hash_embed(c["embedding_text"]) for c in chunks]
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        doc = repo.create_document(workspace_id, path.name, len(pages), chunks)
        doc_ids[path.name] = doc["id"]
    return doc_ids


def main() -> int:
    offline = os.getenv("STUDY_MIND_EVAL_OFFLINE") == "1"
    use_real = embeddings.embeddings_available() and not offline
    embedder = (f"{embeddings.MODEL_NAME} via "
                f"{embeddings.active_embedder_name()}") if use_real else \
        "hashing bag-of-words fallback (offline)"
    if not use_real:
        _use_fallback_embedder()

    repo = MemoryRepository()
    ws = repo.create_workspace("Golden set fixture", "eval")
    doc_ids = _load_corpus(repo, ws["id"], use_real)

    golden = json.loads((HERE / "golden_set.json").read_text(encoding="utf-8"))
    rows = []
    recall_hits = 0
    citation_hits = 0
    precision_total = 0.0
    for item in golden:
        chunks, mode = rag.retrieve(item["question"], ws["id"], repo,
                                    top_k=TOP_K)
        expected_id = doc_ids[item["document"]]
        recall = any(c["document_id"] == expected_id for c in chunks)
        citation_ok = False
        if chunks:
            top = chunks[0]
            keywords = [k.lower() for k in item["keywords"]]
            citation_ok = top["document_id"] == expected_id and any(
                k in top["text"].lower() for k in keywords)
        precision = (sum(1 for c in chunks
                         if c["document_id"] == expected_id) / len(chunks)
                     if chunks else 0.0)
        recall_hits += recall
        citation_hits += citation_ok
        precision_total += precision
        rows.append({
            "question": item["question"],
            "expected_document": item["document"],
            "retrieval_mode": mode,
            "recall_at_5": recall,
            "citation_correct": citation_ok,
            "context_precision": round(precision, 4),
            "top_document": chunks[0]["filename"] if chunks else None,
        })

    total = len(golden)
    results = {
        "corpus": "fixture_corpus (sample lecture texts written for "
                  "this harness - fixture-based results, not real notes)",
        "embedder": embedder,
        "questions": total,
        "recall_at_5": round(recall_hits / total, 4),
        "citation_correctness": round(citation_hits / total, 4),
        "context_precision": round(precision_total / total, 4),
        "citation_coverage": round(citation_hits / total, 4),
        "metrics_note": "context_precision and citation_coverage are "
                        "offline proxy metrics computed from document "
                        "identity and expected keywords. They are NOT "
                        "RAGAS scores (no LLM judge, no faithfulness "
                        "measurement).",
        "rows": rows,
    }
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"Golden-set evaluation ({total} questions, fixture corpus)")
    print(f"Embedder: {embedder}")
    print(f"Recall@5:            {recall_hits}/{total} = "
          f"{results['recall_at_5']:.2f}")
    print(f"Citation correctness: {citation_hits}/{total} = "
          f"{results['citation_correctness']:.2f}")
    print(f"Context precision (offline proxy): "
          f"{results['context_precision']:.2f}")
    print(f"Citation coverage (offline proxy): "
          f"{results['citation_coverage']:.2f}")
    for row in rows:
        if not (row["recall_at_5"] and row["citation_correct"]):
            print(f"  miss: {row['question']} "
                  f"(top: {row['top_document']}, mode: {row['retrieval_mode']})")
    print("Wrote backend/eval/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
