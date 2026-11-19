# Golden-set evaluation

Offline check of the retrieval pipeline (hybrid vector + keyword
search, RRF fusion, refusal threshold, code-assigned citations) on a
known fixture corpus.

Run it:

```bash
python backend/eval/run_eval.py
```

It prints Recall@5, citation correctness, and two offline proxy
metrics for the 20 questions in `golden_set.json` and writes the full
per-question breakdown to `results.json`:

- **Context precision (proxy)** — share of the top-5 retrieved chunks
  that come from the expected document, averaged over questions.
- **Citation coverage (proxy)** — the same per-question test as
  citation correctness, reported under its proxy name.

Both proxies are computed from document identity and keyword overlap
only; they are **not** RAGAS metrics (no LLM judge runs here), and
`results.json` labels them as offline proxies.

## Honesty notes

- The corpus in `fixture_corpus/` is two short **sample** lecture texts
  written for this harness (Data Structures: stacks, queues, binary
  search trees; DBMS: normalization, ACID, keys). Numbers from this
  fixture show the pipeline working on material whose answers are
  known; they are **not** accuracy measurements on real student notes.
  A real golden set should be built from actual course PDFs and
  hand-checked questions.
- When a real embeddings runtime is installed (fastembed, or
  sentence-transformers), the app's real embedding model
  (BAAI/bge-small-en-v1.5) embeds both chunks and queries. When
  neither is, a deterministic hashing bag-of-words embedder stands in
  for both sides so the harness still runs offline with no API keys;
  `results.json` records which embedder produced the numbers. CI sets
  `STUDY_MIND_EVAL_OFFLINE=1` to force the fallback embedder so no
  model is downloaded.
- Citation correctness means: the top retrieved chunk ([S1]) is from
  the expected document and contains at least one expected keyword.
