# Golden-set evaluation

Offline check of the retrieval pipeline (hybrid vector + keyword
search, RRF fusion, refusal threshold, code-assigned citations) on a
known fixture corpus.

Run it:

```bash
python backend/eval/run_eval.py
```

It prints Recall@5 and citation correctness for the 20 questions in
`golden_set.json` and writes the full per-question breakdown to
`results.json`.

## Honesty notes

- The corpus in `fixture_corpus/` is two short **sample** lecture texts
  written for this harness (Data Structures: stacks, queues, binary
  search trees; DBMS: normalization, ACID, keys). Numbers from this
  fixture show the pipeline working on material whose answers are
  known; they are **not** accuracy measurements on real student notes.
  A real golden set should be built from actual course PDFs and
  hand-checked questions.
- When sentence-transformers is installed, the app's real embedding
  model (BAAI/bge-small-en-v1.5) embeds both chunks and queries. When
  it is not, a deterministic hashing bag-of-words embedder stands in
  for both sides so the harness still runs offline with no API keys;
  `results.json` records which embedder produced the numbers.
- Citation correctness means: the top retrieved chunk ([S1]) is from
  the expected document and contains at least one expected keyword.
