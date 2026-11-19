# StudyMind AI - Architecture

## Decision: website, not a native app

StudyMind is a web app (FastAPI + React), deployed on Render (backend) and Vercel (frontend) - the same pattern as ShieldNet. Students work with PDFs on a laptop; file upload, cited reading, and quiz review are desktop jobs. A live link is also what GitHub reviewers, internship panels, and hackathon judges can open. A PWA install option may come later from this same codebase. No Android/Kotlin/Flutter work is planned - RailX already covers Android in the portfolio.

## Flow

1. **Ingest:** a PDF is uploaded to a subject workspace. Uploads are validated first (`.pdf` extension, `%PDF-` magic bytes, `MAX_UPLOAD_MB` size cap). `pdf_parser` extracts text per page — pypdf by default, or Docling when that package is importable (it is an optional local upgrade, deliberately not in `requirements.txt`: torch-scale dependencies do not fit the free tier). The document records which extractor ran. Pages with almost no extracted text are counted as `pages_needing_ocr` — scanned images, reported honestly on the upload response; there is no OCR. `chunking` splits each page into ~800-character parent chunks that snap to sentence boundaries and never cross a page, and each parent also gets ~200-character child chunks. Every chunk keeps its document, page and section, and its embedded text is prepended with the `Subject > Document > Section` context path.
2. **Embed:** chunks are embedded locally with `BAAI/bge-small-en-v1.5` (384-dim, free, no key) and stored in Supabase pgvector (schema: `supabase/schema.sql`). The runtime is fastembed (ONNX — small enough for the free tier, no torch); sentence-transformers is used instead when it is the one installed, and `/health` names the active embedder. Questions are embedded with the bge query instruction prefix; passages are not. Parent chunks that have children are not embedded — their children represent them in search.
3. **Ask:** hybrid search — vector similarity and keyword ranking (`tsvector` on Postgres) each return a pool, fused with Reciprocal Rank Fusion and reranked by a local FlashRank cross-encoder when installed. With parent-child retrieval on (default; `PARENT_CHILD_RETRIEVAL=off` disables it), the pools keep child chunks only; a matched child contributes its parent's full text to the prompt while its citation still names the child's document, page and section. A refusal threshold drops weak matches with no LLM call. HyDE is an opt-in per-request toggle (default off): a short hypothetical answer is generated and embedded *instead of* the raw question for the vector pool; any HyDE failure falls back silently to normal retrieval. The LLM (Gemini or Groq free tier, behind `services/llm.py`) must answer using only the retrieved sources and cite the server-assigned labels `[S1]`, `[S2]` — it never generates document names or page numbers itself. If retrieval finds nothing relevant, the answer is "not in your material" - never a guess.
4. **Generate:** summaries, flashcards, and quizzes are generated from the same retrieved chunks, so they stay grounded in the student's own notes. Quizzes are validated item by item (4 distinct options, one correct answer, keyword grounding) with one regeneration retry before the labelled extractive fallback. Generated artifacts are cached per workspace/document/kind/count — repeat requests make no LLM call unless `force` is set.

## Scheduling (flashcard review)

Flashcards carry spaced-repetition state and a due/review queue (`GET /generate/flashcards/due`, `POST /generate/flashcards/review`). Two engines implement the same stored-state contract, selected by `SRS_ENGINE`:

- **FSRS** (`services/fsrs.py`, default) — FSRS-5 with the published default weights; tracks stability and difficulty per card, with retrievability on the FSRS forgetting curve.
- **SM-2** (`services/srs.py`) — the classic algorithm, kept unchanged.

API grades stay on the 0-5 scale; under FSRS they are mapped to Again/Hard/Good/Easy ratings (0-1 / 2-3 / 4 / 5).

## LLM quota armour

Free-tier quota is the scarcest resource, so `services/llm.py` wraps every provider call: retry with exponential backoff + jitter on HTTP 429/5xx, timeouts and transport errors (`LLM_MAX_ATTEMPTS`, default 3); a per-provider cooldown after consecutive failures (`LLM_COOLDOWN_THRESHOLD` / `LLM_COOLDOWN_SECONDS`) during which the provider is skipped; and a provider chain — configured provider, then the other, then the labelled extractive fallback. The generation cache (above) is the other half: most repeat clicks never reach the LLM at all.

## Security posture

Security headers on every response, CORS from an env allowlist, upload validation, and per-IP rate limiting (in-memory token bucket) on chat/generate. Postgres connections disable prepared statements (`prepare_threshold=None`) because Supabase's transaction pooler recycles server connections. The honest limits — no user auth yet, the backend connection bypasses RLS by design — are listed in `docs/SECURITY.md`.

## Evaluation

`backend/eval/` runs a 20-question golden set over two sample lecture fixtures through the real retrieval pipeline: Recall@5, citation correctness, plus context precision and citation coverage as offline proxy metrics (document-identity and keyword checks — not RAGAS). CI (`.github/workflows/ci.yml`) runs the backend tests, this harness with the offline fallback embedder (no model downloads), and the frontend build.

## Persistence

All data access goes through `backend/app/repositories/` — one interface, two backends:

- **Postgres + pgvector** (`postgres.py`) — used when `DATABASE_URL` points at a Supabase Postgres connection string. On first use it applies `supabase/schema.sql` idempotently, embeds chunks at ingest, and searches with the pgvector cosine operator `<=>` (a `match_chunks` SQL function is also in the schema). Verified live: a 384-dim cosine search on Supabase Postgres 17 ranked a near-identical vector at similarity 0.994 vs 0.110 for an unrelated one.
- **In-memory** (`memory.py`) — used when `DATABASE_URL` is unset, so the app and tests run anywhere with zero setup. It mirrors the same interface, including in-Python cosine search over stored embeddings.

Retrieval mode is labelled in chat responses (`retrieval: hybrid | vector | keyword`, with `+rerank` when FlashRank actually ran): if the local embedding model is not installed, retrieval falls back to keyword overlap instead of silently pretending to do vector search. Answer modes are likewise labelled (`extractive_scaffold`, `no_match`, later `llm`). Nothing is presented as LLM-generated or measured until it actually is.

To switch on real persistence: create a Supabase project, then set `DATABASE_URL` — no code change. (The project's own tables are never mixed into another app's database.)

## Components

- `backend/app/routers/` - workspaces, documents, chat, generate
- `backend/app/services/` - pdf_parser, chunking, embeddings, rag, llm, rerank, quiz_validation, srs (SM-2), fsrs, scheduling (engine selection), rate_limit
- `backend/app/repositories/` - base interface, memory backend, Postgres/pgvector backend
- `backend/eval/` - golden-set evaluation harness and results
- `frontend/src/` - sidebar of subject workspaces, chat panel with citation cards, material/generate panel
