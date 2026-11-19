# StudyMind AI - Architecture

## Decision: website, not a native app

StudyMind is a web app (FastAPI + React), deployed on Render (backend) and Vercel (frontend) - the same pattern as ShieldNet. Students work with PDFs on a laptop; file upload, cited reading, and quiz review are desktop jobs. A live link is also what GitHub reviewers, internship panels, and hackathon judges can open. A PWA install option may come later from this same codebase. No Android/Kotlin/Flutter work is planned - RailX already covers Android in the portfolio.

## Flow

1. **Ingest:** a PDF is uploaded to a subject workspace. `pdf_parser` extracts text per page, `chunking` splits each page into ~800-character chunks that snap to sentence boundaries and never cross a page. Every chunk keeps its document, page and section, and its embedded text is prepended with the `Subject > Document > Section` context path.
2. **Embed:** chunks are embedded locally with `BAAI/bge-small-en-v1.5` (384-dim, free, no key) and stored in Supabase pgvector (schema: `supabase/schema.sql`). Questions are embedded with the bge query instruction prefix; passages are not.
3. **Ask:** hybrid search — vector similarity and keyword ranking (`tsvector` on Postgres) each return a pool, fused with Reciprocal Rank Fusion and reranked by a local FlashRank cross-encoder when installed. A refusal threshold drops weak matches with no LLM call. The LLM (Gemini or Groq free tier, behind `services/llm.py`) must answer using only the retrieved sources and cite the server-assigned labels `[S1]`, `[S2]` — it never generates document names or page numbers itself. If retrieval finds nothing relevant, the answer is "not in your material" - never a guess.
4. **Generate:** summaries, flashcards, and quizzes are generated from the same retrieved chunks, so they stay grounded in the student's own notes.

## Persistence

All data access goes through `backend/app/repositories/` — one interface, two backends:

- **Postgres + pgvector** (`postgres.py`) — used when `DATABASE_URL` points at a Supabase Postgres connection string. On first use it applies `supabase/schema.sql` idempotently, embeds chunks at ingest, and searches with the pgvector cosine operator `<=>` (a `match_chunks` SQL function is also in the schema). Verified live: a 384-dim cosine search on Supabase Postgres 17 ranked a near-identical vector at similarity 0.994 vs 0.110 for an unrelated one.
- **In-memory** (`memory.py`) — used when `DATABASE_URL` is unset, so the app and tests run anywhere with zero setup. It mirrors the same interface, including in-Python cosine search over stored embeddings.

Retrieval mode is labelled in chat responses (`retrieval: hybrid | vector | keyword`, with `+rerank` when FlashRank actually ran): if the local embedding model is not installed, retrieval falls back to keyword overlap instead of silently pretending to do vector search. Answer modes are likewise labelled (`extractive_scaffold`, `no_match`, later `llm`). Nothing is presented as LLM-generated or measured until it actually is.

To switch on real persistence: create a Supabase project, then set `DATABASE_URL` — no code change. (The project's own tables are never mixed into another app's database.)

## Components

- `backend/app/routers/` - workspaces, documents, chat, generate
- `backend/app/services/` - pdf_parser, chunking, embeddings, rag, llm
- `backend/app/repositories/` - base interface, memory backend, Postgres/pgvector backend
- `frontend/src/` - sidebar of subject workspaces, chat panel with citation cards, material/generate panel
