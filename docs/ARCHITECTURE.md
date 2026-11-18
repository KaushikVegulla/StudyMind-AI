# StudyMind AI - Architecture

## Decision: website, not a native app

StudyMind is a web app (FastAPI + React), deployed on Render (backend) and Vercel (frontend) - the same pattern as ShieldNet. Students work with PDFs on a laptop; file upload, cited reading, and quiz review are desktop jobs. A live link is also what GitHub reviewers, internship panels, and hackathon judges can open. A PWA install option may come later from this same codebase. No Android/Kotlin/Flutter work is planned - RailX already covers Android in the portfolio.

## Flow

1. **Ingest:** a PDF is uploaded to a subject workspace. `pdf_parser` extracts text per page, `chunking` splits each page into ~800-character chunks with 120 characters of overlap. Every chunk keeps its document and page.
2. **Embed:** chunks are embedded locally with `all-MiniLM-L6-v2` (384-dim, free, no key) and stored in Supabase pgvector (schema: `supabase/schema.sql`).
3. **Ask:** the question is embedded, pgvector returns the top-k nearest chunks, and the LLM (Gemini or Groq free tier, behind `services/llm.py`) must answer using only those chunks and cite document + page for each claim. If retrieval finds nothing relevant, the answer is "not in your material" - never a guess.
4. **Generate:** summaries, flashcards, and quizzes are generated from the same retrieved chunks, so they stay grounded in the student's own notes.

## Scaffold state

The scaffold runs with no keys and no database: an in-memory store, keyword-overlap retrieval, and extractive answers that quote the retrieved chunks and return the same citation shape the vector version will. Modes are labelled honestly in API responses (`extractive_scaffold`, `no_match`, later `llm`). Nothing is presented as LLM-generated or measured until it actually is.

## Components

- `backend/app/routers/` - workspaces, documents, chat, generate
- `backend/app/services/` - pdf_parser, chunking, embeddings, rag, llm
- `backend/app/store.py` - in-memory store (scaffold), replaced by Supabase
- `frontend/src/` - sidebar of subject workspaces, chat panel with citation cards, material/generate panel
