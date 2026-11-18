# StudyMind AI - Demo notes

## Local demo (scaffold)

1. Start the backend: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000`
2. Check `http://localhost:8000/health` - expect `{"status": "ok", ...}` with the seeded workspaces (Data Structures, Databases, Cloud).
3. Start the frontend: `cd frontend && npm install && npm run dev`, open `http://localhost:5173`.
4. Pick a subject, upload a lecture PDF - the panel reports pages and chunk count.
5. Ask a question covered by the PDF - the answer quotes your notes and shows citation cards with document, page, and chunk.
6. Ask something the PDF does not cover - StudyMind says it could not find it in your material instead of inventing an answer.
7. Try Summary / Flashcards / Quiz - scaffold output is extractive and labelled `extractive_scaffold`.

## Tests

```bash
cd backend && python -m pytest tests/ -q
```

## Not in the demo yet

- LLM-generated fluent answers (needs a Gemini or Groq key)
- pgvector persistence (schema is ready in `supabase/schema.sql`)
- Login, deployment links, and any measured accuracy numbers - these are added only when they are real.
