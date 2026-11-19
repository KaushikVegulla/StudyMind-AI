# StudyMind AI — security notes

An honest checklist of what this deployment does and does not do.
StudyMind is currently a single-user demo: there is **no user
authentication yet**, and this document says plainly where that
leaves the security posture instead of papering over it.

## In place

- **Row Level Security enabled** on every table in
  `supabase/schema.sql` (workspaces, documents, chunks, messages,
  generated_items, flashcards). No permissive policies are shipped;
  owner-scoped policies arrive with the auth step.
- **CORS allowlist from the environment** (`CORS_ORIGINS`), not a
  wildcard: only the origins named there can call the API from a
  browser.
- **Upload validation** on `POST /api/documents/upload`:
  - `.pdf` extension required,
  - PDF magic bytes (`%PDF-`) verified — the extension alone is
    never trusted,
  - size capped by `MAX_UPLOAD_MB` (default 25 MB) with a clean
    `413` response,
  - unparseable files rejected with a clean `400`.
- **Security headers** on every API response:
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, and a restrictive
  `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`
  (the API serves JSON only).
- **Per-IP rate limiting** on the chat and generate endpoints
  (in-memory token bucket, `RATE_LIMIT_PER_MINUTE`, default 60),
  returning `429` with `Retry-After`. This protects the shared
  free-tier LLM quota from a single noisy client.
- **Secrets only from the environment**: `GEMINI_API_KEY`,
  `GROQ_API_KEY` and `DATABASE_URL` are read from env vars, never
  logged, never returned in responses, and never committed (see
  `.env.example` for the full list).
- **Prepared statements disabled** on the Postgres connection
  (`prepare_threshold=None`): Supabase's transaction pooler recycles
  server connections, which otherwise breaks psycopg prepared
  statements (`DuplicatePreparedStatement`).
- **Grounded answers**: chat answers are built only from retrieved
  chunks, citations are assigned by server code (never by the LLM),
  and weak matches are refused instead of answered from unrelated
  material — a prompt-injection and hallucination control, not just
  a quality feature.

## Not in place yet — read before sharing a deployment

- **No user authentication or authorisation.** Anyone who can reach
  the API can create workspaces, upload PDFs and read generated
  content. Do not expose a deployment with real student data until
  auth lands.
- **The backend's Postgres connection bypasses RLS by design.** The
  API connects with the database connection string (owner-level), so
  RLS policies do not constrain the backend itself; RLS currently
  protects only direct database access paths (e.g. Supabase client
  libraries with the anon key). Per-user isolation must come from
  the auth step plus owner checks in the API.
- **Rate limiting is per-process, in-memory.** It resets on restart
  and does not coordinate across instances; the free tier runs one
  instance, which is the only topology it claims to cover.
- **No OCR and no malware scanning of uploads.** Uploaded PDFs are
  parsed for text (pypdf, or docling when installed) but not scanned;
  scanned-image pages are detected and reported
  (`pages_needing_ocr`), not processed.
- **LLM providers see the prompts.** Chat and generation send the
  retrieved chunk text to Gemini/Groq over HTTPS. Uploaded material
  should be treated as visible to those providers.
