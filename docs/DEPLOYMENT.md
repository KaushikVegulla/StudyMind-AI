# StudyMind AI - Deployment guide

Same pattern as ShieldNet: **Render** runs the FastAPI backend, **Vercel** hosts the React frontend, **Supabase** (when created) provides Postgres + pgvector.

## 1. Backend on Render

The repo root has a `render.yaml` blueprint for a free web service named `studymind-backend` (root directory `backend`, health check `/health`).

1. Render dashboard -> New -> Blueprint -> connect `KaushikVegulla/StudyMind-AI` -> Apply.
2. When the first deploy finishes, open `https://<service>.onrender.com/health` - expect `{"status": "ok", ...}`. On the free plan the service sleeps when idle, so the first request can take about a minute.
3. In the service's Environment tab, set:
   - `GEMINI_API_KEY` - the free Gemini key from Google AI Studio. Without it the backend still works, in labelled extractive mode.
   - `CORS_ORIGINS` - the Vercel URL from step 2, once it exists (comma-separated if there are several).
   - `DATABASE_URL` - only when the Supabase project exists (step 3).
4. Save - Render redeploys automatically.

## 2. Frontend on Vercel

1. Vercel dashboard -> Add New -> Project -> import `KaushikVegulla/StudyMind-AI`.
2. Set **Root Directory** to `frontend` (framework: Vite, detected automatically).
3. Environment variable: `VITE_API_URL` = the Render backend URL from step 1 (no trailing slash).
4. Deploy. Then go back to Render and set `CORS_ORIGINS` to this Vercel URL, so the browser allows the calls.

## 3. Supabase (persistence, when ready)

1. Create a Supabase project named `studymind` (any region close to you, e.g. Mumbai `ap-south-1`).
2. Nothing to paste into the SQL editor: the backend applies `supabase/schema.sql` itself on first start when `DATABASE_URL` is set.
3. Copy the project's connection string (Project Settings -> Database -> URI, pooled) into Render as `DATABASE_URL` and redeploy. `/health` then reports `"persistence": "postgres"` instead of `"memory"`.

Until step 3 is done, the deployed backend keeps workspaces and documents in memory - fine for a demo, but uploads reset when the service restarts or sleeps.

## Notes

- Free Render services sleep after inactivity; the first hit after sleep is slow. That is expected, not a bug.
- Never commit real keys. `.env` is gitignored; production secrets live only in the Render/Vercel dashboards.
