-- StudyMind AI - Supabase Postgres + pgvector schema
-- Run in the Supabase SQL editor. Embeddings: BAAI/bge-small-en-v1.5, 384 dims.
-- NOTE: chunks embedded with the old all-MiniLM-L6-v2 model must be
-- re-embedded (re-upload) after this change - the vector space differs
-- even though the dimension is the same.
create extension if not exists vector;

create table if not exists workspaces (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid,
  name text not null,
  description text default '',
  created_at timestamptz default now()
);

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  filename text not null,
  pages int not null default 0,
  extractor text not null default 'pypdf',
  pages_needing_ocr int not null default 0,
  created_at timestamptz default now()
);

-- For projects created before the extractor columns existed.
alter table documents add column if not exists extractor text default 'pypdf';
alter table documents add column if not exists pages_needing_ocr int default 0;

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  workspace_id uuid not null references workspaces(id) on delete cascade,
  page int not null,
  chunk_index int not null,
  text text not null,
  section text default '',
  embedding vector(384),
  created_at timestamptz default now()
);

-- For projects created before the section column existed.
alter table chunks add column if not exists section text default '';

-- Parent-child retrieval: a chunk with parent_id set is a small child
-- chunk; retrieval matches children and answers from the parent's text.
alter table chunks add column if not exists parent_id uuid
  references chunks(id) on delete cascade;
create index if not exists chunks_parent_idx on chunks(parent_id);

-- HNSW over IVFFlat: better recall at this scale, no training step.
drop index if exists chunks_embedding_idx;
create index if not exists chunks_embedding_hnsw_idx
  on chunks using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);
create index if not exists chunks_workspace_idx on chunks(workspace_id);
-- Full-text index for the keyword half of hybrid search.
create index if not exists chunks_text_fts_idx
  on chunks using gin (to_tsvector('english', text));

create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid references workspaces(id) on delete cascade,
  question text not null,
  answer text not null,
  citations jsonb default '[]'::jsonb,
  mode text,
  created_at timestamptz default now()
);

create table if not exists generated_items (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  kind text not null check (kind in ('summary','flashcards','quiz')),
  content jsonb not null,
  created_at timestamptz default now()
);

-- Generation cache keys: one stored payload per document (or per whole
-- workspace when document_id is null), kind and requested count.
alter table generated_items
  add column if not exists document_id uuid
    references documents(id) on delete cascade;
alter table generated_items add column if not exists item_count int default 5;

-- Flashcards carry SM-2 spaced-repetition state (backend services/srs.py).
-- A new card is due immediately: due_at defaults to its creation time.
create table if not exists flashcards (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references workspaces(id) on delete cascade,
  document_id uuid references documents(id) on delete cascade,
  front text not null,
  back text not null,
  ease_factor double precision not null default 2.5,
  interval_days int not null default 0,
  repetitions int not null default 0,
  due_at timestamptz not null default now(),
  created_at timestamptz default now()
);
create index if not exists flashcards_due_idx
  on flashcards(workspace_id, due_at);

-- FSRS scheduling state (backend services/fsrs.py). The review engine
-- is chosen by the SRS_ENGINE env var (fsrs by default, sm2 keeps the
-- columns above). FSRS fields stay null until a card is reviewed under
-- the FSRS engine; last_review_at feeds its retrievability estimate
-- and is maintained by both engines.
alter table flashcards add column if not exists fsrs_stability double precision;
alter table flashcards add column if not exists fsrs_difficulty double precision;
alter table flashcards add column if not exists last_review_at timestamptz;

-- Vector search: nearest chunks to a question embedding, cosine distance.
-- The backend inlines the same ORDER BY in its repository queries; this
-- function is provided for SQL-side callers and debugging in the editor.
create or replace function match_chunks(
  query_embedding vector(384),
  match_workspace uuid default null,
  match_count int default 4
)
returns table (
  id uuid, document_id uuid, workspace_id uuid, filename text,
  page int, chunk_index int, text text, similarity float
)
language sql stable
as $$
  select c.id, c.document_id, c.workspace_id, d.filename, c.page,
         c.chunk_index, c.text,
         1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  join documents d on d.id = c.document_id
  where c.embedding is not null
    and (match_workspace is null or c.workspace_id = match_workspace)
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

alter table workspaces enable row level security;
alter table documents enable row level security;
alter table chunks enable row level security;
alter table messages enable row level security;
alter table generated_items enable row level security;
alter table flashcards enable row level security;
-- NOTE: owner-scoped policies are added with the auth step.
-- Do not ship permissive policies to production.
