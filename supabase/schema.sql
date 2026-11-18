-- StudyMind AI - Supabase Postgres + pgvector schema
-- Run in the Supabase SQL editor. Embeddings: all-MiniLM-L6-v2, 384 dims.
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
  created_at timestamptz default now()
);

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  workspace_id uuid not null references workspaces(id) on delete cascade,
  page int not null,
  chunk_index int not null,
  text text not null,
  embedding vector(384),
  created_at timestamptz default now()
);

create index if not exists chunks_embedding_idx
  on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists chunks_workspace_idx on chunks(workspace_id);

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

alter table workspaces enable row level security;
alter table documents enable row level security;
alter table chunks enable row level security;
alter table messages enable row level security;
alter table generated_items enable row level security;
-- NOTE: owner-scoped policies are added with the auth step.
-- Do not ship permissive policies to production.
