-- Relay ticket routing schema for the prepared reference/evaluation CSVs.
-- This migration intentionally stores no direct customer identity fields.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;

create table if not exists public.teams (
  label text primary key check (label in (
    'Billing inquiry',
    'Cancellation request',
    'Product inquiry',
    'Refund request',
    'Technical issue'
  )),
  created_at timestamptz not null default now()
);

insert into public.teams (label) values
  ('Billing inquiry'),
  ('Cancellation request'),
  ('Product inquiry'),
  ('Refund request'),
  ('Technical issue')
on conflict (label) do nothing;

create table if not exists public.tickets (
  rag_id text primary key,
  source_ticket_id text not null unique,
  product text not null,
  ticket_type text not null references public.teams(label),
  subject text not null,
  description text not null,
  rag_text text not null,
  resolution text,
  resolution_quality text not null check (resolution_quality in ('missing', 'very_short', 'usable')),
  status text not null,
  priority text not null,
  channel text not null,
  split text not null check (split in ('reference', 'evaluation')),
  duplicate_group_id text not null,
  created_at timestamptz not null default now()
);

create index if not exists tickets_ticket_type_idx
  on public.tickets (ticket_type);

create index if not exists tickets_split_idx
  on public.tickets (split);

-- rag_id is the primary key, which already provides its required B-tree index.

create index if not exists tickets_split_ticket_type_idx
  on public.tickets (split, ticket_type);

create table if not exists public.ticket_embeddings (
  ticket_id text primary key references public.tickets(rag_id) on delete cascade,
  embedding extensions.vector(768) not null,
  embedding_model text not null,
  embedding_dimensions integer not null check (embedding_dimensions = 768),
  created_at timestamptz not null default now()
);

create index if not exists ticket_embeddings_hnsw_cosine_idx
  on public.ticket_embeddings
  using hnsw (embedding extensions.vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create or replace function public.match_reference_tickets(
  query_embedding extensions.vector(768),
  match_count integer default 5
)
returns table (
  ticket_id text,
  subject text,
  product text,
  ticket_type text,
  resolution text,
  similarity double precision
)
language sql
stable
set search_path = ''
as $$
  select
    e.ticket_id,
    t.subject,
    t.product,
    t.ticket_type,
    t.resolution,
    1 - (e.embedding <=> query_embedding) as similarity
  from public.ticket_embeddings as e
  join public.tickets as t on t.rag_id = e.ticket_id
  where t.split = 'reference'
  order by e.embedding <=> query_embedding
  limit least(greatest(coalesce(match_count, 5), 1), 100);
$$;

alter table public.teams enable row level security;
alter table public.tickets enable row level security;
alter table public.ticket_embeddings enable row level security;

-- No anon/public policies are created. Authenticated demo users can browse
-- teams and reference examples; evaluation rows and their labels remain hidden.
drop policy if exists teams_authenticated_read on public.teams;
create policy teams_authenticated_read
  on public.teams for select to authenticated
  using (true);

drop policy if exists tickets_authenticated_reference_read on public.tickets;
create policy tickets_authenticated_reference_read
  on public.tickets for select to authenticated
  using (split = 'reference');

drop policy if exists embeddings_authenticated_reference_read on public.ticket_embeddings;
create policy embeddings_authenticated_reference_read
  on public.ticket_embeddings for select to authenticated
  using (exists (
    select 1
    from public.tickets as t
    where t.rag_id = ticket_embeddings.ticket_id
      and t.split = 'reference'
  ));
