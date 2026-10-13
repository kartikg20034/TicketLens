-- Additive schema for an isolated synthetic ticket-routing experiment.
-- This migration does not alter or backfill public.tickets or its embeddings.

create table if not exists public.synthetic_tickets (
  rag_id text primary key check (rag_id like 'synthetic:%'),
  source_ticket_id text not null unique,
  dataset_name text not null default 'realistic_app_support_tickets_1000',
  app_name text not null,
  category text not null references public.teams(label),
  subject text not null,
  description text not null,
  rag_text text not null,
  created_at timestamptz,
  city text,
  channel text,
  priority text,
  status text,
  device_model text,
  os_version text,
  app_version text,
  transaction_amount_inr numeric,
  transaction_state text,
  trip_origin text,
  trip_destination text,
  event_time_local text,
  first_response_minutes numeric,
  resolution_minutes numeric,
  scenario_group_id text not null,
  split text not null check (split = 'train'),
  source_type text not null,
  is_synthetic boolean not null default true check (is_synthetic),
  inserted_at timestamptz not null default now(),
  check (length(btrim(rag_text)) > 0)
);

create index if not exists synthetic_tickets_category_idx
  on public.synthetic_tickets (category);

create index if not exists synthetic_tickets_scenario_group_idx
  on public.synthetic_tickets (scenario_group_id);

create table if not exists public.synthetic_ticket_embeddings (
  ticket_id text primary key references public.synthetic_tickets(rag_id) on delete restrict,
  embedding extensions.vector(768) not null,
  embedding_model text not null,
  embedding_dimensions integer not null check (embedding_dimensions = 768),
  created_at timestamptz not null default now()
);

create index if not exists synthetic_ticket_embeddings_hnsw_cosine_idx
  on public.synthetic_ticket_embeddings
  using hnsw (embedding extensions.vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create or replace function public.match_synthetic_train_tickets(
  query_embedding extensions.vector(768),
  match_count integer default 5
)
returns table (
  ticket_id text,
  subject text,
  app_name text,
  category text,
  similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
  select
    e.ticket_id,
    t.subject,
    t.app_name,
    t.category,
    1 - (e.embedding OPERATOR(extensions.<=>) query_embedding) as similarity
  from public.synthetic_ticket_embeddings as e
  join public.synthetic_tickets as t on t.rag_id = e.ticket_id
  where t.split = 'train'
    and t.is_synthetic is true
    and e.embedding_model = 'BAAI/bge-base-en-v1.5'
    and e.embedding_dimensions = 768
  order by e.embedding OPERATOR(extensions.<=>) query_embedding
  limit least(greatest(coalesce(match_count, 5), 1), 100);
$$;

revoke execute on function public.match_synthetic_train_tickets(extensions.vector, integer)
  from public;
revoke execute on function public.match_synthetic_train_tickets(extensions.vector, integer)
  from anon;
grant execute on function public.match_synthetic_train_tickets(extensions.vector, integer)
  to postgres;

alter table public.synthetic_tickets enable row level security;
alter table public.synthetic_ticket_embeddings enable row level security;

drop policy if exists synthetic_tickets_authenticated_train_read on public.synthetic_tickets;
create policy synthetic_tickets_authenticated_train_read
  on public.synthetic_tickets for select to authenticated
  using (split = 'train' and is_synthetic is true);

drop policy if exists synthetic_embeddings_authenticated_train_read on public.synthetic_ticket_embeddings;
create policy synthetic_embeddings_authenticated_train_read
  on public.synthetic_ticket_embeddings for select to authenticated
  using (exists (
    select 1 from public.synthetic_tickets as t
    where t.rag_id = synthetic_ticket_embeddings.ticket_id
      and t.split = 'train' and t.is_synthetic is true
  ));
