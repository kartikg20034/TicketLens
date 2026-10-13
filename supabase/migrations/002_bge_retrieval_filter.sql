-- Restrict nearest-neighbor retrieval to the local BGE embedding generation.
-- This preserves all ticket metadata and vectors; stale Gemini vectors remain
-- stored until their corresponding reference ticket is safely upserted as BGE.

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
    and e.embedding_model = 'BAAI/bge-base-en-v1.5'
  order by e.embedding <=> query_embedding
  limit least(greatest(coalesce(match_count, 5), 1), 100);
$$;
