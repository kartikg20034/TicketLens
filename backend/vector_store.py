from __future__ import annotations
import os, json, hashlib, math, re
from typing import Iterable
from . import db

USE_SUPABASE=bool(os.getenv("SUPABASE_DB_URL"))
DIMENSIONS=1536
INDEX_VERSION="openai:text-embedding-3-small"
STOP_WORDS={"a","an","and","are","as","at","be","by","for","from","in","into","is","it","of","on","or","the","to","was","were","with","after","when","during","this","that","than","then","app","ticket","issue","error"}

def _document(ticket):
    return " ".join(str(ticket.get(k,"")) for k in ("title","description","feature","platform","app","severity","device","os_version","logs") if ticket.get(k))[:12000]

def _is_knowledge(ticket):
    return ticket.get("dataset_split") == "reference" and ticket.get("team") not in (None,"","Unassigned") and ticket.get("status") != "New"

def _local_embed(text):
    words=[w for w in re.findall(r"[a-z0-9]+",text.lower()) if w not in STOP_WORDS]; features=words+[f"{a}_{b}" for a,b in zip(words,words[1:])]; v=[0.0]*768
    for f in features:
        d=hashlib.blake2b(f.encode(),digest_size=8).digest(); slot=int.from_bytes(d[:4],"little")%768; v[slot]+=1.0 if d[4]&1 else -1.0
    n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v] if n else v

def initialize(tickets:Iterable[dict]=()):
    if USE_SUPABASE:
        with db.connect() as conn:
            conn.execute("""create table if not exists public.ticket_embeddings(ticket_id text primary key references public.tickets(id) on delete cascade,embedding extensions.vector(1536) not null,document text not null,metadata jsonb not null default '{}'::jsonb,embedding_model text not null,created_at timestamptz default now())""")
            conn.execute("create index if not exists ticket_embeddings_hnsw on public.ticket_embeddings using hnsw (embedding vector_cosine_ops)")
        # Existing reference tickets are indexed lazily by reindex command.
        return
    with db.connect() as conn: conn.execute("""CREATE TABLE IF NOT EXISTS ticket_vectors(ticket_id TEXT PRIMARY KEY,embedding TEXT NOT NULL,document TEXT NOT NULL,metadata TEXT NOT NULL,index_version TEXT NOT NULL)""")
    for t in tickets:
        if _is_knowledge(t): upsert(t)

def upsert(ticket):
    if not _is_knowledge(ticket): return False
    document=_document(ticket); metadata={k:ticket.get(k) for k in ("id","title","team","status","platform","app","feature","severity","description","triage_rationale","triage_confidence","dataset_split")}
    if USE_SUPABASE:
        from .embedding import embed
        vector=embed(document)
        with db.connect() as conn:
            conn.execute("""insert into public.ticket_embeddings(ticket_id,embedding,document,metadata,embedding_model) values (%s,%s::extensions.vector,%s,%s::jsonb,%s)
            on conflict(ticket_id) do update set embedding=excluded.embedding,document=excluded.document,metadata=excluded.metadata,embedding_model=excluded.embedding_model""",(ticket["id"],str(vector),document,json.dumps(metadata),os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")))
        return True
    with db.connect() as conn:
        conn.execute("""INSERT INTO ticket_vectors(ticket_id,embedding,document,metadata,index_version) VALUES(?,?,?,?,?) ON CONFLICT(ticket_id) DO UPDATE SET embedding=excluded.embedding,document=excluded.document,metadata=excluded.metadata,index_version=excluded.index_version""",(ticket["id"],json.dumps(_local_embed(document)),document,json.dumps(metadata),"relay-hash-v1"))
    return True

def search(ticket,limit=5):
    document=_document(ticket)
    if USE_SUPABASE:
        from .embedding import embed
        q=embed(document)
        with db.connect() as conn:
            rows=conn.execute("select ticket_id,metadata,similarity from public.match_ticket_embeddings(%s::extensions.vector, %s, %s, %s)",(str(q),0.0,limit,ticket.get("id"))).fetchall()
        return [{**(r["metadata"] if isinstance(r["metadata"],dict) else json.loads(r["metadata"])),"similarity":round(float(r["similarity"]),4)} for r in rows]
    q=_local_embed(document)
    with db.connect() as conn: rows=conn.execute("SELECT ticket_id,embedding,metadata FROM ticket_vectors WHERE ticket_id != ?",(ticket.get("id",""),)).fetchall()
    out=[]
    for r in rows:
        v=json.loads(r["embedding"]); sim=sum(a*b for a,b in zip(q,v));
        if sim>0:
            m=json.loads(r["metadata"]);m["similarity"]=round(sim,4);out.append(m)
    return sorted(out,key=lambda x:x["similarity"],reverse=True)[:min(limit,10)]

def stats():
    with db.connect() as conn:
        n=conn.execute("select count(*) from public.ticket_embeddings" if USE_SUPABASE else "SELECT COUNT(*) FROM ticket_vectors").fetchone()[0]
    return {"provider":"supabase-pgvector" if USE_SUPABASE else "sqlite-local-vector-store","embedding":INDEX_VERSION if USE_SUPABASE else "relay-hash-v1","dimensions":DIMENSIONS if USE_SUPABASE else 768,"indexed_tickets":n}

def reindex(tickets):
    refs=[t for t in tickets if _is_knowledge(t)]
    if USE_SUPABASE:
        with db.connect() as conn: conn.execute("delete from public.ticket_embeddings")
    else:
        with db.connect() as conn: conn.execute("DELETE FROM ticket_vectors")
    return sum(1 for t in refs if upsert(t))
