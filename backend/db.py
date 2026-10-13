from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from .seed import generate_tickets

ROOT=Path(__file__).resolve().parents[1]
DB_PATH=ROOT/"data"/"relay.db"
USE_SUPABASE=bool(os.getenv("SUPABASE_DB_URL"))

if USE_SUPABASE:
    import psycopg
    from psycopg.rows import dict_row

def connect():
    if USE_SUPABASE:
        return psycopg.connect(os.environ["SUPABASE_DB_URL"], row_factory=dict_row)
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB_PATH)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db

def _json(row):
    if row is None: return None
    payload=row["payload"] if isinstance(row, dict) else row[0]
    return payload if isinstance(payload, dict) else json.loads(payload)

def init_db():
    if USE_SUPABASE:
        with connect() as db:
            db.execute("""create table if not exists public.tickets (id text primary key,payload jsonb not null,created_at timestamptz not null,updated_at timestamptz not null default now(),dataset_split text not null default 'reference')""")
            db.execute("""create table if not exists public.intake_queue (ticket_id text primary key references public.tickets(id) on delete cascade,state text not null,enqueued_at timestamptz not null,attempts integer not null default 0)""")
            db.execute("""create table if not exists public.queue_events (id bigint generated always as identity primary key,ticket_id text,action text,at timestamptz default now())""")
            count=db.execute("select count(*) as n from public.tickets").fetchone()["n"]
            if count == 0:
                seed=generate_tickets()
                for t in seed:
                    db.execute("insert into public.tickets(id,payload,created_at,updated_at,dataset_split) values (%s,%s::jsonb,%s,%s,%s)", (t["id"],json.dumps(t),t["created_at"],t["updated_at"],t["dataset_split"]))
                for t in seed:
                    if t["status"] == "New":
                        db.execute("insert into public.intake_queue(ticket_id,state,enqueued_at) values (%s,'pending',%s)", (t["id"],t["created_at"]))
        return
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS intake_queue (ticket_id TEXT PRIMARY KEY, state TEXT NOT NULL, enqueued_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)")
        db.execute("CREATE TABLE IF NOT EXISTS queue_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, action TEXT, at TEXT DEFAULT CURRENT_TIMESTAMP)")
        if db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
            seed=generate_tickets()
            db.executemany("INSERT INTO tickets(id,payload) VALUES(?,?)",[(t["id"],json.dumps(t)) for t in seed])
            db.executemany("INSERT INTO intake_queue(ticket_id,state,enqueued_at) VALUES(?,?,?)",[(t["id"],"pending",t["created_at"]) for t in seed if t["status"]=="New"])

def all_tickets():
    with connect() as db:
        if USE_SUPABASE:
            rows=db.execute("select payload from public.tickets order by created_at desc").fetchall()
        else:
            rows=db.execute("SELECT payload FROM tickets ORDER BY json_extract(payload,'$.created_at') DESC").fetchall()
    return [_json(r) for r in rows]

def get_ticket(ticket_id):
    with connect() as db:
        row=db.execute("select payload from public.tickets where id=%s" if USE_SUPABASE else "SELECT payload FROM tickets WHERE id=?",(ticket_id,)).fetchone()
    return _json(row)

def save_ticket(ticket):
    with connect() as db:
        if USE_SUPABASE:
            db.execute("""insert into public.tickets(id,payload,created_at,updated_at,dataset_split) values (%s,%s::jsonb,%s,%s,%s)
            on conflict(id) do update set payload=excluded.payload,updated_at=excluded.updated_at,dataset_split=excluded.dataset_split""",(ticket["id"],json.dumps(ticket),ticket["created_at"],ticket["updated_at"],ticket.get("dataset_split","live")))
        else:
            db.execute("INSERT INTO tickets(id,payload) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",(ticket["id"],json.dumps(ticket)))

def add_queue(ticket_id,at):
    with connect() as db:
        if USE_SUPABASE:
            db.execute("insert into public.intake_queue(ticket_id,state,enqueued_at) values (%s,'pending',%s) on conflict(ticket_id) do nothing",(ticket_id,at))
            db.execute("insert into public.queue_events(ticket_id,action) values (%s,'enqueued')",(ticket_id,))
        else:
            db.execute("INSERT OR IGNORE INTO intake_queue(ticket_id,state,enqueued_at) VALUES(?,?,?)",(ticket_id,"pending",at)); db.execute("INSERT INTO queue_events(ticket_id,action) VALUES(?,?)",(ticket_id,"enqueued"))

def pending_queue():
    with connect() as db:
        ids=[r["ticket_id"] if isinstance(r,dict) else r[0] for r in db.execute("select ticket_id from public.intake_queue where state='pending' order by enqueued_at" if USE_SUPABASE else "SELECT ticket_id FROM intake_queue WHERE state='pending' ORDER BY enqueued_at")]
    return [get_ticket(i) for i in ids]

def queue_snapshot():
    with connect() as db:
        if USE_SUPABASE:
            stats={r["state"]:r["count"] for r in db.execute("select state,count(*) from public.intake_queue group by state")}
            rows=db.execute("select ticket_id,state,enqueued_at,attempts from public.intake_queue order by enqueued_at limit 20").fetchall()
            items=[dict(r) for r in rows]
        else:
            stats={r[0]:r[1] for r in db.execute("SELECT state,COUNT(*) FROM intake_queue GROUP BY state")}; rows=db.execute("SELECT ticket_id,state,enqueued_at,attempts FROM intake_queue ORDER BY enqueued_at LIMIT 20").fetchall();items=[dict(r) for r in rows]
    return {"topic":"support.intake.v1","broker":"supabase-postgres" if USE_SUPABASE else "sqlite-demo-adapter","pending":stats.get("pending",0),"consumed":stats.get("consumed",0),"dead_letter":stats.get("failed",0),"lag":stats.get("pending",0),"items":items}

def consume_ticket(ticket_id):
    with connect() as db:
        if USE_SUPABASE:
            db.execute("update public.intake_queue set state='consumed',attempts=attempts+1 where ticket_id=%s",(ticket_id,)); db.execute("insert into public.queue_events(ticket_id,action) values (%s,'consumed')",(ticket_id,))
        else:
            db.execute("UPDATE intake_queue SET state='consumed',attempts=attempts+1 WHERE ticket_id=?",(ticket_id,)); db.execute("INSERT INTO queue_events(ticket_id,action) VALUES(?,?)",(ticket_id,"consumed"))
