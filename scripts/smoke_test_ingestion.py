#!/usr/bin/env python3
"""Locally embed and upsert only the first five reference tickets."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.local_embeddings import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    encode_texts,
    load_embedding_model,
)

load_dotenv(dotenv_path=".env")
REFERENCE_CSV = PROJECT_ROOT / "data/processed/rag_reference.csv"

TICKET_UPSERT = """
INSERT INTO public.tickets (
  rag_id, source_ticket_id, product, ticket_type, subject, description, rag_text,
  resolution, resolution_quality, status, priority, channel, split, duplicate_group_id
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (rag_id) DO UPDATE SET
  source_ticket_id=EXCLUDED.source_ticket_id, product=EXCLUDED.product,
  ticket_type=EXCLUDED.ticket_type, subject=EXCLUDED.subject,
  description=EXCLUDED.description, rag_text=EXCLUDED.rag_text,
  resolution=EXCLUDED.resolution, resolution_quality=EXCLUDED.resolution_quality,
  status=EXCLUDED.status, priority=EXCLUDED.priority, channel=EXCLUDED.channel,
  split=EXCLUDED.split, duplicate_group_id=EXCLUDED.duplicate_group_id
"""
EMBEDDING_UPSERT = """
INSERT INTO public.ticket_embeddings (ticket_id, embedding, embedding_model, embedding_dimensions)
VALUES (%s, %s::extensions.vector(768), %s, %s)
ON CONFLICT (ticket_id) DO UPDATE SET
  embedding=EXCLUDED.embedding, embedding_model=EXCLUDED.embedding_model,
  embedding_dimensions=EXCLUDED.embedding_dimensions
"""


def load_first_five() -> list[dict[str, str]]:
    with REFERENCE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = [next(reader, None) for _ in range(5)]
    if any(row is None for row in rows):
        raise ValueError("Reference CSV must contain at least five rows")
    result = [row for row in rows if row is not None]
    if any(row.get("split") != "reference" for row in result):
        raise ValueError("Non-reference ticket found; refusing smoke ingestion")
    if len({row["rag_id"] for row in result}) != 5:
        raise ValueError("First five reference rows do not have distinct rag_id values")
    return result


def main() -> int:
    database_url = os.getenv("SUPABASE_DB_URL", "")
    if not database_url:
        print("Required environment variable unavailable: SUPABASE_DB_URL")
        return 1
    try:
        rows = load_first_five()
        vectors = encode_texts(load_embedding_model(), [row["rag_text"] for row in rows])
        with psycopg.connect(database_url, connect_timeout=15) as connection:
            connection.execute("SET search_path TO extensions, public")
            register_vector(connection)
            with connection.cursor() as cursor:
                for row, values in zip(rows, vectors, strict=True):
                    cursor.execute(
                        TICKET_UPSERT,
                        (
                            row["rag_id"], row["Ticket ID"], row["Product Purchased"],
                            row["Ticket Type"], row["Ticket Subject"], row["Ticket Description"],
                            row["rag_text"], row["Resolution"] or None, row["resolution_quality"],
                            row["Ticket Status"], row["Ticket Priority"], row["Ticket Channel"],
                            "reference", row.get("duplicate_group_id", ""),
                        ),
                    )
                    cursor.execute(
                        EMBEDDING_UPSERT,
                        (row["rag_id"], Vector(values), MODEL_ID, EMBEDDING_DIMENSIONS),
                    )
                cursor.execute("SELECT count(*) FROM public.tickets")
                tickets = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM public.ticket_embeddings")
                embeddings = cursor.fetchone()[0]
        print(f"Upserted reference tickets: {len(rows)} using {MODEL_ID}")
        print(f"Tickets in database: {tickets}")
        print(f"Embeddings in database: {embeddings}")
        return 0
    except Exception as error:
        # Avoid printing raw database errors, which can contain connection data.
        print(f"Smoke ingestion failed: {type(error).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
