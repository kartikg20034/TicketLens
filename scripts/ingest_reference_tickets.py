#!/usr/bin/env python3
"""Locally embed and resumably ingest the reference split into Supabase."""

from __future__ import annotations

import csv
import os
import argparse
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

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
EXPECTED_ROWS = 6771
BATCH_SIZE = 50
ENCODE_BATCH_SIZE = 16
DATABASE_ATTEMPTS = 3
DATABASE_STATEMENT_TIMEOUT_MS = 30_000
DATABASE_LOCK_TIMEOUT_MS = 10_000
RETRY_BASE_DELAY_SECONDS = 2.0

TICKET_UPSERT = """
INSERT INTO public.tickets (
  rag_id, source_ticket_id, product, ticket_type, subject, description,
  rag_text, resolution, resolution_quality, status, priority, channel,
  split, duplicate_group_id
) VALUES (
  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (rag_id) DO UPDATE SET
  source_ticket_id = EXCLUDED.source_ticket_id,
  product = EXCLUDED.product,
  ticket_type = EXCLUDED.ticket_type,
  subject = EXCLUDED.subject,
  description = EXCLUDED.description,
  rag_text = EXCLUDED.rag_text,
  resolution = EXCLUDED.resolution,
  resolution_quality = EXCLUDED.resolution_quality,
  status = EXCLUDED.status,
  priority = EXCLUDED.priority,
  channel = EXCLUDED.channel,
  split = EXCLUDED.split,
  duplicate_group_id = EXCLUDED.duplicate_group_id
"""

EMBEDDING_UPSERT = """
INSERT INTO public.ticket_embeddings (
  ticket_id, embedding, embedding_model, embedding_dimensions
) VALUES (%s, %s::extensions.vector(768), %s, %s)
ON CONFLICT (ticket_id) DO UPDATE SET
  embedding = EXCLUDED.embedding,
  embedding_model = EXCLUDED.embedding_model,
  embedding_dimensions = EXCLUDED.embedding_dimensions
"""


def load_reference_rows() -> list[dict[str, str]]:
    required = {
        "Ticket ID", "Product Purchased", "Ticket Type", "Ticket Subject",
        "Ticket Description", "Ticket Status", "Ticket Priority", "Ticket Channel",
        "Resolution", "resolution_quality", "rag_text", "split", "duplicate_group_id", "rag_id",
    }
    with REFERENCE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} reference tickets; found {len(rows)}")
    if any(row["split"] != "reference" for row in rows):
        raise ValueError("Input contains a non-reference row; refusing to ingest it")
    rag_ids = [row["rag_id"] for row in rows]
    source_ids = [row["Ticket ID"] for row in rows]
    if len(set(rag_ids)) != EXPECTED_ROWS or len(set(source_ids)) != EXPECTED_ROWS:
        raise ValueError("Reference CSV contains duplicate rag_id or Ticket ID values")
    for index, row in enumerate(rows, start=1):
        if re.search(r"(?im)^\s*Ticket Type\s*:", row["rag_text"]):
            raise ValueError(
                f"Reference row {index} embeds a Ticket Type field; regenerate the clean and split CSVs"
            )
    return rows


def reference_embedding_text(row: dict[str, str]) -> str:
    """Return reference text only after rejecting any target-label field."""
    text = row["rag_text"].strip()
    if not text:
        raise ValueError(f"Reference ticket {row.get('rag_id', '<unknown>')} has empty rag_text")
    if re.search(r"(?im)^\s*Ticket Type\s*:", text):
        raise ValueError(
            f"Reference ticket {row.get('rag_id', '<unknown>')} contains target Ticket Type text"
        )
    return text


def open_database(database_url: str) -> psycopg.Connection:
    return psycopg.connect(
        database_url,
        connect_timeout=20,
        autocommit=True,
        options=(
            f"-c statement_timeout={DATABASE_STATEMENT_TIMEOUT_MS} "
            f"-c lock_timeout={DATABASE_LOCK_TIMEOUT_MS}"
        ),
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        tcp_user_timeout=30_000,
    )


def with_database_retry(operation, description: str):
    for attempt in range(1, DATABASE_ATTEMPTS + 1):
        try:
            return operation()
        except (psycopg.OperationalError, psycopg.InterfaceError) as error:
            if attempt == DATABASE_ATTEMPTS:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(
                f"Database retry for {description}: attempt {attempt + 1}/{DATABASE_ATTEMPTS} "
                f"after {type(error).__name__}; waiting {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError("Database retry loop ended unexpectedly")


def ticket_values(row: dict[str, str]) -> tuple[str | None, ...]:
    return (
        row["rag_id"], row["Ticket ID"], row["Product Purchased"], row["Ticket Type"],
        row["Ticket Subject"], row["Ticket Description"], row["rag_text"],
        row["Resolution"] or None, row["resolution_quality"], row["Ticket Status"],
        row["Ticket Priority"], row["Ticket Channel"], "reference", row["duplicate_group_id"],
    )


def existing_bge_embeddings(connection: psycopg.Connection) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT ticket_id
               FROM public.ticket_embeddings
               WHERE embedding_model = %s AND embedding_dimensions = %s""",
            (MODEL_ID, EMBEDDING_DIMENSIONS),
        )
        return {row[0] for row in cursor.fetchall()}


def load_reusable_embedding_ids(database_url: str) -> set[str]:
    def query() -> set[str]:
        with open_database(database_url) as connection:
            connection.execute("SET search_path TO extensions, public")
            register_vector(connection)
            return existing_bge_embeddings(connection)

    return with_database_retry(query, "existing BGE embedding lookup")


def reusable_embedding_ids(database_url: str, *, rebuild_embeddings: bool) -> set[str]:
    """Skip cache lookup during a deliberate full rebuild of stale vectors."""
    if rebuild_embeddings:
        return set()
    return load_reusable_embedding_ids(database_url)


def commit_batch(
    rows: list[dict[str, str]], vectors: dict[str, Vector], database_url: str
) -> None:
    def write() -> None:
        # Encode before opening a database session; commit each 50-ticket unit.
        with open_database(database_url) as connection:
            connection.execute("SET search_path TO extensions, public")
            register_vector(connection)
            with connection.transaction():
                with connection.cursor() as cursor:
                    for row in rows:
                        cursor.execute(TICKET_UPSERT, ticket_values(row))
                        vector = vectors.get(row["rag_id"])
                        if vector is not None:
                            cursor.execute(
                                EMBEDDING_UPSERT,
                                (row["rag_id"], vector, MODEL_ID, EMBEDDING_DIMENSIONS),
                            )

    # All writes are idempotent, so retrying a failed batch is safe.
    with_database_retry(write, f"batch ending at {rows[-1]['rag_id']}")


def database_counts(database_url: str) -> tuple[int, int]:
    def query() -> tuple[int, int]:
        with open_database(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM public.tickets")
                ticket_count = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM public.ticket_embeddings")
                embedding_count = cursor.fetchone()[0]
            return ticket_count, embedding_count

    return with_database_retry(query, "final row counts")


def ingest(
    rows: list[dict[str, str]], database_url: str, *, rebuild_embeddings: bool = False
) -> tuple[int, int, int, int, int, int]:
    model = load_embedding_model()
    processed = generated = reused = errors = 0
    # Existing vectors predate the label-free text correction. Explicit rebuild
    # mode bypasses the model/dimension cache and upserts a fresh vector per row.
    reusable_ids = reusable_embedding_ids(database_url, rebuild_embeddings=rebuild_embeddings)
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(range(0, len(rows), BATCH_SIZE), start=1):
        batch = rows[start:start + BATCH_SIZE]
        missing = [row for row in batch if row["rag_id"] not in reusable_ids]
        reused += len(batch) - len(missing)
        vectors: dict[str, Vector] = {}
        print(
            f"Starting batch {batch_number}/{total_batches}: tickets "
            f"{start + 1}-{start + len(batch)} of {len(rows)}; "
            f"local embeddings needed={len(missing)}"
        )

        # A local model encodes this bounded slice; no hosted embedding API is used.
        try:
            generated_vectors = encode_texts(
                model, [reference_embedding_text(row) for row in missing], batch_size=ENCODE_BATCH_SIZE
            )
            vectors = {
                row["rag_id"]: Vector(vector)
                for row, vector in zip(missing, generated_vectors, strict=True)
            }
        except Exception as error:
            errors += len(missing)
            print(
                f"Local embedding error in batch {batch_number}: "
                f"{len(missing)} tickets pending retry ({type(error).__name__})"
            )

        commit_batch(batch, vectors, database_url)
        processed += len(batch)
        generated += len(vectors)
        reusable_ids.update(vectors)
        print(
            f"Batch {batch_number}/{total_batches}: processed {processed}/{len(rows)}; "
            f"BGE embeddings generated={generated}, reused={reused}, errors={errors}"
        )

    ticket_count, embedding_count = database_counts(database_url)
    return processed, generated, reused, errors, ticket_count, embedding_count


def safe_error(error: BaseException, database_url: str) -> str:
    message = str(error)
    if database_url:
        message = message.replace(database_url, "[redacted connection string]")
        try:
            parsed = urlsplit(database_url)
        except ValueError:
            parsed = None
        if parsed is not None:
            for part in (parsed.netloc, parsed.hostname):
                if part:
                    message = re.sub(re.escape(part), "[redacted]", message, flags=re.IGNORECASE)
            if parsed.path and parsed.path != "/":
                message = message.replace(parsed.path.lstrip("/"), "[redacted]")
            if parsed.password:
                for secret in (parsed.password, unquote(parsed.password)):
                    message = message.replace(secret, "[redacted]")
            for _key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if value:
                    message = message.replace(value, "[redacted]")
        message = re.sub(r"(?i)(postgres(?:ql)?://)[^@\s]+@", r"\1[redacted]@", message)
        message = re.sub(r"(?i)(password\s*=\s*)[^\s]+", r"\1[redacted]", message)
    return message or error.__class__.__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Re-embed every reference ticket, including existing BGE vectors (required after text changes).",
    )
    args = parser.parse_args()
    database_url = os.getenv("SUPABASE_DB_URL", "")
    if not database_url:
        print("Required environment variable unavailable: SUPABASE_DB_URL")
        return 1
    try:
        rows = load_reference_rows()
        processed, generated, reused, errors, ticket_count, embedding_count = ingest(
            rows, database_url, rebuild_embeddings=args.rebuild_embeddings
        )
        print(
            f"Finished: processed={processed}, generated={generated}, "
            f"reused={reused}, errors={errors}"
        )
        print(f"Tickets in database: {ticket_count}")
        print(f"Embeddings in database: {embedding_count}")
        return 1 if errors else 0
    except Exception as error:
        print("Reference ingestion failed: " + safe_error(error, database_url))
        return 1


if __name__ == "__main__":
    sys.exit(main())
