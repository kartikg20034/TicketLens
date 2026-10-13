#!/usr/bin/env python3
"""Ingest only the training split of the isolated synthetic ticket dataset."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_embeddings import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    encode_texts,
    load_embedding_model,
)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATASET_CSV = PROJECT_ROOT / "data/processed/realistic_app_support_tickets_1000.csv"
DATASET_NAME = "realistic_app_support_tickets_1000"
EXPECTED_SPLIT_COUNTS = {"train": 700, "validation": 150, "test": 150}
LABELS = (
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
)
REQUIRED_COLUMNS = {
    "ticket_id", "app_name", "category", "subject", "description", "rag_text",
    "created_at", "city", "channel", "priority", "status", "device_model",
    "os_version", "app_version", "transaction_amount_inr", "transaction_state",
    "trip_origin", "trip_destination", "event_time_local", "first_response_minutes",
    "resolution_minutes", "scenario_group_id", "split", "source_type", "is_synthetic",
}
TARGET_LABEL_LINE = re.compile(r"(?im)^\s*(?:category|ticket\s+type)\s*:[^\r\n]*(?:\r?\n|$)")
INSERT_TICKET_SQL = """
INSERT INTO public.synthetic_tickets (
  rag_id, source_ticket_id, dataset_name, app_name, category, subject, description,
  rag_text, created_at, city, channel, priority, status, device_model, os_version,
  app_version, transaction_amount_inr, transaction_state, trip_origin,
  trip_destination, event_time_local, first_response_minutes, resolution_minutes,
  scenario_group_id, split, source_type, is_synthetic
) VALUES (
  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s, %s, %s, %s, true
) ON CONFLICT (rag_id) DO NOTHING
"""
INSERT_EMBEDDING_SQL = """
INSERT INTO public.synthetic_ticket_embeddings (
  ticket_id, embedding, embedding_model, embedding_dimensions
) VALUES (%s, %s::extensions.vector(768), %s, %s)
ON CONFLICT (ticket_id) DO NOTHING
"""
EXISTING_VECTOR_PROVENANCE_NOTE = (
    "The current schema stores embedding_model and embedding_dimensions but no rag_text hash; "
    "matching ticket metadata therefore cannot cryptographically prove which text produced an existing vector."
)


def validate_dataset_rows(rows: list[dict[str, str]], *, require_full_dataset: bool = False) -> None:
    if not rows:
        raise ValueError("Synthetic dataset is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Synthetic dataset is missing columns: {sorted(missing)}")
    ids: set[str] = set()
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    for number, row in enumerate(rows, start=1):
        ticket_id = row.get("ticket_id", "").strip()
        if not ticket_id:
            raise ValueError(f"Row {number} has an empty ticket_id")
        if ticket_id in ids:
            raise ValueError(f"Duplicate ticket_id: {ticket_id}")
        ids.add(ticket_id)
        if row.get("category") not in LABELS:
            raise ValueError(f"Row {number} has a category outside the five routing labels")
        split = row.get("split")
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"Row {number} has invalid split {split!r}")
        if not row.get("scenario_group_id", "").strip():
            raise ValueError(f"Row {number} has an empty scenario_group_id")
        scenario_splits[row["scenario_group_id"]].add(split)
        label_counts[row["category"]].add(split)
        split_counts[split] += 1
        if row.get("is_synthetic", "").strip().casefold() not in {"true", "1", "yes"}:
            raise ValueError(f"Row {number} is not marked synthetic")
        if not row.get("rag_text", "").strip():
            raise ValueError(f"Row {number} has empty rag_text")
        if TARGET_LABEL_LINE.search(row["rag_text"]):
            raise ValueError(f"Row {number} has an explicit category/target line in rag_text")
        if row["category"].casefold() in row["rag_text"].casefold():
            raise ValueError(f"Row {number} leaks its category label into rag_text")
        if not row.get("subject", "").strip() or not row.get("app_name", "").strip():
            raise ValueError(f"Row {number} is missing subject or app_name")
    crossing = [group for group, splits in scenario_splits.items() if len(splits) > 1]
    if crossing:
        raise ValueError(f"Scenario groups cross splits: {crossing[:5]}")
    if set(label_counts) != set(LABELS):
        raise ValueError("Dataset must contain all five routing categories")
    if require_full_dataset and dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Expected split counts {EXPECTED_SPLIT_COUNTS}; found {dict(split_counts)}"
        )


def load_dataset(path: Path = DATASET_CSV, *, require_full_dataset: bool = True) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Synthetic dataset CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Synthetic dataset is missing columns: {sorted(missing)}")
        rows = list(reader)
    validate_dataset_rows(rows, require_full_dataset=require_full_dataset)
    return rows


def select_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return train rows only, after validating split isolation over all rows."""
    validate_dataset_rows(rows)
    training = [row for row in rows if row["split"] == "train"]
    if not training:
        raise ValueError("No synthetic training rows are available")
    if any(row["split"] != "train" for row in training):
        raise ValueError("Non-training ticket reached synthetic ingestion")
    return training


def synthetic_embedding_text(row: dict[str, str]) -> str:
    """Return only the supplied rag_text, never its category target."""
    text = (row.get("rag_text") or "").strip()
    label = row.get("category", "")
    if not text:
        raise ValueError("Synthetic rag_text is empty")
    if label not in LABELS:
        raise ValueError("Synthetic category is outside the five routing labels")
    if TARGET_LABEL_LINE.search(text) or label.casefold() in text.casefold():
        raise ValueError("Synthetic rag_text contains target-label leakage")
    return text


def namespaced_rag_id(ticket_id: str) -> str:
    if not ticket_id or ticket_id.startswith("synthetic:"):
        raise ValueError("ticket_id must be a non-empty, unprefixed source ID")
    return f"synthetic:{ticket_id}"


def find_existing_id_collisions(
    candidate_source_ids: set[str], candidate_rag_ids: set[str], existing_ids: set[str]
) -> set[str]:
    """Pure helper used by DB preflight and collision tests."""
    return candidate_source_ids & existing_ids | candidate_rag_ids & existing_ids


def _nullable(row: dict[str, str], field: str) -> str | None:
    value = row.get(field, "")
    return value if value != "" else None


def ticket_values(row: dict[str, str]) -> tuple[Any, ...]:
    if row.get("split") != "train":
        raise ValueError("Refusing to build an insert for a non-training ticket")
    synthetic_embedding_text(row)
    return (
        namespaced_rag_id(row["ticket_id"]), row["ticket_id"], DATASET_NAME,
        row["app_name"], row["category"], row["subject"], row["description"],
        row["rag_text"], _nullable(row, "created_at"), _nullable(row, "city"),
        _nullable(row, "channel"), _nullable(row, "priority"), _nullable(row, "status"),
        _nullable(row, "device_model"), _nullable(row, "os_version"),
        _nullable(row, "app_version"), _nullable(row, "transaction_amount_inr"),
        _nullable(row, "transaction_state"), _nullable(row, "trip_origin"),
        _nullable(row, "trip_destination"), _nullable(row, "event_time_local"),
        _nullable(row, "first_response_minutes"), _nullable(row, "resolution_minutes"),
        row["scenario_group_id"], "train", row["source_type"],
    )


def database_existing_ids(connection: psycopg.Connection, source_ids: list[str], rag_ids: list[str]) -> set[str]:
    """Find any ID collision in the original corpus before synthetic writes."""
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT rag_id, source_ticket_id FROM public.tickets
               WHERE rag_id = ANY(%s) OR source_ticket_id = ANY(%s)""",
            (rag_ids + source_ids, rag_ids + source_ids),
        )
        found = set()
        for rag_id, source_id in cursor.fetchall():
            found.update((str(rag_id), str(source_id)))
        return found


def validate_existing_ticket_metadata(
    incoming: dict[str, str], stored: dict[str, Any]
) -> None:
    """Fail before encoding if an existing synthetic ID has different metadata."""
    ticket_id = namespaced_rag_id(incoming["ticket_id"])
    expected = {
        "source_ticket_id": incoming["ticket_id"],
        "dataset_name": DATASET_NAME,
        "category": incoming["category"],
        "rag_text": incoming["rag_text"],
    }
    for field, expected_value in expected.items():
        if stored.get(field) != expected_value:
            raise ValueError(
                f"Conflicting existing synthetic ticket {ticket_id}: field {field} differs; "
                "no existing data was changed"
            )


def validate_existing_embedding(ticket_id: str, model: str, dimensions: int) -> None:
    if model != MODEL_ID:
        raise ValueError(
            f"Existing synthetic embedding {ticket_id} has embedding_model={model!r}; "
            "expected the current BGE model; no existing data was changed"
        )
    if int(dimensions) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Existing synthetic embedding {ticket_id} has embedding_dimensions={dimensions!r}; "
            "expected 768; no existing data was changed"
        )


def existing_synthetic_state(
    connection: psycopg.Connection,
    incoming_rows: list[dict[str, str]],
) -> tuple[set[str], set[str]]:
    """Verify existing metadata and embeddings before deciding what to encode.

    Stored row text can be compared to incoming rag_text, but this schema has no
    embedding-input hash, so it cannot prove the vector was generated from that
    text. Reuse is therefore limited to rows whose stored metadata and model
    metadata match exactly; callers must preserve the provenance limitation.
    """
    requested_ids = [namespaced_rag_id(row["ticket_id"]) for row in incoming_rows]
    incoming_by_id = {
        namespaced_rag_id(row["ticket_id"]): row for row in incoming_rows
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT rag_id, source_ticket_id, dataset_name, category, rag_text
               FROM public.synthetic_tickets
               WHERE rag_id = ANY(%s) OR source_ticket_id = ANY(%s)""",
            (requested_ids, [row["ticket_id"] for row in incoming_rows]),
        )
        stored_rows = cursor.fetchall()
        stored_tickets: dict[str, dict[str, Any]] = {}
        for rag_id, source_ticket_id, dataset_name, category, rag_text in stored_rows:
            rag_id = str(rag_id)
            if rag_id not in incoming_by_id:
                raise ValueError(
                    f"Conflicting existing synthetic ticket ID for source ticket "
                    f"{source_ticket_id}: stored rag_id {rag_id} is not the expected namespaced ID; "
                    "no existing data was changed"
                )
            stored_tickets[rag_id] = {
                "source_ticket_id": source_ticket_id,
                "dataset_name": dataset_name,
                "category": category,
                "rag_text": rag_text,
            }
        for rag_id, stored in stored_tickets.items():
            validate_existing_ticket_metadata(incoming_by_id[rag_id], stored)

        cursor.execute(
            """SELECT e.ticket_id
               FROM public.synthetic_ticket_embeddings AS e
               LEFT JOIN public.synthetic_tickets AS t ON t.rag_id = e.ticket_id
               WHERE t.rag_id IS NULL
               LIMIT 1"""
        )
        orphan = cursor.fetchone()
        if orphan:
            raise RuntimeError(
                f"Orphaned synthetic embedding {orphan[0]} has no synthetic ticket record; "
                "no existing data was changed"
            )

        cursor.execute(
            """SELECT ticket_id, embedding_model, embedding_dimensions
               FROM public.synthetic_ticket_embeddings WHERE ticket_id = ANY(%s)""",
            (requested_ids,),
        )
        embedding_rows = cursor.fetchall()

    ticket_ids = set(stored_tickets)
    embedding_ids: set[str] = set()
    for raw_ticket_id, model, dimensions in embedding_rows:
        ticket_id = str(raw_ticket_id)
        # The orphan query above checks the whole synthetic embedding table;
        # this guard also protects against unusual/faulty mocked or changed SQL results.
        if ticket_id not in ticket_ids:
            raise RuntimeError(
                f"Orphaned synthetic embedding {ticket_id} has no verified ticket record; "
                "no existing data was changed"
            )
        validate_existing_embedding(ticket_id, model, dimensions)
        embedding_ids.add(ticket_id)
    return ticket_ids, embedding_ids


def ingest_training_rows(rows: list[dict[str, str]], database_url: str, *, batch_size: int = 50) -> dict[str, Any]:
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Ingestion accepts training rows only")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = None
    processed = new_tickets = new_embeddings = reused_embeddings = 0
    with psycopg.connect(
        database_url, connect_timeout=20, autocommit=True,
        options="-c statement_timeout=30000 -c lock_timeout=10000",
    ) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        source_ids = [row["ticket_id"] for row in rows]
        rag_ids = [namespaced_rag_id(ticket_id) for ticket_id in source_ids]
        collision_ids = database_existing_ids(connection, source_ids, rag_ids)
        if collision_ids:
            raise ValueError("Synthetic dataset ID collision with existing public.tickets; no rows were written")
        existing_tickets, existing_embeddings = existing_synthetic_state(connection, rows)
        pending_embedding_ids = {namespaced_rag_id(row["ticket_id"]) for row in rows} - existing_embeddings
        if pending_embedding_ids:
            model = load_embedding_model()
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            with connection.transaction():
                # Hold write-blocking locks while checking and inserting this
                # batch so a concurrent ingest cannot create a conflicting row
                # between validation and ON CONFLICT DO NOTHING.
                connection.execute("""LOCK TABLE
                    public.tickets,
                    public.synthetic_tickets,
                    public.synthetic_ticket_embeddings
                    IN SHARE ROW EXCLUSIVE MODE""")
                batch_source_ids = [row["ticket_id"] for row in batch]
                batch_rag_ids = [namespaced_rag_id(ticket_id) for ticket_id in batch_source_ids]
                collisions = database_existing_ids(connection, batch_source_ids, batch_rag_ids)
                if collisions:
                    raise ValueError(
                        "Synthetic dataset ID collision with existing public.tickets; "
                        "no rows in this batch were written"
                    )
                batch_tickets, batch_embeddings = existing_synthetic_state(connection, batch)
                needing_vectors = [
                    row for row in batch
                    if namespaced_rag_id(row["ticket_id"]) not in batch_embeddings
                ]
                vectors: dict[str, Vector] = {}
                if needing_vectors:
                    if model is None:
                        model = load_embedding_model()
                    encoded = encode_texts(
                        model,
                        [synthetic_embedding_text(row) for row in needing_vectors],
                        batch_size=min(16, batch_size),
                    )
                    vectors = {
                        namespaced_rag_id(row["ticket_id"]): Vector(vector)
                        for row, vector in zip(needing_vectors, encoded, strict=True)
                    }
                with connection.cursor() as cursor:
                    for row in batch:
                        cursor.execute(INSERT_TICKET_SQL, ticket_values(row))
                        db_id = namespaced_rag_id(row["ticket_id"])
                        if db_id in vectors:
                            cursor.execute(
                                INSERT_EMBEDDING_SQL,
                                (db_id, vectors[db_id], MODEL_ID, EMBEDDING_DIMENSIONS),
                            )
            processed += len(batch)
            new_tickets += sum(namespaced_rag_id(row["ticket_id"]) not in batch_tickets for row in batch)
            new_embeddings += len(vectors)
            reused_embeddings += len(batch) - len(needing_vectors)
            print(
                f"Synthetic train ingestion progress: {processed}/{len(rows)}; "
                f"new tickets={new_tickets}, new embeddings={new_embeddings}, "
                f"reused embeddings={reused_embeddings}"
            )
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.synthetic_tickets WHERE split = 'train'")
            ticket_count = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM public.synthetic_ticket_embeddings")
            embedding_count = cursor.fetchone()[0]
    return {
        "processed_train_rows": processed,
        "new_tickets": new_tickets,
        "new_embeddings": new_embeddings,
        "reused_embeddings": reused_embeddings,
        "synthetic_train_ticket_count": int(ticket_count),
        "synthetic_embedding_count": int(embedding_count),
        "existing_vector_provenance_limitation": EXISTING_VECTOR_PROVENANCE_NOTE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    rows = select_training_rows(load_dataset())
    database_url = os.getenv("SUPABASE_DB_URL", "")
    if not database_url:
        raise SystemExit("SUPABASE_DB_URL is not set; no database connection was attempted")
    print(f"Dataset: {DATASET_NAME} (synthetic); training rows only: {len(rows)}")
    try:
        result = ingest_training_rows(rows, database_url, batch_size=args.batch_size)
    except Exception as error:
        # Never expose credentials or ticket descriptions in operational output.
        print(f"Synthetic ingestion stopped: {type(error).__name__}: operation aborted safely")
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
