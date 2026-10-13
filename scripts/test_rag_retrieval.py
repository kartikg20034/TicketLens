#!/usr/bin/env python3
"""Evaluate retrieval for one held-out ticket using local BGE embeddings."""

from __future__ import annotations

import csv
import os
import re
import sys
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
    retrieval_function_uses_model,
)

load_dotenv(dotenv_path=".env")

EVALUATION_CSV = PROJECT_ROOT / "data/processed/rag_evaluation.csv"
MATCH_COUNT = 5


def evaluation_query_text(rag_text: str) -> str:
    """Drop a Ticket Type line if present so the target cannot be a query feature."""
    query_text = re.sub(
        r"(?im)^\s*Ticket Type\s*:[^\r\n]*(?:\r?\n|$)", "", rag_text
    ).strip()
    if re.search(r"(?im)^\s*Ticket Type\s*:", query_text):
        raise ValueError("Evaluation query still contains Ticket Type; refusing label leakage")
    return query_text


def read_first_evaluation_ticket() -> dict[str, str]:
    with EVALUATION_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Ticket Subject", "Ticket Type", "rag_text", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV is missing required columns: {sorted(missing)}")
        row = next(reader, None)
    if row is None:
        raise ValueError("Evaluation CSV contains no tickets")
    if row["split"] != "evaluation":
        raise ValueError("First CSV row is not marked as an evaluation ticket")
    if not row["rag_text"].strip():
        raise ValueError("First evaluation ticket has empty rag_text")
    if not row["Ticket Subject"].strip() or not row["Ticket Type"].strip():
        raise ValueError("First evaluation ticket is missing its subject or true Ticket Type")
    # The prepared rag_text template contains Ticket Type. Remove that target
    # field before creating the evaluation query vector.
    row["query_text"] = evaluation_query_text(row["rag_text"])
    if not row["query_text"]:
        raise ValueError("Evaluation query is empty after removing the target label")
    return row


def function_definition(connection: psycopg.Connection) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT pg_get_functiondef(p.oid)
               FROM pg_proc AS p
               JOIN pg_namespace AS n ON n.oid = p.pronamespace
               WHERE n.nspname = 'public' AND p.proname = 'match_reference_tickets'
               ORDER BY p.oid DESC LIMIT 1"""
        )
        result = cursor.fetchone()
    return result[0] if result else None


def retrieve_reference_tickets(database_url: str, query_embedding: Vector) -> list[tuple]:
    with psycopg.connect(database_url, connect_timeout=15) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        definition = function_definition(connection)
        if not retrieval_function_uses_model(definition):
            raise RuntimeError(
                "Database retrieval function is not restricted to BGE vectors. Apply "
                "supabase/migrations/002_bge_retrieval_filter.sql before running retrieval."
            )
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT subject, ticket_type, similarity
                   FROM public.match_reference_tickets(%s::extensions.vector(768), %s)""",
                (query_embedding, MATCH_COUNT),
            )
            return cursor.fetchall()


def redact_error(error: BaseException, database_url: str) -> str:
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
        message = re.sub(r"(?i)(password\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", message)
    return message or error.__class__.__name__


def main() -> int:
    database_url = os.getenv("SUPABASE_DB_URL", "")
    if not database_url:
        print("Required environment variable unavailable: SUPABASE_DB_URL")
        return 1
    try:
        ticket = read_first_evaluation_ticket()
        model = load_embedding_model()
        query_embedding = Vector(encode_texts(model, [ticket["query_text"]])[0])
        matches = retrieve_reference_tickets(database_url, query_embedding)

        print(f"Embedding model: {MODEL_ID} ({EMBEDDING_DIMENSIONS} dimensions)")
        print(f"Evaluation ticket subject: {ticket['Ticket Subject']}")
        print(f"True Ticket Type (evaluation only): {ticket['Ticket Type']}")
        print(f"Retrieved reference tickets ({len(matches)}):")
        for index, (subject, ticket_type, similarity) in enumerate(matches, start=1):
            print(f"{index}. Subject: {subject}")
            print(f"   Ticket Type: {ticket_type}")
            print(f"   Similarity: {float(similarity):.4f}")
        return 0
    except Exception as error:
        print("Evaluation retrieval failed: " + redact_error(error, database_url))
        return 1


if __name__ == "__main__":
    sys.exit(main())
