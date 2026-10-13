#!/usr/bin/env python3
"""Recommend a synthetic routing category for one ticket using BGE retrieval.

Reads ticket text from --text or standard input. It performs a read-only query
against the isolated synthetic training corpus and never writes ticket data.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
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
from scripts.evaluate_rag_retrieval import majority_vote  # noqa: E402
from scripts.evaluate_synthetic_routing import (  # noqa: E402
    function_definition,
    retrieval_function_isolated,
    validate_retrieved,
)
from scripts.ingest_synthetic_train import LABELS  # noqa: E402

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
MATCH_COUNT = 5
TARGET_FIELD = re.compile(r"(?im)^\s*(?:category|ticket\s+type)\s*:")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)\+?\d(?:[\d .()-]{6,}\d)(?!\w)")
SECRET = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|password|secret|authorization)\b\s*[:=]\s*(?:Bearer\s+)?[^\s,;]+"
)


def make_query_text(ticket_text: str, subject: str | None = None) -> str:
    """Build an input document without a target category or target field."""
    description = ticket_text.strip()
    clean_subject = (subject or "").strip()
    if not description and not clean_subject:
        raise ValueError("Provide ticket text with --text or through standard input")
    if TARGET_FIELD.search(description) or TARGET_FIELD.search(clean_subject):
        raise ValueError("Input must not contain a Category or Ticket Type target field")
    text = (
        f"Subject: {clean_subject}\nDescription: {description}"
        if clean_subject
        else f"Description: {description}"
    )
    if TARGET_FIELD.search(text):
        raise ValueError("Input must not contain a Category or Ticket Type target field")
    return text


def safe_preview(text: str, limit: int = 320) -> str:
    """Return a bounded preview with common secrets and email addresses redacted."""
    preview = SECRET.sub(r"\1=[REDACTED]", text)
    preview = EMAIL.sub("[EMAIL REDACTED]", preview)
    preview = PHONE.sub("[PHONE REDACTED]", preview)
    preview = " ".join(preview.split())
    if len(preview) > limit:
        preview = preview[: max(limit - 1, 0)].rstrip() + "…"
    return preview


def summarize_retrieval(query_text: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    if len(retrieved) != MATCH_COUNT:
        raise ValueError(f"Expected {MATCH_COUNT} training examples, received {len(retrieved)}")
    labels = [item["category"] for item in retrieved]
    if any(label not in LABELS for label in labels):
        raise ValueError("Retrieved a category outside the five supported labels")
    top1 = labels[0]
    top5 = majority_vote(labels)
    nearest = retrieved[0]
    votes = {label: labels.count(label) for label in LABELS if label in labels}
    rationale = (
        f"The nearest training example is {safe_preview(str(nearest['subject']), 120)!r} "
        f"({nearest['similarity']:.3f} cosine similarity) and has category {top1!r}. "
        f"The five retrieved examples vote {votes}."
    )
    return {
        "ticket_preview": safe_preview(query_text),
        "top1_category_recommendation": top1,
        "top5_majority_category": top5,
        "team_recommendation": None,
        "team_warning": "Unsupported: no validated category-to-team mapping is defined.",
        "rationale": rationale,
        "similar_training_tickets": [
            {
                "subject": safe_preview(str(item["subject"]), 160),
                "category": item["category"],
                "similarity": round(float(item["similarity"]), 4),
            }
            for item in retrieved
        ],
    }


def retrieve_one(query_text: str, database_url: str) -> dict[str, Any]:
    model = load_embedding_model()
    vector = encode_texts(model, [query_text], batch_size=1)[0]
    with psycopg.connect(
        database_url,
        connect_timeout=20,
        autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    ) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        if not retrieval_function_isolated(function_definition(connection)):
            raise RuntimeError("The synthetic retrieval function is missing its train-only BGE filters")
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT ticket_id, subject, app_name, category, similarity
                   FROM public.match_synthetic_train_tickets(
                     %s::extensions.vector(768), %s
                   )""",
                (Vector(vector), MATCH_COUNT),
            )
            matches = cursor.fetchall()
        retrieved = validate_retrieved(connection, matches)
    if EMBEDDING_DIMENSIONS != 768 or MODEL_ID != "BAAI/bge-base-en-v1.5":
        raise RuntimeError("The configured embedding model or dimension does not match the experiment")
    return summarize_retrieval(query_text, retrieved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="ticket description; if omitted, read it from standard input")
    parser.add_argument("--subject", help="optional ticket subject")
    parser.add_argument("--preview-limit", type=int, default=320)
    args = parser.parse_args(argv)
    if args.preview_limit < 40 or args.preview_limit > 1000:
        parser.error("--preview-limit must be between 40 and 1000")
    ticket_text = args.text if args.text is not None else sys.stdin.read()
    try:
        query_text = make_query_text(ticket_text, args.subject)
        database_url = os.getenv("SUPABASE_DB_URL", "")
        if not database_url:
            raise RuntimeError("SUPABASE_DB_URL is not set")
        result = retrieve_one(query_text, database_url)
        result["ticket_preview"] = safe_preview(query_text, args.preview_limit)
    except Exception as error:
        # Exception text can contain connection details; print only the class.
        print(f"Routing failed ({type(error).__name__}); ticket and credential values were not printed.", file=sys.stderr)
        return 2
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("This is a category recommendation from synthetic training examples, not an operational assignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
