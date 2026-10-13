#!/usr/bin/env python3
"""Evaluate local BGE retrieval against every held-out evaluation ticket.

This is a read-only evaluator: evaluation rows are only embedded as queries;
the Supabase function supplies reference-only candidate tickets.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
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
    retrieval_function_uses_model,
)
from scripts.test_rag_retrieval import (  # noqa: E402
    evaluation_query_text,
    function_definition,
    redact_error,
)

load_dotenv(dotenv_path=".env")

EVALUATION_CSV = PROJECT_ROOT / "data/processed/rag_evaluation.csv"
MATCH_COUNT = 5
EMBED_BATCH_SIZE = 32
PROGRESS_INTERVAL = 50
LABELS = (
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
)


def majority_vote(labels: list[str]) -> str:
    """Choose the most frequent label; tied votes go to the highest-ranked result."""
    if not labels:
        raise ValueError("Cannot vote without retrieved labels")
    counts = Counter(labels)
    best_count = max(counts.values())
    return next(label for label in labels if counts[label] == best_count)


def classification_metrics(
    truths: list[str], predictions: list[str], labels: tuple[str, ...] = LABELS
) -> dict:
    if len(truths) != len(predictions):
        raise ValueError("Truth and prediction counts differ")
    if not truths:
        return {
            "accuracy": None,
            "macro_f1": None,
            "per_class": {
                label: {"precision": None, "recall": None, "f1": None}
                for label in labels
            },
            "confusion_matrix": {actual: {predicted: 0 for predicted in labels} for actual in labels},
        }

    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for truth, prediction in zip(truths, predictions, strict=True):
        if truth not in matrix or prediction not in matrix:
            raise ValueError("Truth or prediction contains a label outside the five categories")
        matrix[truth][prediction] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[other][label] for other in labels if other != label)
        false_negative = sum(matrix[label][other] for other in labels if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "accuracy": sum(truth == pred for truth, pred in zip(truths, predictions, strict=True)) / len(truths),
        "macro_f1": sum(values["f1"] for values in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def read_evaluation_rows() -> list[dict[str, str]]:
    required = {"Ticket ID", "Ticket Subject", "Ticket Type", "rag_text", "split"}
    with EVALUATION_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Evaluation CSV contains no tickets")
    for index, row in enumerate(rows, start=1):
        if row["split"] != "evaluation":
            raise ValueError(f"CSV row {index} is not marked as evaluation")
        if row["Ticket Type"] not in LABELS:
            raise ValueError(f"CSV row {index} has an unexpected Ticket Type")
        if not row["Ticket Subject"].strip() or not row["rag_text"].strip():
            raise ValueError(f"CSV row {index} is missing its subject or rag_text")
        # The ground-truth field is kept separate and never passed to the encoder.
        row["query_text"] = evaluation_query_text(row["rag_text"])
        if not row["query_text"]:
            raise ValueError(f"CSV row {index} has empty query text after label removal")
    return rows


def validate_matches(connection: psycopg.Connection, matches: list[tuple]) -> list[dict]:
    if len(matches) != MATCH_COUNT:
        raise ValueError(f"Expected {MATCH_COUNT} results, received {len(matches)}")
    ticket_ids = [str(result[0]) for result in matches]
    if len(set(ticket_ids)) != len(ticket_ids):
        raise ValueError("Retrieval returned duplicate ticket IDs")

    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT t.rag_id, t.split, e.embedding_model, e.embedding_dimensions
               FROM public.tickets AS t
               JOIN public.ticket_embeddings AS e ON e.ticket_id = t.rag_id
               WHERE t.rag_id = ANY(%s)""",
            (ticket_ids,),
        )
        metadata = {str(row[0]): row[1:] for row in cursor.fetchall()}

    if set(metadata) != set(ticket_ids):
        raise ValueError("One or more retrieved IDs do not resolve to stored reference embeddings")
    for ticket_id, (split, embedding_model, dimensions) in metadata.items():
        if split != "reference":
            raise ValueError(f"Retrieved ticket {ticket_id} is outside the reference split")
        if embedding_model != MODEL_ID or int(dimensions) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Retrieved ticket {ticket_id} does not use the required BGE embedding")

    normalized = []
    for result in matches:
        if len(result) != 6:
            raise ValueError("Retrieval function returned an unexpected result shape")
        ticket_id, subject, product, ticket_type, resolution, similarity = result
        if ticket_type not in LABELS:
            raise ValueError(f"Retrieved ticket {ticket_id} has an unknown Ticket Type")
        normalized.append({
            "ticket_id": str(ticket_id),
            "subject": subject,
            "product": product,
            "ticket_type": ticket_type,
            "resolution": resolution,
            "similarity": float(similarity),
        })
    # The SQL function orders by cosine distance. Add a stable secondary key
    # for exact similarity ties before applying deterministic majority voting.
    return sorted(normalized, key=lambda result: (-result["similarity"], result["ticket_id"]))


def retrieve(connection: psycopg.Connection, query_embedding: Vector) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT ticket_id, subject, product, ticket_type, resolution, similarity
               FROM public.match_reference_tickets(%s::extensions.vector(768), %s)""",
            (query_embedding, MATCH_COUNT),
        )
        return cursor.fetchall()


def evaluate(rows: list[dict[str, str]], database_url: str) -> dict:
    model = load_embedding_model()
    predictions: list[dict[str, str]] = []
    failure_counts: Counter[str] = Counter()
    retrieved_results = 0

    # A session-level default prevents accidental writes even if the retrieval
    # function or a future code change attempts to issue a mutation.
    connect_options = "-c default_transaction_read_only=on -c statement_timeout=30000"
    with psycopg.connect(
        database_url, connect_timeout=20, autocommit=True, options=connect_options
    ) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        definition = function_definition(connection)
        if not retrieval_function_uses_model(definition):
            raise RuntimeError(
                "Database match_reference_tickets is not restricted to reference BGE rows. "
                "Apply supabase/migrations/002_bge_retrieval_filter.sql first."
            )

        total = len(rows)
        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch = rows[batch_start:batch_start + EMBED_BATCH_SIZE]
            try:
                vectors = encode_texts(
                    model, [row["query_text"] for row in batch], batch_size=EMBED_BATCH_SIZE
                )
            except Exception as error:
                failure_counts[type(error).__name__] += len(batch)
                completed = batch_start + len(batch)
                print(
                    f"Queries {batch_start + 1}-{completed}/{total} embedding failed: "
                    f"{type(error).__name__}; excluded from metrics"
                )
                if completed % PROGRESS_INTERVAL < len(batch) or completed == total:
                    print(
                        f"Progress: {completed}/{total}; successful={len(predictions)}; "
                        f"failures={sum(failure_counts.values())}; retrieved_results={retrieved_results}"
                    )
                continue
            for offset, (row, vector) in enumerate(zip(batch, vectors, strict=True)):
                query_number = batch_start + offset + 1
                try:
                    matches = retrieve(connection, Vector(vector))
                    retrieved_results += len(matches)
                    verified = validate_matches(connection, matches)
                    top1 = verified[0]["ticket_type"]
                    top5 = majority_vote([item["ticket_type"] for item in verified])
                    predictions.append({
                        "truth": row["Ticket Type"],
                        "top1": top1,
                        "top5": top5,
                    })
                except Exception as error:
                    failure_counts[type(error).__name__] += 1
                    print(
                        f"Query {query_number}/{total} failed: {type(error).__name__}; "
                        "excluded from metrics"
                    )

                completed = query_number
                if completed % PROGRESS_INTERVAL == 0 or completed == total:
                    print(
                        f"Progress: {completed}/{total}; successful={len(predictions)}; "
                        f"failures={sum(failure_counts.values())}; retrieved_results={retrieved_results}"
                    )

    truths = [record["truth"] for record in predictions]
    top1_metrics = classification_metrics(truths, [record["top1"] for record in predictions])
    top5_metrics = classification_metrics(truths, [record["top5"] for record in predictions])
    return {
        "total_queries": len(rows),
        "successful_queries": len(predictions),
        "failed_queries": sum(failure_counts.values()),
        "retrieved_results": retrieved_results,
        "failure_counts_by_type": dict(sorted(failure_counts.items())),
        "top1": top1_metrics,
        "top5_majority": top5_metrics,
    }


def print_report(report: dict) -> None:
    print("\nRAG retrieval evaluation")
    print(f"Model: {MODEL_ID} ({EMBEDDING_DIMENSIONS} dimensions)")
    print(f"Queries: {report['total_queries']}")
    print(f"Successful queries: {report['successful_queries']}")
    print(f"Failed queries: {report['failed_queries']}")
    print(f"Retrieved results: {report['retrieved_results']}")
    if report["failure_counts_by_type"]:
        print("Failure types: " + json.dumps(report["failure_counts_by_type"], sort_keys=True))

    def score(value):
        return f"{value:.4f}" if value is not None else "n/a"

    for name, metrics in (("Top-1", report["top1"]), ("Top-5 majority vote", report["top5_majority"])):
        print(f"\n{name}")
        print(f"Accuracy: {score(metrics['accuracy'])}")
        print(f"Macro F1: {score(metrics['macro_f1'])}")
        if metrics["per_class"]:
            print("Per-class metrics (precision / recall / F1):")
            for label in LABELS:
                values = metrics["per_class"][label]
                print(
                    f"  {label}: {score(values['precision'])} / "
                    f"{score(values['recall'])} / {score(values['f1'])}"
                )

    matrix = report["top5_majority"].get("confusion_matrix")
    if matrix is not None:
        print("\nTop-5 majority confusion matrix (rows=true, columns=predicted):")
        print("true \\ predicted | " + " | ".join(LABELS))
        for actual in LABELS:
            print(actual + " | " + " | ".join(str(matrix[actual][predicted]) for predicted in LABELS))


def main() -> int:
    database_url = os.getenv("SUPABASE_DB_URL", "")
    if not database_url:
        print("Required environment variable unavailable: SUPABASE_DB_URL")
        return 1
    try:
        rows = read_evaluation_rows()
        print(f"Loaded {len(rows)} evaluation queries; no evaluation rows will be inserted.")
        report = evaluate(rows, database_url)
        print_report(report)
        return 0 if report["successful_queries"] else 1
    except Exception as error:
        print("Evaluation stopped: " + redact_error(error, database_url))
        return 1


if __name__ == "__main__":
    sys.exit(main())
