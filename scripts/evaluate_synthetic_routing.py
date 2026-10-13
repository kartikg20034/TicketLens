#!/usr/bin/env python3
"""Evaluate synthetic-ticket BGE retrieval and TF-IDF baselines.

Validation is the default. The test split requires --final-comparison to make
its held-back status an explicit choice. No rows are written to the database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_embeddings import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    encode_texts,
    load_embedding_model,
)
from scripts.evaluate_rag_retrieval import classification_metrics, majority_vote  # noqa: E402
from scripts.ingest_synthetic_train import (  # noqa: E402
    DATASET_NAME,
    LABELS,
    load_dataset,
    namespaced_rag_id,
    select_training_rows,
    synthetic_embedding_text,
)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

MATCH_COUNT = 5
QUERY_BATCH_SIZE = 32
PROGRESS_INTERVAL = 25
REPORT_DIR = PROJECT_ROOT / "reports"


def tfidf_feature(row: dict[str, str]) -> str:
    """Use rag_text alone; category is accepted only as separate scoring y."""
    return synthetic_embedding_text(row)


def validate_test_release(split: str, final_comparison: bool) -> None:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split == "test" and not final_comparison:
        raise ValueError("Test evaluation is held back; pass --final-comparison only for the final comparison")


def retrieval_function_isolated(definition: str | None) -> bool:
    if not definition:
        return False
    normalized = definition.casefold()
    return all(
        phrase in normalized
        for phrase in (
            "synthetic_ticket_embeddings",
            "synthetic_tickets",
            "split = 'train'",
            "is_synthetic is true",
            f"embedding_model = '{MODEL_ID.casefold()}'",
        )
    )


def function_definition(connection: psycopg.Connection) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT pg_get_functiondef(p.oid)
               FROM pg_proc AS p
               JOIN pg_namespace AS n ON n.oid = p.pronamespace
               WHERE n.nspname = 'public' AND p.proname = 'match_synthetic_train_tickets'
               ORDER BY p.oid DESC LIMIT 1"""
        )
        result = cursor.fetchone()
    return result[0] if result else None


def tfidf_metrics(train_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not train_rows or not evaluation_rows:
        raise ValueError("TF-IDF requires training and evaluation rows")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("TF-IDF training input contains a non-training row")
    if len({row["ticket_id"] for row in train_rows} & {row["ticket_id"] for row in evaluation_rows}):
        raise ValueError("Training and evaluation ticket IDs overlap")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    train_text = [tfidf_feature(row) for row in train_rows]
    train_y = [row["category"] for row in train_rows]
    eval_text = [tfidf_feature(row) for row in evaluation_rows]
    eval_y = [row["category"] for row in evaluation_rows]
    model.fit(train_text, train_y)
    predictions = model.predict(eval_text).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        eval_y, predictions, labels=list(LABELS), zero_division=0
    )
    matrix = confusion_matrix(eval_y, predictions, labels=list(LABELS))
    return {
        "model": "TF-IDF(1,2-grams) + LogisticRegression(max_iter=1000, random_state=0)",
        "feature_source": "rag_text only; category is used only as y",
        "training_split": "train",
        "training_ticket_count": len(train_rows),
        "evaluation_ticket_count": len(evaluation_rows),
        "evaluation_rows_used_for_fit": 0,
        "accuracy": float(accuracy_score(eval_y, predictions)),
        "macro_f1": float(sum(f1) / len(LABELS)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            actual: {
                predicted: int(matrix[i, j])
                for j, predicted in enumerate(LABELS)
            }
            for i, actual in enumerate(LABELS)
        },
    }


def validate_retrieved(
    connection: psycopg.Connection, matches: list[tuple[Any, ...]]
) -> list[dict[str, Any]]:
    if len(matches) != MATCH_COUNT:
        raise ValueError(f"Expected {MATCH_COUNT} synthetic train matches, got {len(matches)}")
    ids = [str(match[0]) for match in matches]
    if len(set(ids)) != len(ids):
        raise ValueError("Synthetic retrieval returned duplicate IDs")
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT t.rag_id, t.split, t.is_synthetic, e.embedding_model, e.embedding_dimensions
               FROM public.synthetic_tickets t
               JOIN public.synthetic_ticket_embeddings e ON e.ticket_id = t.rag_id
               WHERE t.rag_id = ANY(%s)""",
            (ids,),
        )
        metadata = {str(row[0]): row[1:] for row in cursor.fetchall()}
    if set(metadata) != set(ids):
        raise ValueError("Retrieved ID is absent from the isolated synthetic corpus")
    for ticket_id, (split, is_synthetic, model, dimensions) in metadata.items():
        if split != "train" or not is_synthetic:
            raise ValueError(f"Retrieved ID {ticket_id} is not a synthetic training ticket")
        if model != MODEL_ID or int(dimensions) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Retrieved ID {ticket_id} has an unexpected embedding model or size")
    normalized = []
    for match in matches:
        if len(match) != 5:
            raise ValueError("Synthetic retrieval function returned an unexpected result shape")
        ticket_id, subject, app_name, category, similarity = match
        if category not in LABELS:
            raise ValueError(f"Retrieved ID {ticket_id} has an unknown category")
        normalized.append({
            "ticket_id": str(ticket_id),
            "subject": subject,
            "app_name": app_name,
            "category": category,
            "similarity": float(similarity),
        })
    return sorted(normalized, key=lambda item: (-item["similarity"], item["ticket_id"]))


def evaluate_retrieval(
    train_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    database_url: str,
) -> dict[str, Any]:
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("RAG reference input contains a non-training row")
    if any(row.get("split") not in {"validation", "test"} for row in evaluation_rows):
        raise ValueError("RAG query input contains a train or unknown split row")
    model = load_embedding_model()
    predictions: list[dict[str, str]] = []
    failures: Counter[str] = Counter()
    retrieved_result_count = 0
    with psycopg.connect(
        database_url,
        connect_timeout=20,
        autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    ) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        definition = function_definition(connection)
        if not retrieval_function_isolated(definition):
            raise RuntimeError(
                "match_synthetic_train_tickets is missing its synthetic/train/BGE filters; "
                "apply migration 003 before evaluating"
            )
        for start in range(0, len(evaluation_rows), QUERY_BATCH_SIZE):
            batch = evaluation_rows[start:start + QUERY_BATCH_SIZE]
            try:
                vectors = encode_texts(
                    model,
                    [synthetic_embedding_text(row) for row in batch],
                    batch_size=QUERY_BATCH_SIZE,
                )
            except Exception as error:
                failures[type(error).__name__] += len(batch)
                print(f"Embedding failed for query batch beginning at {start + 1}: {type(error).__name__}")
                continue
            for row, vector in zip(batch, vectors, strict=True):
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """SELECT ticket_id, subject, app_name, category, similarity
                               FROM public.match_synthetic_train_tickets(
                                 %s::extensions.vector(768), %s
                               )""",
                            (Vector(vector), MATCH_COUNT),
                        )
                        matches = cursor.fetchall()
                    retrieved_result_count += len(matches)
                    retrieved = validate_retrieved(connection, matches)
                    predictions.append({
                        "truth": row["category"],
                        "top1": retrieved[0]["category"],
                        "top5": majority_vote([item["category"] for item in retrieved]),
                    })
                except Exception as error:
                    failures[type(error).__name__] += 1
                    print(f"Retrieval failed for one {row['split']} query: {type(error).__name__}; excluded from scores")
            done = start + len(batch)
            if done % PROGRESS_INTERVAL < len(batch) or done == len(evaluation_rows):
                print(
                    f"Synthetic {evaluation_rows[0]['split']} progress: {done}/{len(evaluation_rows)}; "
                    f"successful={len(predictions)}; failures={sum(failures.values())}; "
                    f"retrieved_results={retrieved_result_count}"
                )
    truths = [item["truth"] for item in predictions]
    return {
        "total_queries": len(evaluation_rows),
        "successful_queries": len(predictions),
        "failed_queries": sum(failures.values()),
        "failure_counts_by_type": dict(sorted(failures.items())),
        "retrieved_results": retrieved_result_count,
        "top1": classification_metrics(truths, [item["top1"] for item in predictions]),
        "top5_majority": classification_metrics(truths, [item["top5"] for item in predictions]),
    }


def make_report(
    split: str,
    train_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    retrieval: dict[str, Any],
    tfidf: dict[str, Any],
    *,
    final_comparison: bool,
) -> dict[str, Any]:
    return {
        "title": "Synthetic Ticket Routing Experiment",
        "dataset": DATASET_NAME,
        "synthetic_data_notice": "All records are synthetic. Metrics are for pipeline demonstration and do not represent real-world performance.",
        "evaluation_split": split,
        "test_results_released": bool(split == "test" and final_comparison),
        "reference_split": "train",
        "reference_ticket_count": len(train_rows),
        "evaluation_ticket_count": len(evaluation_rows),
        "target_label": "category",
        "embedding_input": "rag_text only; category excluded",
        "embedding_model": MODEL_ID,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "validation_and_test_ingested": False,
        "bge_retrieval": retrieval,
        "tfidf_logistic_regression": tfidf,
        "interpretation": "Compare methods on the same selected synthetic split. Validation should be used for iteration; inspect the test report only after finalizing choices.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-comparison", action="store_true")
    parser.add_argument(
        "--tfidf-only", action="store_true",
        help="run and save the train-only TF-IDF baseline without connecting to Supabase",
    )
    args = parser.parse_args()
    validate_test_release(args.split, args.final_comparison)
    rows = load_dataset()
    train_rows = select_training_rows(rows)
    evaluation_rows = [row for row in rows if row["split"] == args.split]
    tfidf = tfidf_metrics(train_rows, evaluation_rows)
    if args.tfidf_only:
        retrieval = {
            "status": "not_run_by_request",
            "total_queries": len(evaluation_rows),
            "successful_queries": None,
            "failed_queries": None,
            "failure_counts_by_type": {},
            "retrieved_results": 0,
            "top1": classification_metrics([], []),
            "top5_majority": classification_metrics([], []),
        }
    else:
        database_url = os.getenv("SUPABASE_DB_URL", "")
        if not database_url:
            raise SystemExit("SUPABASE_DB_URL is not set; no database connection was attempted")
        try:
            retrieval = evaluate_retrieval(train_rows, evaluation_rows, database_url)
        except Exception as error:
            retrieval = {
                "total_queries": len(evaluation_rows),
                "successful_queries": 0,
                "failed_queries": len(evaluation_rows),
                "failure_counts_by_type": {type(error).__name__: len(evaluation_rows)},
                "retrieved_results": 0,
                "top1": classification_metrics([], []),
                "top5_majority": classification_metrics([], []),
                "fatal_error_type": type(error).__name__,
            }
            print(f"BGE retrieval unavailable: {type(error).__name__}; TF-IDF comparison remains valid.")
    report = make_report(args.split, train_rows, evaluation_rows, retrieval, tfidf, final_comparison=args.final_comparison)
    output = REPORT_DIR / f"synthetic_routing_{args.split}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "split": args.split, "bge_successful": retrieval["successful_queries"], "bge_failures": retrieval["failed_queries"], "tfidf_accuracy": tfidf["accuracy"], "tfidf_macro_f1": tfidf["macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
