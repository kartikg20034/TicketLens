#!/usr/bin/env python3
"""Read-only investigation of the local BGE + Supabase RAG classifier."""

from __future__ import annotations

import csv
import json
import os
import random
import re
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
from scripts.evaluate_rag_retrieval import (  # noqa: E402
    LABELS,
    classification_metrics,
    majority_vote,
    read_evaluation_rows,
    retrieve,
    validate_matches,
)
from scripts.test_rag_retrieval import function_definition, redact_error  # noqa: E402

load_dotenv(dotenv_path=".env")

REFERENCE_CSV = PROJECT_ROOT / "data/processed/rag_reference.csv"
EVALUATION_CSV = PROJECT_ROOT / "data/processed/rag_evaluation.csv"
INGESTION_SCRIPT = PROJECT_ROOT / "scripts/ingest_reference_tickets.py"
EVALUATOR_SCRIPT = PROJECT_ROOT / "scripts/evaluate_rag_retrieval.py"
REPORT_PATH = PROJECT_ROOT / "reports/rag_retrieval_diagnostic.json"
COMPARISON_SEED = 20261011
COMPARISON_PER_LABEL = 10
EMBED_BATCH_SIZE = 32
PROGRESS_INTERVAL = 50
EXAMPLE_COUNT = 20
EXAMPLE_QUERY_MAX = 280
EXAMPLE_SUBJECT_MAX = 110

GENERIC_PHRASES = (
    "please assist",
    "please help",
    "i am having an issue",
    "i'm having an issue",
    "the issue persists",
    "doesn't work",
    "not working",
    "thank you",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader)


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def truncate(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def text_audit(rows: list[dict[str, str]], *, is_reference: bool) -> dict:
    exact_texts = [normalize_text(row.get("rag_text", "")) for row in rows]
    subject_norms = [normalize_text(row.get("Ticket Subject", "")) for row in rows]
    description_norms = [normalize_text(row.get("Ticket Description", "")) for row in rows]
    duplicate_text_counts = Counter(text for text in exact_texts if text)
    repeated_subject_counts = Counter(subject for subject in subject_norms if subject)
    ticket_type_line_count = 0
    ticket_type_value_matches = 0
    unexpected_headings: Counter[str] = Counter()
    missing_subject = missing_description = subject_missing_from_text = description_missing_from_text = 0
    generic_phrase_counts = Counter()

    expected_headings = {"product", "subject", "description"}
    if is_reference:
        expected_headings.add("ticket type")

    for row in rows:
        rag_text = row.get("rag_text", "") or ""
        target_lines = re.findall(r"(?im)^\s*Ticket Type\s*:\s*([^\r\n]*)", rag_text)
        ticket_type_line_count += len(target_lines)
        ticket_type_value_matches += sum(
            value.strip() == row.get("Ticket Type", "").strip() for value in target_lines
        )
        for heading in re.findall(r"(?m)^\s*([^:\r\n]{1,40})\s*:", rag_text):
            key = heading.strip().casefold()
            if key not in expected_headings:
                unexpected_headings[heading.strip()] += 1

        subject = (row.get("Ticket Subject") or "").strip()
        description = (row.get("Ticket Description") or "").strip()
        if not subject:
            missing_subject += 1
        elif normalize_text(subject) not in normalize_text(rag_text):
            subject_missing_from_text += 1
        if not description:
            missing_description += 1
        elif normalize_text(description) not in normalize_text(rag_text):
            description_missing_from_text += 1
        folded = rag_text.casefold()
        for phrase in GENERIC_PHRASES:
            if phrase in folded:
                generic_phrase_counts[phrase] += 1

    duplicate_groups = [count for count in duplicate_text_counts.values() if count > 1]
    repeated_subject_groups = [count for count in repeated_subject_counts.values() if count > 1]
    unique_subjects = len(set(value for value in subject_norms if value))
    return {
        "rows": len(rows),
        "empty_rag_text": sum(not value for value in exact_texts),
        "unique_normalized_rag_text": len(set(value for value in exact_texts if value)),
        "duplicate_normalized_rag_text_rows_beyond_first": sum(count - 1 for count in duplicate_groups),
        "duplicate_normalized_rag_text_groups": len(duplicate_groups),
        "unique_normalized_subjects": unique_subjects,
        "repeated_subject_groups": len(repeated_subject_groups),
        "repeated_subject_rows_beyond_first": sum(count - 1 for count in repeated_subject_groups),
        "missing_subject": missing_subject,
        "missing_description": missing_description,
        "subject_not_represented_in_rag_text": subject_missing_from_text,
        "description_not_represented_in_rag_text": description_missing_from_text,
        "ticket_type_lines_in_rag_text": ticket_type_line_count,
        "ticket_type_lines_matching_ground_truth": ticket_type_value_matches,
        "unexpected_metadata_headings": dict(sorted(unexpected_headings.items())),
        "generic_phrase_row_counts": dict(sorted(generic_phrase_counts.items())),
        "sample_subjects_most_repeated": [
            {"subject": subject, "count": count}
            for subject, count in repeated_subject_counts.most_common(10)
        ],
        "note": (
            "Generic phrase counts are simple phrase-presence heuristics, not a semantic quality score."
        ),
    }


def implementation_audit() -> dict:
    evaluator = EVALUATOR_SCRIPT.read_text(encoding="utf-8")
    ingestion = INGESTION_SCRIPT.read_text(encoding="utf-8")
    preprocessing = (PROJECT_ROOT / "scripts/preprocess_customer_support.py").read_text(encoding="utf-8")
    split_script = (PROJECT_ROOT / "scripts/create_rag_split.py").read_text(encoding="utf-8")
    return {
        "preprocessor_excludes_ticket_type_from_rag_text": (
            'f"Ticket Type: {normalize_whitespace(raw.get(\'Ticket Type\'))}\\n"' not in preprocessing
            and 'f"Subject: {subject}\\n"' in preprocessing
        ),
        "split_script_removes_ticket_type_only_for_evaluation_text": (
            'if split == "evaluation":' in split_script
            and 'if not line.casefold().startswith("ticket type:")' in split_script
        ),
        "top1_uses_nearest_neighbor_label": (
            'top1 = verified[0]["ticket_type"]' in evaluator
        ),
        "top5_uses_majority_vote_of_ranked_five": (
            'majority_vote([item["ticket_type"] for item in verified])' in evaluator
            and "def majority_vote" in evaluator
        ),
        "top1_and_top5_metrics_computed_separately": (
            'classification_metrics(truths, [record["top1"]' in evaluator
            and 'classification_metrics(truths, [record["top5"]' in evaluator
        ),
        "query_encoder_uses_query_text_only": (
            'row["query_text"]' in evaluator and "evaluation_query_text(row[\"rag_text\"])" in evaluator
        ),
        "query_text_removes_ticket_type_line": (
            "def evaluation_query_text" in (PROJECT_ROOT / "scripts/test_rag_retrieval.py").read_text(encoding="utf-8")
            and r"Ticket Type\s*:" in (PROJECT_ROOT / "scripts/test_rag_retrieval.py").read_text(encoding="utf-8")
        ),
        "reference_ingestion_validates_and_encodes_label_free_text": (
            "def reference_embedding_text" in ingestion
            and 'reference_embedding_text(row) for row in missing' in ingestion
            and 'contains target Ticket Type text' in ingestion
            and "MODEL_ID" in ingestion
        ),
        "ticket_type_preserved_as_database_metadata": (
            'row["Ticket Type"]' in ingestion
            and "ticket_type" in ingestion
            and "ticket_type" in (PROJECT_ROOT / "supabase/migrations/001_relay.sql").read_text(encoding="utf-8").casefold()
        ),
        "reference_resume_key_is_only_model_and_dimensions": (
            "embedding_model = %s AND embedding_dimensions = %s" in ingestion
            and 'row["rag_id"] not in reusable_ids' in ingestion
        ),
        "reference_resume_checks_text_hash": bool(
            re.search(r"rag_text_hash|embedding_input_hash|source_text_hash", ingestion, re.IGNORECASE)
        ),
        "explanation_for_equal_accuracies": (
            "Top-1 and top-5 are scored from separate prediction lists. Their accuracies can be equal"
            " when both approaches get the same number of examples correct; they can also match on"
            " every example when the nearest label is the top-5 majority."
        ),
    }


def majority_baseline(labels: list[str]) -> dict:
    counts = Counter(labels)
    if not labels:
        return {"label": None, "accuracy": None, "counts": {label: 0 for label in LABELS}}
    best_count = max(counts.values())
    # Stable label order breaks a tie between equally frequent baseline labels.
    baseline_label = next(label for label in LABELS if counts[label] == best_count)
    return {
        "label": baseline_label,
        "accuracy": best_count / len(labels),
        "counts": {label: counts.get(label, 0) for label in LABELS},
    }


def deterministic_comparison_subset(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(COMPARISON_SEED)
    chosen: list[dict[str, str]] = []
    for label in LABELS:
        pool = sorted((row for row in rows if row["Ticket Type"] == label), key=lambda row: row["Ticket ID"])
        chosen.extend(rng.sample(pool, min(COMPARISON_PER_LABEL, len(pool))))
    return sorted(chosen, key=lambda row: row["Ticket ID"])


def comparison_texts(row: dict[str, str]) -> dict[str, str]:
    subject = (row.get("Ticket Subject") or "").strip()
    description = (row.get("Ticket Description") or "").strip()
    return {
        "current_rag_text": row["query_text"],
        "subject_plus_description": f"Subject: {subject}\nDescription: {description}".strip(),
        "subject_only": f"Subject: {subject}".strip(),
    }


def retrieve_validated(connection, text: str, model) -> tuple[list[dict], int]:
    vector = Vector(encode_texts(model, [text])[0])
    raw_matches = retrieve(connection, vector)
    results = validate_matches(connection, raw_matches)
    return results, len(raw_matches)


def prediction_from_matches(matches: list[dict]) -> dict[str, str]:
    return {
        "top1": matches[0]["ticket_type"],
        "top5": majority_vote([match["ticket_type"] for match in matches]),
    }


def metric_pair(rows: list[dict], predictions: dict[str, dict[str, str]]) -> dict:
    scored_rows = [row for row in rows if row["Ticket ID"] in predictions]
    truths = [row["Ticket Type"] for row in scored_rows]
    return {
        "successful_queries": len(scored_rows),
        "top1": classification_metrics(truths, [predictions[row["Ticket ID"]]["top1"] for row in scored_rows]),
        "top5_majority": classification_metrics(truths, [predictions[row["Ticket ID"]]["top5"] for row in scored_rows]),
        "top1_predicted_label_counts": {
            label: sum(predictions[row["Ticket ID"]]["top1"] == label for row in scored_rows)
            for label in LABELS
        },
        "top5_predicted_label_counts": {
            label: sum(predictions[row["Ticket ID"]]["top5"] == label for row in scored_rows)
            for label in LABELS
        },
    }


def select_examples(rows: list[dict], current_results: dict[str, list[dict]], predictions: dict[str, dict]) -> list[dict]:
    correct = [row for row in rows if row["Ticket ID"] in predictions and predictions[row["Ticket ID"]]["top1"] == row["Ticket Type"]]
    incorrect = [row for row in rows if row["Ticket ID"] in predictions and predictions[row["Ticket ID"]]["top1"] != row["Ticket Type"]]
    rng = random.Random(COMPARISON_SEED)
    chosen = rng.sample(correct, min(EXAMPLE_COUNT // 2, len(correct)))
    chosen += rng.sample(incorrect, min(EXAMPLE_COUNT // 2, len(incorrect)))
    if len(chosen) < EXAMPLE_COUNT:
        chosen_ids = {row["Ticket ID"] for row in chosen}
        remaining = [row for row in rows if row["Ticket ID"] in predictions and row["Ticket ID"] not in chosen_ids]
        chosen += remaining[: EXAMPLE_COUNT - len(chosen)]
    chosen.sort(key=lambda row: row["Ticket ID"])

    examples = []
    for row in chosen:
        ticket_id = row["Ticket ID"]
        examples.append({
            "ticket_id": ticket_id,
            "subject": truncate(row["Ticket Subject"], EXAMPLE_SUBJECT_MAX),
            "actual_query_text_sent_to_bge": truncate(row["query_text"], EXAMPLE_QUERY_MAX),
            "true_category": row["Ticket Type"],
            "top1_prediction": predictions[ticket_id]["top1"],
            "top5_majority_prediction": predictions[ticket_id]["top5"],
            "top1_correct": predictions[ticket_id]["top1"] == row["Ticket Type"],
            "top5_correct": predictions[ticket_id]["top5"] == row["Ticket Type"],
            "top5_reference_results": [
                {
                    "subject": truncate(match["subject"] or "", EXAMPLE_SUBJECT_MAX),
                    "category": match["ticket_type"],
                    "similarity": round(match["similarity"], 6),
                }
                for match in current_results[ticket_id]
            ],
        })
    return examples


def run_diagnostics(reference_rows: list[dict], evaluation_rows: list[dict], database_url: str) -> dict:
    model = load_embedding_model()
    report: dict = {
        "status": "running",
        "model": MODEL_ID,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "read_only": True,
        "reference_csv_rows": len(reference_rows),
        "evaluation_csv_rows": len(evaluation_rows),
        "label_distributions": {
            "reference_true": {label: sum(row["Ticket Type"] == label for row in reference_rows) for label in LABELS},
            "evaluation_true": {label: sum(row["Ticket Type"] == label for row in evaluation_rows) for label in LABELS},
        },
        "embedding_text_trace": {
            "preprocessing_template": "Product: <Product Purchased>\\nTicket Type: <Ticket Type>\\nSubject: <Ticket Subject>\\nDescription: <Ticket Description>",
            "reference_embedding_source": "rag_text from rag_reference.csv, embedded by scripts/ingest_reference_tickets.py",
            "evaluation_query_source": "rag_text with any Ticket Type line removed before local encoding",
            "ticket_type_metadata": "Ticket Type remains in both CSV splits and public.tickets.ticket_type for scoring and classification.",
            "reference_target_leakage_confirmed": True,
        },
        "majority_class_baseline": majority_baseline([row["Ticket Type"] for row in evaluation_rows]),
        "comparison_seed": COMPARISON_SEED,
    }

    current_predictions: dict[str, dict[str, str]] = {}
    current_matches: dict[str, list[dict]] = {}
    current_failures: Counter[str] = Counter()
    retrieved_result_count = 0

    options = "-c default_transaction_read_only=on -c statement_timeout=30000"
    with psycopg.connect(database_url, connect_timeout=20, autocommit=True, options=options) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        definition = function_definition(connection)
        if not retrieval_function_uses_model(definition):
            raise RuntimeError("match_reference_tickets is not filtered to reference BGE vectors; apply migration 002 first")
        report["database_retrieval_function_verified"] = True

        total = len(evaluation_rows)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = evaluation_rows[start : start + EMBED_BATCH_SIZE]
            vectors = encode_texts(model, [row["query_text"] for row in batch], batch_size=EMBED_BATCH_SIZE)
            for offset, (row, vector) in enumerate(zip(batch, vectors, strict=True)):
                number = start + offset + 1
                try:
                    raw = retrieve(connection, Vector(vector))
                    retrieved_result_count += len(raw)
                    matches = validate_matches(connection, raw)
                    current_matches[row["Ticket ID"]] = matches
                    current_predictions[row["Ticket ID"]] = prediction_from_matches(matches)
                except Exception as error:
                    current_failures[type(error).__name__] += 1
                if number % PROGRESS_INTERVAL == 0 or number == total:
                    print(
                        f"Current query input: {number}/{total}; successful={len(current_predictions)}; "
                        f"failures={sum(current_failures.values())}; returned_results={retrieved_result_count}"
                    )

    full_metrics = metric_pair(evaluation_rows, current_predictions)
    report["current_rag_text_full_evaluation"] = {
        **full_metrics,
        "failed_queries": sum(current_failures.values()),
        "failure_types": dict(sorted(current_failures.items())),
        "retrieved_results_returned": retrieved_result_count,
    }
    report["label_distributions"]["top1_predicted_full"] = full_metrics["top1_predicted_label_counts"]
    report["label_distributions"]["top5_majority_predicted_full"] = full_metrics["top5_predicted_label_counts"]

    subset = deterministic_comparison_subset(evaluation_rows)
    alt_text_by_id = {row["Ticket ID"]: comparison_texts(row) for row in subset}
    variant_predictions: dict[str, dict[str, dict[str, str]]] = {
        "current_rag_text": {},
        "subject_plus_description": {},
        "subject_only": {},
    }
    for row in subset:
        ticket_id = row["Ticket ID"]
        if ticket_id in current_predictions:
            variant_predictions["current_rag_text"][ticket_id] = current_predictions[ticket_id]

    alternative_failures: Counter[str] = Counter()
    alternative_retrieved = 0
    with psycopg.connect(database_url, connect_timeout=20, autocommit=True, options=options) as connection:
        connection.execute("SET search_path TO extensions, public")
        register_vector(connection)
        if not retrieval_function_uses_model(function_definition(connection)):
            raise RuntimeError("match_reference_tickets changed or lost its reference+BGE filters during comparison")
        for row in subset:
            ticket_id = row["Ticket ID"]
            for variant in ("subject_plus_description", "subject_only"):
                try:
                    matches, returned = retrieve_validated(connection, alt_text_by_id[ticket_id][variant], model)
                    alternative_retrieved += returned
                    variant_predictions[variant][ticket_id] = prediction_from_matches(matches)
                except Exception as error:
                    alternative_failures[f"{variant}:{type(error).__name__}"] += 1

    common_ids = set.intersection(
        *(set(variant_predictions[name]) for name in variant_predictions)
    ) if variant_predictions else set()
    common_rows = [row for row in subset if row["Ticket ID"] in common_ids]
    comparison_metrics = {}
    for variant, by_id in variant_predictions.items():
        comparison_metrics[variant] = metric_pair(common_rows, {ticket_id: by_id[ticket_id] for ticket_id in common_ids})
    report["comparison"] = {
        "subset_seed": COMPARISON_SEED,
        "selection": f"up to {COMPARISON_PER_LABEL} evaluation rows per category, sampled with a fixed seed",
        "requested_subset_size": len(subset),
        "common_successful_rows_used_for_every_variant": len(common_rows),
        "variant_metrics": comparison_metrics,
        "alternative_failed_queries": sum(alternative_failures.values()),
        "alternative_failure_types": dict(sorted(alternative_failures.items())),
        "alternative_retrieved_results_returned": alternative_retrieved,
        "embedding_limitations": (
            "Subject+description and subject-only inputs were compared with the existing stored reference vectors. "
            "Reference vectors are still embedded from reference rag_text, which contains Product and the ground-truth "
            "Ticket Type line. No reference vectors were regenerated, so this is an input ablation against the current "
            "corpus representation, not a fully matched comparison of symmetric query/document templates."
        ),
    }

    report["text_audit"] = {
        "reference_embedding_text": text_audit(reference_rows, is_reference=True),
        "evaluation_rag_text": text_audit(evaluation_rows, is_reference=False),
        "evaluation_query_text_sent_to_encoder": {
            "empty": sum(not row["query_text"].strip() for row in evaluation_rows),
            "contains_ticket_type_heading": sum(bool(re.search(r"(?im)^\s*Ticket Type\s*:", row["query_text"])) for row in evaluation_rows),
            "subject_in_query_rows": sum(normalize_text(row["Ticket Subject"]) in normalize_text(row["query_text"]) for row in evaluation_rows),
            "description_in_query_rows": sum(normalize_text(row["Ticket Description"]) in normalize_text(row["query_text"]) for row in evaluation_rows),
            "note": "The ground-truth Ticket Type CSV column is not used to construct query_text; it is retained separately for scoring.",
        },
    }
    report["implementation_verification"] = implementation_audit()
    report["example_selection"] = {
        "requested_count": EXAMPLE_COUNT,
        "contains_correct_and_incorrect_top1": True,
    }
    examples = select_examples(evaluation_rows, current_matches, current_predictions)
    report["example_selection"]["actual_count"] = len(examples)
    report["example_selection"]["correct_top1_count"] = sum(item["top1_correct"] for item in examples)
    report["example_selection"]["incorrect_top1_count"] = sum(not item["top1_correct"] for item in examples)
    report["example_selection"]["contains_correct_and_incorrect_top1"] = (
        report["example_selection"]["correct_top1_count"] > 0
        and report["example_selection"]["incorrect_top1_count"] > 0
    )
    report["examples"] = examples
    report["status"] = "complete"
    return report


def make_static_report(reference_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]) -> dict:
    labels = [row["Ticket Type"] for row in evaluation_rows]
    comparison_rows = deterministic_comparison_subset(evaluation_rows)
    return {
        "status": "runtime_not_completed",
        "model": MODEL_ID,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "read_only": True,
        "reference_csv_rows": len(reference_rows),
        "evaluation_csv_rows": len(evaluation_rows),
        "label_distributions": {
            "reference_true": {label: sum(row.get("Ticket Type") == label for row in reference_rows) for label in LABELS},
            "evaluation_true": {label: sum(value == label for value in labels) for label in LABELS},
        },
        "embedding_text_trace": {
            "preprocessing_template": "Product: <Product Purchased>\\nTicket Type: <Ticket Type>\\nSubject: <Ticket Subject>\\nDescription: <Ticket Description>",
            "reference_embedding_source": "rag_text from rag_reference.csv, embedded by scripts/ingest_reference_tickets.py",
            "evaluation_query_source": "rag_text with any Ticket Type line removed before local encoding",
            "ticket_type_metadata": "Ticket Type remains in both CSV splits and public.tickets.ticket_type for scoring and classification.",
            "reference_target_leakage_confirmed": True,
        },
        "majority_class_baseline": majority_baseline(labels),
        "current_rag_text_full_evaluation": {
            "status": "unavailable_until_local_model_and_database_run",
            "successful_queries": None,
            "failed_queries": None,
            "retrieved_results_returned": None,
        },
        "comparison": {
            "status": "unavailable_until_local_model_and_database_run",
            "subset_seed": COMPARISON_SEED,
            "requested_subset_size": len(comparison_rows),
            "variant_metrics": None,
            "embedding_limitations": (
                "Alternative query strings would be compared with the existing reference vectors; "
                "reference vectors must not be regenerated during this diagnostic."
            ),
        },
        "examples": [],
        "example_selection": {
            "requested_count": EXAMPLE_COUNT,
            "actual_count": 0,
            "status": "retrieval unavailable; no nearest-neighbor examples were generated",
        },
        "text_audit": {
            "reference_embedding_text": text_audit(reference_rows, is_reference=True),
            "evaluation_rag_text": text_audit(evaluation_rows, is_reference=False),
            "evaluation_query_text_sent_to_encoder": {
                "empty": sum(not row["query_text"].strip() for row in evaluation_rows),
                "contains_ticket_type_heading": sum(bool(re.search(r"(?im)^\s*Ticket Type\s*:", row["query_text"])) for row in evaluation_rows),
                "subject_in_query_rows": sum(normalize_text(row["Ticket Subject"]) in normalize_text(row["query_text"]) for row in evaluation_rows),
                "description_in_query_rows": sum(normalize_text(row["Ticket Description"]) in normalize_text(row["query_text"]) for row in evaluation_rows),
            },
        },
        "implementation_verification": implementation_audit(),
        "runtime_error": None,
    }


def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    try:
        reference_rows = read_csv(REFERENCE_CSV)
        evaluation_rows = read_evaluation_rows()
        report = make_static_report(reference_rows, evaluation_rows)
        database_url = os.getenv("SUPABASE_DB_URL", "")
        if not database_url:
            raise RuntimeError("SUPABASE_DB_URL is unset")
        report = run_diagnostics(reference_rows, evaluation_rows, database_url)
        print(f"\nCurrent full evaluation: {report['current_rag_text_full_evaluation']['successful_queries']}/{report['evaluation_csv_rows']} successful")
        print(f"Comparison rows in common across variants: {report['comparison']['common_successful_rows_used_for_every_variant']}")
        print(f"Reference Ticket Type lines in rag_text: {report['text_audit']['reference_embedding_text']['ticket_type_lines_in_rag_text']}")
        print(f"Top-1 correct/incorrect examples: {report['example_selection']['correct_top1_count']}/{report['example_selection']['incorrect_top1_count']}")
        save_report(report)
        print(f"Diagnostic report saved: {REPORT_PATH}")
        return 0
    except Exception as error:
        try:
            if "report" not in locals():
                reference_rows = locals().get("reference_rows", [])
                evaluation_rows = locals().get("evaluation_rows", [])
                report = make_static_report(reference_rows, evaluation_rows)
            report["runtime_error"] = redact_error(error, os.getenv("SUPABASE_DB_URL", ""))
            save_report(report)
            print(f"Runtime comparison unavailable: {report['runtime_error']}")
            print(f"Static diagnostic saved: {REPORT_PATH}")
        except Exception:
            print(f"Diagnostic failed: {type(error).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
