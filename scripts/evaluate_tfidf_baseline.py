#!/usr/bin/env python3
"""Train and evaluate a leakage-safe TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = PROJECT_ROOT / "data/processed/rag_reference.csv"
EVALUATION_CSV = PROJECT_ROOT / "data/processed/rag_evaluation.csv"
BGE_REPORT = PROJECT_ROOT / "reports/rag_retrieval_diagnostic.json"
REPORT_PATH = PROJECT_ROOT / "reports/tfidf_logreg_baseline.json"
LABELS = (
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
)
TICKET_TYPE_LINE = re.compile(r"(?im)^\s*Ticket Type\s*:[^\r\n]*(?:\r?\n|$)")


def feature_text(rag_text: str) -> str:
    """Use only rag_text, removing the explicit target field if present."""
    text = TICKET_TYPE_LINE.sub("", rag_text or "").strip()
    if not text:
        raise ValueError("rag_text is empty after removing target-label fields")
    if re.search(r"(?im)^\s*Ticket Type\s*:", text):
        raise ValueError("Target Ticket Type field remains in model input")
    return text


def read_split(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Ticket ID", "Ticket Type", "rag_text", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} contains no tickets")
    for row_number, row in enumerate(rows, start=1):
        if row["split"] != expected_split:
            raise ValueError(
                f"{path.name} row {row_number} is marked {row['split']!r}; expected {expected_split!r}"
            )
        if row["Ticket Type"] not in LABELS:
            raise ValueError(f"{path.name} row {row_number} has an unexpected Ticket Type")
        # The feature is derived exclusively from rag_text. The label column is
        # only read separately as y, never concatenated into the feature.
        row["model_input"] = feature_text(row["rag_text"])
    return rows


def classification_report(truths: list[str], predictions: list[str]) -> dict[str, Any]:
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        truths, predictions, labels=list(LABELS), zero_division=0
    )
    matrix = confusion_matrix(truths, predictions, labels=list(LABELS))
    return {
        "accuracy": float(accuracy),
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
                predicted: int(matrix[row_index, column_index])
                for column_index, predicted in enumerate(LABELS)
            }
            for row_index, actual in enumerate(LABELS)
        },
    }


def fit_and_evaluate(
    reference_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit the entire pipeline on reference rows only, then score evaluation."""
    if not reference_rows or not evaluation_rows:
        raise ValueError("Both reference and evaluation rows are required")
    ref_ids = {row["Ticket ID"] for row in reference_rows}
    eval_ids = {row["Ticket ID"] for row in evaluation_rows}
    if ref_ids & eval_ids:
        raise ValueError("Ticket IDs overlap between reference and evaluation splits")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    # Sanitize again at the fit boundary so callers cannot accidentally bypass
    # the leakage guard by constructing rows manually.
    x_train = [feature_text(row["rag_text"]) for row in reference_rows]
    y_train = [row["Ticket Type"] for row in reference_rows]
    x_eval = [feature_text(row["rag_text"]) for row in evaluation_rows]
    y_eval = [row["Ticket Type"] for row in evaluation_rows]
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval).tolist()
    metrics = classification_report(y_eval, predictions)
    metrics["training_ticket_count"] = len(reference_rows)
    metrics["evaluation_ticket_count"] = len(evaluation_rows)
    metrics["reference_feature_texts_fit"] = len(x_train)
    metrics["evaluation_feature_texts_used_only_for_prediction"] = len(x_eval)
    metrics["evaluation_only_ids_used_for_fit"] = 0
    metrics["feature_source"] = "rag_text only, with any explicit Ticket Type line removed"
    return model, metrics


def bge_comparison(report_path: Path = BGE_REPORT) -> dict[str, Any]:
    """Load completed BGE metrics when present; never run or modify that evaluator."""
    if not report_path.exists():
        return {
            "status": "unavailable",
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "note": "No existing BGE diagnostic report was found.",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    full = report.get("current_rag_text_full_evaluation", {})
    top1 = full.get("top1", {})
    top5 = full.get("top5_majority", {})
    if top1.get("accuracy") is None or top5.get("accuracy") is None:
        return {
            "status": "metrics_unavailable",
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "diagnostic_status": report.get("status"),
            "top1_accuracy": top1.get("accuracy"),
            "top1_macro_f1": top1.get("macro_f1"),
            "top5_majority_accuracy": top5.get("accuracy"),
            "top5_majority_macro_f1": top5.get("macro_f1"),
            "note": (
                "The existing diagnostic report has no completed BGE evaluation metrics, "
                "so a numeric comparison cannot be made."
            ),
        }
    return {
        "status": "available",
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "top1_accuracy": top1.get("accuracy"),
        "top1_macro_f1": top1.get("macro_f1"),
        "top5_majority_accuracy": top5.get("accuracy"),
        "top5_majority_macro_f1": top5.get("macro_f1"),
    }


def make_report(
    reference_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    source_reference_leak_rows = sum(
        bool(TICKET_TYPE_LINE.search(row["rag_text"])) for row in reference_rows
    )
    source_evaluation_leak_rows = sum(
        bool(TICKET_TYPE_LINE.search(row["rag_text"])) for row in evaluation_rows
    )
    feature_leak_rows = sum(
        bool(TICKET_TYPE_LINE.search(feature_text(row["rag_text"])))
        for row in reference_rows + evaluation_rows
    )
    return {
        "model": "scikit-learn LogisticRegression",
        "vectorizer": {
            "class": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
            "fit_on": "reference split only",
        },
        "target": "Ticket Type",
        "labels": list(LABELS),
        "data": {
            "reference_count": len(reference_rows),
            "evaluation_count": len(evaluation_rows),
            "reference_split_validated": all(row["split"] == "reference" for row in reference_rows),
            "evaluation_split_validated": all(row["split"] == "evaluation" for row in evaluation_rows),
            "overlapping_ticket_ids": 0,
        },
        "leakage_checks": {
            "feature_source": "rag_text only",
            "target_column_used_only_as_y": True,
            "source_reference_rows_with_ticket_type_heading": source_reference_leak_rows,
            "source_evaluation_rows_with_ticket_type_heading": source_evaluation_leak_rows,
            "feature_rows_with_ticket_type_heading_after_sanitization": feature_leak_rows,
            "evaluation_rows_used_for_fit": 0,
            "passed": feature_leak_rows == 0 and metrics["evaluation_only_ids_used_for_fit"] == 0,
        },
        "metrics": metrics,
        "comparison_with_existing_bge": bge_comparison(),
    }


def main() -> int:
    reference_rows = read_split(REFERENCE_CSV, "reference")
    evaluation_rows = read_split(EVALUATION_CSV, "evaluation")
    model, metrics = fit_and_evaluate(reference_rows, evaluation_rows)
    report = make_report(reference_rows, evaluation_rows, metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "metrics": metrics, "bge_comparison": report["comparison_with_existing_bge"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
