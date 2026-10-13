#!/usr/bin/env python3
"""Read-only quality audit for the ticket-routing data and text baselines."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data/processed/customer_support_tickets_clean.csv"
REFERENCE_CSV = ROOT / "data/processed/rag_reference.csv"
EVALUATION_CSV = ROOT / "data/processed/rag_evaluation.csv"
BASELINE_REPORT = ROOT / "reports/tfidf_logreg_baseline.json"
BGE_DIAGNOSTIC_REPORT = ROOT / "reports/rag_retrieval_diagnostic.json"
REPORT_JSON = ROOT / "reports/ticket_data_quality_audit.json"
REPORT_MARKDOWN = ROOT / "reports/ticket_data_quality_audit.md"
LABELS = (
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
TICKET_TYPE_LINE = re.compile(r"(?im)^\s*Ticket Type\s*:[^\r\n]*(?:\r?\n|$)")
NEAR_DUPLICATE_THRESHOLD = 0.82
TRIGRAM_MAX_DOCUMENT_FREQUENCY = 0.01
SAMPLE_SEED = "ticket-data-quality-audit-v1"
SAMPLE_PER_LABEL = 4
MAX_CONFLICT_EXAMPLES = 12

# Weak, transparent lexical cues for text/category consistency checks. These
# cues are analysis only and are never features in any classifier.
CATEGORY_CUES = {
    "Billing inquiry": ("bill", "billing", "invoice", "charge", "payment", "fee", "statement"),
    "Cancellation request": ("cancel", "cancellation", "terminate", "subscription", "membership"),
    "Product inquiry": ("product", "feature", "compatib", "specification", "recommend", "availability"),
    "Refund request": ("refund", "reimburse", "money back", "return", "reversal"),
    "Technical issue": ("technical", "error", "bug", "crash", "not working", "doesn't work", "login", "install", "network", "device"),
}
GENERIC_PHRASES = (
    "i'm having an issue",
    "i am having an issue",
    "please assist",
    "please help",
    "the issue persists",
    "not working",
    "doesn't work",
    "thank you",
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")
POSTAL_RE = re.compile(r"(?i)\b((?:zip|postal\s+code)\s*(?:is|:)?\s*)\d{5}(?:-\d{4})?\b")
GREETING_NAME_RE = re.compile(r"(?i)\b(hi|hello|dear)\s+[A-Z][a-z]{1,20}\b")
NAME_PHRASE_RE = re.compile(r"(?i)\b(my name is|name is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
HANDLE_WITH_NAME_RE = re.compile(r"\b[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+\s*\(@[A-Za-z0-9_]+\)")
TIMESTAMP_NAME_RE = re.compile(r"(?i)(\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
BRACKET_TIMESTAMP_NAME_RE = re.compile(r"\[[^\]]*\d{1,2}:\d{2}[^\]]*\]\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
SOCIAL_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,}")
FAMILY_DETAIL_RE = re.compile(r"(?i)\b(?:my|our)\s+(?:daughter|son|child|mother|father|wife|husband|parent)\b[^.!?]*(?:[.!?]|$)")
PERSONAL_WELFARE_RE = re.compile(r"(?i)\bI\s+(?:do not|don't)\s+want\s+to\s+be\s+here\b[^.!?]*(?:[.!?]|$)")
PLACEHOLDER_RE = re.compile(r"\{[^{}]{1,120}\}")


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def safe_feature_text(rag_text: str | None) -> str:
    """Use rag_text only, excluding any explicit ground-truth field line."""
    value = TICKET_TYPE_LINE.sub("", rag_text or "").strip()
    if re.search(r"(?im)^\s*Ticket Type\s*:", value):
        raise ValueError("Ticket Type target field remains in rag_text")
    return redact_sensitive_text(value)


def redact_sensitive_text(value: str | None) -> str:
    """Remove common direct identifiers embedded in free-text fields."""
    text = " ".join((value or "").split())
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = URL_RE.sub("[REDACTED_URL]", text)
    text = POSTAL_RE.sub(r"\1[REDACTED_POSTAL_CODE]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = NAME_PHRASE_RE.sub(r"\1 [REDACTED_NAME]", text)
    text = GREETING_NAME_RE.sub(r"\1 [REDACTED_NAME]", text)
    return re.sub(r"\b\d{6,}\b", "[REDACTED_NUMBER]", text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def read_split(path: Path, expected_split: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {"Ticket ID", "Ticket Type", "Ticket Subject", "Ticket Description", "rag_text", "split"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row["split"] != expected_split:
            raise ValueError(f"{path.name} row {index} has split={row['split']!r}")
        if row["Ticket Type"] not in LABELS:
            raise ValueError(f"{path.name} row {index} has unexpected label")
        if row["Ticket ID"] in seen_ids:
            raise ValueError(f"{path.name} contains duplicate Ticket ID values")
        seen_ids.add(row["Ticket ID"])
    return rows


def normalized_document(row: dict[str, str]) -> str:
    return normalize_text(row.get("rag_text"))


def _trigrams(value: str) -> set[tuple[str, str, str]]:
    words = TOKEN_RE.findall(value)
    return set(zip(words, words[1:], words[2:]))


def _category_pairs(labels: set[str]) -> list[str]:
    return [f"{left} | {right}" for left, right in itertools.combinations(sorted(labels), 2)]


def _safe_excerpt(value: str | None, limit: int = 360) -> str:
    text = " ".join((value or "").split())
    text = HANDLE_WITH_NAME_RE.sub("[REDACTED_NAME_AND_HANDLE]", text)
    text = BRACKET_TIMESTAMP_NAME_RE.sub("[REDACTED_TIMESTAMP_AND_NAME]", text)
    text = FAMILY_DETAIL_RE.sub("[REDACTED_PERSONAL_DETAIL]", text)
    text = PERSONAL_WELFARE_RE.sub("[REDACTED_PERSONAL_DETAIL]", text)
    text = redact_sensitive_text(text)
    text = TIMESTAMP_NAME_RE.sub(r"\1[REDACTED_NAME]", text)
    text = SOCIAL_HANDLE_RE.sub("[REDACTED_HANDLE]", text)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def similarity_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Find exact and approximate text groups using trigram Jaccard.

    Approximate candidate pairs share a word trigram present in at most 1% of
    the combined corpus; pairs are retained at Jaccard >= 0.82. Exact matches
    are grouped independently. The method is deterministic and API-free.
    """
    normalized = [normalized_document(row) for row in rows]
    count = len(rows)
    exact_buckets: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(normalized):
        if text:
            exact_buckets[text].append(index)

    exact_groups = [indices for indices in exact_buckets.values() if len(indices) > 1]
    exact_conflict_groups = [
        indices for indices in exact_groups
        if len({rows[index]["Ticket Type"] for index in indices}) > 1
    ]
    exact_conflict_pairs = sum(
        sum(1 for left, right in itertools.combinations(indices, 2)
            if rows[left]["Ticket Type"] != rows[right]["Ticket Type"])
        for indices in exact_conflict_groups
    )

    shingle_sets = [_trigrams(text) for text in normalized]
    postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, grams in enumerate(shingle_sets):
        for gram in grams:
            postings[gram].append(index)
    max_df = max(2, int(count * TRIGRAM_MAX_DOCUMENT_FREQUENCY))
    candidates: set[tuple[int, int]] = set()
    for indexes in postings.values():
        if len(indexes) > max_df:
            continue
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1:]:
                if normalized[left] != normalized[right]:
                    candidates.add((left, right))

    union = UnionFind(count)
    for indices in exact_groups:
        for other in indices[1:]:
            union.union(indices[0], other)

    near_edges: list[tuple[int, int, float]] = []
    for left, right in sorted(candidates):
        a, b = shingle_sets[left], shingle_sets[right]
        if not a or not b:
            continue
        score = len(a & b) / len(a | b)
        if score >= NEAR_DUPLICATE_THRESHOLD:
            near_edges.append((left, right, score))
            union.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        if normalized[index]:
            components[union.find(index)].append(index)
    near_components = [
        members for members in components.values()
        if len(members) > 1 and len({normalized[index] for index in members}) > 1
    ]
    near_component_sizes = [len(members) for members in near_components]
    near_conflict_components = [
        members for members in near_components
        if len({rows[index]["Ticket Type"] for index in members}) > 1
    ]
    cross_label_edges = [
        (left, right, score) for left, right, score in near_edges
        if rows[left]["Ticket Type"] != rows[right]["Ticket Type"]
    ]
    pair_counts: Counter[str] = Counter()
    for left, right, _score in cross_label_edges:
        left_label, right_label = sorted((rows[left]["Ticket Type"], rows[right]["Ticket Type"]))
        pair_counts[f"{left_label} | {right_label}"] += 1

    examples = []
    seen_pairs: set[tuple[int, int]] = set()
    exact_candidates = []
    for members in exact_conflict_groups:
        for left, right in itertools.combinations(members, 2):
            if rows[left]["Ticket Type"] != rows[right]["Ticket Type"]:
                exact_candidates.append((left, right, 1.0))
    for left, right, score in sorted(exact_candidates + cross_label_edges, key=lambda item: (-item[2], item[0], item[1])):
        key = (min(left, right), max(left, right))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        examples.append({
            "match_type": "exact" if normalized[left] == normalized[right] else "approximate",
            "similarity": round(score, 4),
            "categories": [rows[left]["Ticket Type"], rows[right]["Ticket Type"]],
            "subject_a": _safe_excerpt(rows[left].get("Ticket Subject"), 160),
            "subject_b": _safe_excerpt(rows[right].get("Ticket Subject"), 160),
            "text_a": _safe_excerpt(rows[left].get("rag_text")),
            "text_b": _safe_excerpt(rows[right].get("rag_text")),
        })
        if len(examples) >= MAX_CONFLICT_EXAMPLES:
            break

    split_crossing = sum(
        len({rows[index]["split"] for index in members}) > 1
        for members in components.values() if len(members) > 1
    )
    exact_per_split = {}
    near_per_split = {}
    for split in ("reference", "evaluation"):
        indices = {index for index, row in enumerate(rows) if row["split"] == split}
        exact_per_split[split] = sum(
            len([index for index in members if index in indices]) > 1 for members in exact_buckets.values()
        )
        near_per_split[split] = sum(
            left in indices and right in indices for left, right, _score in near_edges
        )

    return {
        "method": {
            "exact_normalization": "casefold and collapse whitespace",
            "approximate_similarity": "Jaccard over unique word trigrams",
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "candidate_blocking": f"shared word trigram document frequency <= {max_df} rows ({TRIGRAM_MAX_DOCUMENT_FREQUENCY:.0%} of corpus)",
            "component_linking": "connected components across exact groups and qualifying approximate pairs",
        },
        "exact": {
            "duplicate_groups": len(exact_groups),
            "rows_in_duplicate_groups": sum(map(len, exact_groups)),
            "duplicate_rows_beyond_first": sum(len(group) - 1 for group in exact_groups),
            "cross_category_conflict_groups": len(exact_conflict_groups),
            "cross_category_conflicting_pairs": exact_conflict_pairs,
            "category_pair_counts": _exact_category_pair_counts(rows, exact_conflict_groups),
        },
        "approximate": {
            "qualifying_distinct_text_pairs": len(near_edges),
            "approximate_connected_groups": len(near_components),
            "rows_in_approximate_groups": sum(near_component_sizes),
            "excess_rows_beyond_first_per_group": sum(size - 1 for size in near_component_sizes),
            "largest_group_size": max(near_component_sizes, default=0),
            "cross_category_conflict_groups": len(near_conflict_components),
            "cross_category_conflicting_pairs": len(cross_label_edges),
            "category_pair_counts": dict(sorted(pair_counts.items())),
        },
        "groups_crossing_splits": split_crossing,
        "exact_duplicate_groups_by_split": exact_per_split,
        "approximate_pairs_by_split": near_per_split,
        "conflict_examples": examples,
        "interpretation": "A conflict is a review candidate, not proof that either Ticket Type is incorrect.",
    }


def _exact_category_pair_counts(rows: list[dict[str, str]], groups: list[list[int]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for members in groups:
        for left, right in itertools.combinations(members, 2):
            left_label, right_label = rows[left]["Ticket Type"], rows[right]["Ticket Type"]
            if left_label != right_label:
                a, b = sorted((left_label, right_label))
                counts[f"{a} | {b}"] += 1
    return dict(sorted(counts.items()))


def _length_summary(values: list[str]) -> dict[str, Any]:
    lengths = sorted(len(" ".join((value or "").split())) for value in values)
    if not lengths:
        return {"count": 0, "empty": 0, "min_chars": None, "median_chars": None, "mean_chars": None, "max_chars": None}
    middle = len(lengths) // 2
    median = lengths[middle] if len(lengths) % 2 else (lengths[middle - 1] + lengths[middle]) / 2
    return {
        "count": len(lengths),
        "empty": sum(length == 0 for length in lengths),
        "min_chars": lengths[0],
        "median_chars": median,
        "mean_chars": round(sum(lengths) / len(lengths), 2),
        "max_chars": lengths[-1],
        "very_short_under_40_chars": sum(length < 40 for length in lengths),
    }


def _duplicate_frequency(values: list[str]) -> dict[str, Any]:
    counts = Counter(normalize_text(value) for value in values if normalize_text(value))
    repeated = {text: count for text, count in counts.items() if count > 1}
    return {
        "unique_nonempty_values": len(counts),
        "duplicate_groups": len(repeated),
        "rows_in_duplicate_groups": sum(repeated.values()),
        "duplicate_rows_beyond_first": sum(count - 1 for count in repeated.values()),
        "top_repeated_values": [
            {"text": _safe_excerpt(text, 180), "count": count}
            for text, count in sorted(repeated.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
    }


def _text_quality(rows: list[dict[str, str]], split_name: str) -> dict[str, Any]:
    rag_values = [row.get("rag_text", "") for row in rows]
    subject_values = [row.get("Ticket Subject", "") for row in rows]
    description_values = [row.get("Ticket Description", "") for row in rows]
    rag_word_counts = [len(TOKEN_RE.findall(safe_feature_text(value))) for value in rag_values]
    low_info_limit = 8
    category_cue_coverage: dict[str, Any] = {}
    for label, cues in CATEGORY_CUES.items():
        class_rows = [row for row in rows if row.get("Ticket Type") == label]
        matched = [
            row for row in class_rows
            if any(cue in (row.get("Ticket Subject", "") + " " + row.get("Ticket Description", "")).casefold() for cue in cues)
        ]
        other_cue_rows = [
            row for row in class_rows
            if any(
                cue in (row.get("Ticket Subject", "") + " " + row.get("Ticket Description", "")).casefold()
                for other_label, other_cues in CATEGORY_CUES.items() if other_label != label
                for cue in other_cues
            )
        ]
        category_cue_coverage[label] = {
            "class_row_count": len(class_rows),
            "assigned_class_rows_with_at_least_one_weak_lexical_cue": len(matched),
            "assigned_class_cue_coverage": round(len(matched) / len(class_rows), 4) if class_rows else None,
            "assigned_class_rows_with_another_category_cue": len(other_cue_rows),
            "other_category_cue_rate": round(len(other_cue_rows) / len(class_rows), 4) if class_rows else None,
            "cues": list(cues),
        }
    generic_counts = {
        phrase: sum(phrase in value.casefold() for value in description_values)
        for phrase in GENERIC_PHRASES
    }
    placeholder_counts: Counter[str] = Counter()
    placeholder_rows = 0
    for value in rag_values:
        matches = PLACEHOLDER_RE.findall(value or "")
        placeholder_rows += bool(matches)
        placeholder_counts.update(matches)
    safe_placeholders = [
        {"pattern": _safe_excerpt(pattern, 100), "count": count}
        for pattern, count in sorted(placeholder_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    return {
        "row_count": len(rows),
        "labels": {label: sum(row.get("Ticket Type") == label for row in rows) for label in LABELS},
        "rag_text_length": _length_summary(rag_values),
        "subject_length": _length_summary(subject_values),
        "description_length": _length_summary(description_values),
        "rag_text_low_information_rows": {
            "definition": f"fewer than {low_info_limit + 1} word tokens after target-line removal",
            "count": sum(count <= low_info_limit for count in rag_word_counts),
        },
        "rag_text_exact_duplicate_frequency": _duplicate_frequency(rag_values),
        "subject_repetition": _duplicate_frequency(subject_values),
        "description_repetition": _duplicate_frequency(description_values),
        "generic_description_phrase_row_counts": generic_counts,
        "rows_with_brace_style_placeholders": int(placeholder_rows),
        "brace_style_placeholder_occurrences": sum(placeholder_counts.values()),
        "unique_brace_style_placeholder_patterns": len(placeholder_counts),
        "most_common_brace_style_placeholders": safe_placeholders,
        "rows_with_any_generic_phrase": sum(
            any(phrase in value.casefold() for phrase in GENERIC_PHRASES)
            for value in description_values
        ),
        "category_cue_coverage_heuristic": category_cue_coverage,
        "note": "Category cue coverage is a lexical heuristic, not a semantic label-quality judgment.",
    }


def _metric_report(truths: list[str], predictions: list[str]) -> dict[str, Any]:
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
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            actual: {predicted: int(matrix[i, j]) for j, predicted in enumerate(LABELS)}
            for i, actual in enumerate(LABELS)
        },
    }


def text_variants(row: dict[str, str]) -> dict[str, str]:
    """Return three feature views, never reading Ticket Type or IDs."""
    return {
        "subject_only": redact_sensitive_text(row.get("Ticket Subject")),
        "description_only": redact_sensitive_text(row.get("Ticket Description")),
        "rag_text": safe_feature_text(row.get("rag_text")),
    }


def evaluate_text_variants(
    reference_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    ref_ids = {row["Ticket ID"] for row in reference_rows}
    eval_ids = {row["Ticket ID"] for row in evaluation_rows}
    if ref_ids & eval_ids:
        raise ValueError("Ticket IDs overlap across reference and evaluation")
    if any(row.get("split") != "reference" for row in reference_rows):
        raise ValueError("Training rows must all belong to the reference split")
    if any(row.get("split") != "evaluation" for row in evaluation_rows):
        raise ValueError("Scoring rows must all belong to the evaluation split")

    variants = ("subject_only", "description_only", "rag_text")
    results: dict[str, Any] = {}
    fitted_vocabularies: dict[str, set[str]] = {}
    truths = [row["Ticket Type"] for row in evaluation_rows]
    for variant in variants:
        x_train = [text_variants(row)[variant] for row in reference_rows]
        x_eval = [text_variants(row)[variant] for row in evaluation_rows]
        y_train = [row["Ticket Type"] for row in reference_rows]
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_eval).tolist()
        results[variant] = {
            **_metric_report(truths, predictions),
            "training_rows": len(x_train),
            "evaluation_rows_scored": len(x_eval),
            "vectorizer_fit_rows": len(x_train),
            "evaluation_rows_used_for_fit": 0,
        }
        fitted_vocabularies[variant] = set(pipeline.named_steps["tfidf"].vocabulary_)
    return results, fitted_vocabularies


def _balanced_sample(evaluation_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for label in LABELS:
        candidates = [row for row in evaluation_rows if row["Ticket Type"] == label]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{SAMPLE_SEED}:{row['Ticket ID']}".encode("utf-8")
        ).hexdigest())
        for row in candidates[:SAMPLE_PER_LABEL]:
            selected.append({
                "category_for_review": label,
                "subject": _safe_excerpt(row.get("Ticket Subject"), 150),
                "description_excerpt": _safe_excerpt(row.get("Ticket Description"), 240),
            })
    return selected


def _provided_baseline_context() -> dict[str, Any]:
    return {
        "bge_nearest_neighbor": {
            "accuracy": 0.1903,
            "macro_f1": 0.1900,
            "source": "user-provided project context; repository diagnostic report is incomplete",
        },
        "tfidf_logistic_regression": {
            "accuracy": 0.1921,
            "macro_f1": 0.1911,
            "source": "existing reports/tfidf_logreg_baseline.json",
        },
    }


def build_report(
    clean_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if {row["Ticket ID"] for row in reference_rows} & {row["Ticket ID"] for row in evaluation_rows}:
        raise ValueError("Reference and evaluation Ticket IDs overlap")
    all_rows = reference_rows + evaluation_rows
    label_counts_all = Counter(row.get("Ticket Type", "") for row in clean_rows)
    similarities = similarity_audit(all_rows)
    experiments, _vocabularies = evaluate_text_variants(reference_rows, evaluation_rows)
    rag_confusion = experiments["rag_text"]["confusion_matrix"]
    category_difficulty = {}
    for label in LABELS:
        alternatives = sorted(
            ((other, count) for other, count in rag_confusion[label].items() if other != label),
            key=lambda item: (-item[1], item[0]),
        )
        category_difficulty[label] = {
            "recall": experiments["rag_text"]["per_class"][label]["recall"],
            "f1": experiments["rag_text"]["per_class"][label]["f1"],
            "most_common_confusions": [
                {"predicted_as": other, "count": count} for other, count in alternatives[:2]
            ],
        }
    per_split_quality = {
        "reference": _text_quality(reference_rows, "reference"),
        "evaluation": _text_quality(evaluation_rows, "evaluation"),
        "cleaned_all": _text_quality(clean_rows, "cleaned_all"),
    }
    report = {
        "title": "Ticket Routing Data Quality Audit",
        "read_only": True,
        "methodology": {
            "training_boundary": "Every vectorizer/classifier is fit on rag_reference.csv only; evaluation is prediction/scoring only.",
            "classifier": "TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True) + LogisticRegression(max_iter=1000, random_state=0).",
            "features": "Subject, Ticket Description, and rag_text are evaluated separately; rag_text strips any explicit Ticket Type line. No target, customer ID, or other metadata is used as input.",
            "near_duplicate_method": similarities["method"],
            "low_information_definition": "rag_text with <= 8 word tokens after target-line removal; short text is <40 characters.",
            "balanced_sample": f"Deterministic SHA-256 ordering, {SAMPLE_PER_LABEL} evaluation examples per category, seed {SAMPLE_SEED}.",
            "privacy": "Report examples omit ticket/customer IDs and redact emails, URLs, postal codes, phone-like sequences, long numbers, and greeting/name phrases.",
        },
        "split_counts": {
            "cleaned_total": len(clean_rows),
            "reference_total": len(reference_rows),
            "evaluation_total": len(evaluation_rows),
            "cleaned_label_distribution": {label: label_counts_all[label] for label in LABELS},
            "reference_label_distribution": per_split_quality["reference"]["labels"],
            "evaluation_label_distribution": per_split_quality["evaluation"]["labels"],
            "ticket_id_overlap": 0,
        },
        "text_quality": per_split_quality,
        "duplicate_and_conflict_analysis": similarities,
        "field_experiments": {
            "same_reference_and_evaluation_rows": True,
            "evaluation_count": len(evaluation_rows),
            "vectorizer_fit_on_evaluation": False,
            "results": experiments,
            "interpretation_caution": "These are diagnostic experiments on the project's current evaluation split, not final model-selection estimates on a new production test set.",
        },
        "category_difficulty_from_rag_text": category_difficulty,
        "balanced_evaluation_examples": _balanced_sample(evaluation_rows),
        "automated_findings": [],
        "hypotheses_requiring_human_review": [],
        "baseline_context": _provided_baseline_context(),
        "source_report_availability": {
            "existing_bge_diagnostic_status": _report_status(BGE_DIAGNOSTIC_REPORT),
            "existing_tfidf_report_status": _report_status(BASELINE_REPORT),
        },
    }
    report["automated_findings"] = _automated_findings(report)
    report["hypotheses_requiring_human_review"] = _review_hypotheses(report)
    return report


def _report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "status": data.get("status", "available"),
        "path": str(path.relative_to(ROOT)),
    }


def _automated_findings(report: dict[str, Any]) -> list[str]:
    quality = report["text_quality"]
    eval_quality = quality["evaluation"]
    variants = report["field_experiments"]["results"]
    conflicts = report["duplicate_and_conflict_analysis"]
    subjects = eval_quality["subject_repetition"]
    descriptions = eval_quality["description_repetition"]
    return [
        f"Evaluation labels contain {len(LABELS)} categories; counts are {report['split_counts']['evaluation_label_distribution']}.",
        f"Evaluation repeats {subjects['unique_nonempty_values']} normalized subjects across {eval_quality['row_count']} rows; {subjects['duplicate_rows_beyond_first']} rows are beyond each subject's first occurrence.",
        f"Exact rag_text conflicts across categories: {conflicts['exact']['cross_category_conflict_groups']} groups / {conflicts['exact']['cross_category_conflicting_pairs']} conflicting pairs.",
        f"Approximate cross-category near-duplicate conflicts at Jaccard >= {NEAR_DUPLICATE_THRESHOLD}: {conflicts['approximate']['cross_category_conflict_groups']} connected groups / {conflicts['approximate']['cross_category_conflicting_pairs']} qualifying pairs.",
        f"Description repetition: {descriptions['duplicate_groups']} exact normalized groups and {descriptions['duplicate_rows_beyond_first']} excess repeated rows in evaluation.",
        "Classification metrics by input view: " + "; ".join(
            f"{name} accuracy={values['accuracy']:.4f}, macro-F1={values['macro_f1']:.4f}"
            for name, values in variants.items()
        ),
        "Most common rag_text confusions: " + "; ".join(
            f"{label} as {values['most_common_confusions'][0]['predicted_as']} ({values['most_common_confusions'][0]['count']})"
            for label, values in report["category_difficulty_from_rag_text"].items()
        ),
        "Weak assigned-category cue coverage: " + "; ".join(
            f"{label} {values['assigned_class_cue_coverage']:.1%}"
            for label, values in quality["evaluation"]["category_cue_coverage_heuristic"].items()
        ),
    ]


def _review_hypotheses(report: dict[str, Any]) -> list[str]:
    results = report["field_experiments"]["results"]
    best = max(results, key=lambda name: results[name]["macro_f1"])
    subject = results["subject_only"]["macro_f1"]
    description = results["description_only"]["macro_f1"]
    rag = results["rag_text"]["macro_f1"]
    hypotheses = [
        "Repeated generic issue wording and repeated subject templates may leave too little category-specific information; manually inspect the balanced examples before concluding labels are wrong.",
        f"The {best} view has the highest measured macro-F1 in this diagnostic, but differences are descriptive and may reflect incidental wording rather than a robust advantage.",
        "The lexical category-cue coverage and similarity conflicts are screening heuristics; annotators should review source context and the intended routing policy before relabeling anything.",
    ]
    if max(subject, description, rag) - min(subject, description, rag) < 0.02:
        hypotheses.append("The small gap among text views may indicate that all three contain similarly weak category signals, but this needs confirmation on a fresh evaluation set.")
    return hypotheses


def markdown_summary(report: dict[str, Any]) -> str:
    experiments = report["field_experiments"]["results"]
    conflicts = report["duplicate_and_conflict_analysis"]
    quality = report["text_quality"]["evaluation"]
    lines = [
        "# Ticket Data Quality Audit",
        "",
        f"Read-only audit of {report['split_counts']['reference_total']:,} reference and {report['split_counts']['evaluation_total']:,} evaluation tickets.",
        "",
        "## Field comparison",
        "",
        "| Input | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]
    for name, metrics in experiments.items():
        lines.append(f"| {name.replace('_', ' ')} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} |")
    lines += [
        "",
        "All three experiments fit TF-IDF and Logistic Regression on the reference split only and score the same evaluation rows.",
        "",
        "## Category distinction",
        "",
    ]
    for label, values in report["category_difficulty_from_rag_text"].items():
        confusions = ", ".join(
            f"{item['predicted_as']} ({item['count']})"
            for item in values["most_common_confusions"]
        )
        lines.append(f"- **{label}** — recall {values['recall']:.3f}, F1 {values['f1']:.3f}; common confusions: {confusions}.")
    lines += [
        "",
        "The assigned-category cue rates in the JSON are weak lexical heuristics; low coverage or competing cues are review indicators, not relabeling decisions.",
        "",
        "## Duplicate conflicts",
        "",
        f"- Exact cross-category conflicts: {conflicts['exact']['cross_category_conflict_groups']} groups and {conflicts['exact']['cross_category_conflicting_pairs']} row pairs.",
        f"- Approximate cross-category conflicts: {conflicts['approximate']['cross_category_conflict_groups']} connected groups and {conflicts['approximate']['cross_category_conflicting_pairs']} qualifying pairs (word-trigram Jaccard >= {NEAR_DUPLICATE_THRESHOLD:.2f}).",
        "- A conflict is a review candidate, not proof of a mislabeled ticket.",
        "",
        "## Repeated text and quality",
        "",
        f"Evaluation has {quality['subject_repetition']['unique_nonempty_values']} normalized subject values across {quality['row_count']} rows; {quality['rows_with_any_generic_phrase']} descriptions contain one of the tracked generic phrases.",
        f"RAG text under 40 characters: {quality['rag_text_length']['very_short_under_40_chars']}; low-information (at most 8 tokens): {quality['rag_text_low_information_rows']['count']}.",
        f"Brace-style placeholders occur in {quality['rows_with_brace_style_placeholders']} evaluation descriptions ({quality['brace_style_placeholder_occurrences']} occurrences; {quality['unique_brace_style_placeholder_patterns']} patterns).",
        "Assigned-category cue rates are weak lexical heuristics; low coverage or competing cues are review indicators, not automated relabeling decisions.",
        "",
        "## Balanced examples for human review",
        "",
    ]
    for item in report["balanced_evaluation_examples"]:
        lines.extend([
            f"- **{item['category_for_review']}** — Subject: {item['subject']}",
            f"  Description: {item['description_excerpt']}",
        ])
    lines += ["", "Automated similarity and lexical-cue findings do not establish label correctness. See the JSON report for full metrics, per-class scores, confusion matrices, method details, and hypotheses.", ""]
    return "\n".join(lines)


def main() -> int:
    clean_rows = read_csv(CLEAN_CSV)
    reference_rows = read_split(REFERENCE_CSV, "reference")
    evaluation_rows = read_split(EVALUATION_CSV, "evaluation")
    clean_ids = {row.get("Ticket ID") for row in clean_rows}
    split_ids = {row["Ticket ID"] for row in reference_rows + evaluation_rows}
    if clean_ids != split_ids:
        raise ValueError("Cleaned rows do not match the combined reference/evaluation Ticket IDs")
    report = build_report(clean_rows, reference_rows, evaluation_rows)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MARKDOWN.write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps({
        "report_json": str(REPORT_JSON),
        "report_markdown": str(REPORT_MARKDOWN),
        "split_counts": report["split_counts"],
        "exact_cross_category_conflicts": report["duplicate_and_conflict_analysis"]["exact"],
        "approximate_cross_category_conflicts": report["duplicate_and_conflict_analysis"]["approximate"],
        "field_experiment_metrics": {
            name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]}
            for name, value in report["field_experiments"]["results"].items()
        },
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
