#!/usr/bin/env python3
"""Build a deterministic, read-only human-review queue from existing ticket data.

This script only selects and redacts review candidates. It never fits a model,
changes labels/splits, calls Supabase, or writes to source datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_ticket_data_quality import (
    CATEGORY_CUES,
    GENERIC_PHRASES,
    LABELS,
    NEAR_DUPLICATE_THRESHOLD,
    TRIGRAM_MAX_DOCUMENT_FREQUENCY,
    TOKEN_RE,
    _safe_excerpt,
    normalize_text,
    read_split,
)


REFERENCE_CSV = ROOT / "data/processed/rag_reference.csv"
EVALUATION_CSV = ROOT / "data/processed/rag_evaluation.csv"
BASELINE_REPORT = ROOT / "reports/tfidf_logreg_baseline.json"
REPORT_CSV = ROOT / "reports/ticket_review_queue.csv"
REPORT_MARKDOWN = ROOT / "reports/ticket_review_queue.md"
PLACEHOLDER_RE = re.compile(r"\{[^{}]{1,120}\}")
SEED = "ticket-review-queue-v1"
CSV_FIELDS = (
    "ticket_id",
    "rag_id",
    "split",
    "ground_truth_category_for_human_review_only",
    "subject_redacted",
    "description_redacted",
    "review_reason_codes",
    "evidence",
    "similar_ticket_ids_and_categories",
    "automated_flags_are_not_proof_of_error",
    "human_review_decision",
    "reviewer_notes",
)


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    tokens = TOKEN_RE.findall(text)
    return set(zip(tokens, tokens[1:], tokens[2:]))


def _stable_key(row: dict[str, str], group: str = "") -> str:
    return hashlib.sha256(f"{SEED}|{group}|{row.get('Ticket ID', '')}".encode()).hexdigest()


def _validate_rows(reference: list[dict[str, str]], evaluation: list[dict[str, str]]) -> None:
    if any(row.get("split") != "reference" for row in reference):
        raise ValueError("reference input contains a non-reference row")
    if any(row.get("split") != "evaluation" for row in evaluation):
        raise ValueError("evaluation input contains a non-evaluation row")
    ids = [row.get("Ticket ID", "") for row in reference + evaluation]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate Ticket ID across inputs")
    if any(row.get("Ticket Type") not in LABELS for row in reference + evaluation):
        raise ValueError("unexpected Ticket Type label")


def find_text_matches(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Match pairs using the audit's rare word-trigram Jaccard method.

    Exact normalized matches are reported separately from approximate matches.
    The blocking threshold is max(2, 1% of corpus rows), matching the audit.
    """
    docs = [normalize_text(row.get("rag_text")) for row in rows]
    buckets: dict[str, list[int]] = defaultdict(list)
    sets = [_trigrams(doc) for doc in docs]
    postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        if doc:
            buckets[doc].append(i)
        for gram in sets[i]:
            postings[gram].append(i)
    matches: dict[tuple[int, int], tuple[str, float]] = {}
    for members in buckets.values():
        if len(members) > 1:
            for pos, left in enumerate(members):
                for right in members[pos + 1:]:
                    matches[(min(left, right), max(left, right))] = ("exact", 1.0)
    max_df = max(2, int(len(rows) * TRIGRAM_MAX_DOCUMENT_FREQUENCY))
    candidates: set[tuple[int, int]] = set()
    for indexes in postings.values():
        if len(indexes) <= max_df:
            for pos, left in enumerate(indexes):
                for right in indexes[pos + 1:]:
                    if docs[left] != docs[right]:
                        candidates.add((min(left, right), max(left, right)))
    for left, right in sorted(candidates):
        union = sets[left] | sets[right]
        score = len(sets[left] & sets[right]) / len(union) if union else 0.0
        if score >= NEAR_DUPLICATE_THRESHOLD:
            matches[(left, right)] = ("approximate", score)
    return [
        {
            "left_index": left,
            "right_index": right,
            "match_type": kind,
            "similarity": round(score, 4),
            "cross_category": rows[left].get("Ticket Type") != rows[right].get("Ticket Type"),
        }
        for (left, right), (kind, score) in sorted(matches.items())
    ]


def _add_reason(
    selected: dict[str, dict[str, Any]], row: dict[str, str], code: str, evidence: str,
    neighbors: list[str] | None = None,
) -> None:
    ticket_id = row["Ticket ID"]
    item = selected.setdefault(ticket_id, {"row": row, "reasons": {}, "neighbors": set()})
    item["reasons"][code] = evidence
    item["neighbors"].update(neighbors or [])


def _choose(rows: list[dict[str, str]], limit: int, group: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (_stable_key(row, group), row.get("Ticket ID", "")))[:limit]


def build_review_queue(
    reference: list[dict[str, str]],
    evaluation: list[dict[str, str]],
    *,
    limit_per_group: int = 30,
    balanced_per_category: int = 8,
    baseline: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return deduplicated queue rows and a deterministic, PII-safe summary."""
    _validate_rows(reference, evaluation)
    if limit_per_group < 1 or balanced_per_category < 1:
        raise ValueError("selection limits must be positive")
    selected: dict[str, dict[str, Any]] = {}
    reason_candidates: dict[str, list[tuple[dict[str, str], str, list[str]]]] = defaultdict(list)

    # Evaluation labels are read solely to choose diagnostic review examples;
    # they are never used to fit a model or generate text features.
    for row in evaluation:
        description = row.get("Ticket Description") or ""
        placeholders = sorted(set(PLACEHOLDER_RE.findall(description)))
        if placeholders:
            patterns = ", ".join(_safe_excerpt(value, 100) for value in placeholders[:8])
            reason_candidates["brace_placeholder"].append((
                row, f"Description contains brace-style placeholder(s): {patterns}.", []
            ))
        text = normalize_text(" ".join((row.get("Ticket Subject") or "", description)))
        generic = [phrase for phrase in GENERIC_PHRASES if phrase in text]
        cues = [cue for cue in CATEGORY_CUES[row["Ticket Type"]] if cue in text]
        if generic:
            reason_candidates["generic_description"].append((
                row, "Tracked generic wording present: " + ", ".join(generic[:5]) + ".", []
            ))
        if not cues:
            reason_candidates["low_category_cue"].append((
                row, f"No configured weak lexical cue for assigned category {row['Ticket Type']!r} was found in subject/description.", []
            ))

    all_rows = reference + evaluation
    matches = find_text_matches(all_rows)
    exact_cross = 0
    approximate_cross = 0
    cross_pair_summary: Counter[str] = Counter()
    cross_matches_by_split: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for match in matches:
        left, right = all_rows[match["left_index"]], all_rows[match["right_index"]]
        if match["cross_category"]:
            exact_cross += match["match_type"] == "exact"
            approximate_cross += match["match_type"] == "approximate"
            labels = " | ".join(sorted((left["Ticket Type"], right["Ticket Type"])))
            cross_pair_summary[labels] += 1
            if left["split"] == right["split"]:
                cross_matches_by_split[left["split"]].append((match, labels))
    for split, edges in cross_matches_by_split.items():
        for match, labels in sorted(edges, key=lambda x: (x[1], x[0]["left_index"], x[0]["right_index"]))[:limit_per_group]:
            left, right = all_rows[match["left_index"]], all_rows[match["right_index"]]
            code = "exact_cross_category_match" if match["match_type"] == "exact" else "near_cross_category_match"
            detail = f"{match['match_type']} {labels} text match; word-trigram Jaccard={match['similarity']:.4f}. Similarity is a screening signal only."
            reason_candidates[code].append((left, detail, [f"{right['Ticket ID']} ({right['Ticket Type']})"]))
            reason_candidates[code].append((right, detail, [f"{left['Ticket ID']} ({left['Ticket Type']})"]))

    # Category-pair examples are selected using only saved aggregate confusion
    # cells. No ticket-level model predictions are available in the report.
    confusion = ((baseline or {}).get("metrics") or {}).get("confusion_matrix") or {}
    pair_specs = (("Billing inquiry", "Refund request"), ("Product inquiry", "Technical issue"))
    for left_label, right_label in pair_specs:
        cells = [
            (confusion.get(left_label, {}).get(right_label), left_label, right_label),
            (confusion.get(right_label, {}).get(left_label), right_label, left_label),
        ]
        if not any(isinstance(count, int) and count > 0 for count, _, _ in cells):
            continue
        code = f"aggregate_confusion_{left_label.split()[0].lower()}_{right_label.split()[0].lower()}"
        for true_label, count, peer in ((left_label, cells[0][0], right_label), (right_label, cells[1][0], left_label)):
            pool = [row for row in evaluation if row["Ticket Type"] == true_label]
            evidence = (
                f"Saved aggregate TF-IDF confusion matrix has {count} {true_label!r}→{peer!r} cases. "
                "No row-level predictions are available; this is a category-comparison sample, not a claimed misclassification."
            )
            for row in _choose(pool, max(1, limit_per_group // 4), code + true_label):
                reason_candidates[code].append((row, evidence, []))

    # Select a balanced evaluation sample, favoring examples with no other
    # automated flag so annotators see ordinary cases as well.
    already_flagged = {
        row["Ticket ID"] for candidates in reason_candidates.values() for row, _evidence, _neighbors in candidates
    }
    for label in LABELS:
        pool = [row for row in evaluation if row["Ticket Type"] == label]
        unflagged = [row for row in pool if row["Ticket ID"] not in already_flagged]
        chosen = _choose(unflagged, balanced_per_category, "balanced:" + label)
        if len(chosen) < balanced_per_category:
            chosen += _choose(
                [row for row in pool if row not in chosen],
                balanced_per_category - len(chosen), "balanced-fill:" + label,
            )
        for row in chosen:
            reason_candidates["balanced_category_sample"].append((
                row,
                "Deterministic category-balanced evaluation sample; selected for human context, not because an error was established.",
                [],
            ))
            if row["Ticket ID"] not in already_flagged:
                reason_candidates["balanced_unflagged_sample"].append((
                    row, "Selected as an unflagged ordinary example for category context.", [],
                ))

    omitted: dict[str, int] = {}
    for code, candidates in sorted(reason_candidates.items()):
        # Do not cap per-ticket pair members independently in a way that loses
        # neighbor coverage; each evidence group is capped deterministically.
        dedup: dict[str, tuple[dict[str, str], str, set[str]]] = {}
        for row, evidence, neighbors in candidates:
            entry = dedup.setdefault(row["Ticket ID"], (row, evidence, set()))
            entry[2].update(neighbors)
        ordered = sorted(dedup.values(), key=lambda item: (_stable_key(item[0], code), item[0]["Ticket ID"]))
        if code in {"balanced_category_sample", "balanced_unflagged_sample"}:
            kept = ordered
        else:
            by_category: dict[str, list[tuple[dict[str, str], str, set[str]]]] = defaultdict(list)
            for entry in ordered:
                by_category[entry[0].get("Ticket Type", "")].append(entry)
            kept = []
            omitted[code] = 0
            for label in LABELS:
                values = by_category[label]
                kept.extend(values[:limit_per_group])
                omitted[code] += max(0, len(values) - limit_per_group)
        for row, evidence, neighbors in kept:
            _add_reason(selected, row, code, evidence, sorted(neighbors))

    queue = []
    for ticket_id in sorted(selected, key=lambda value: (
        selected[value]["row"]["split"],
        (0, int(value)) if value.isdigit() else (1, value),
    )):
        item = selected[ticket_id]
        row = item["row"]
        queue.append({
            "ticket_id": ticket_id,
            "rag_id": row.get("rag_id", ""),
            "split": row["split"],
            "ground_truth_category_for_human_review_only": row["Ticket Type"],
            "subject_redacted": _safe_excerpt(row.get("Ticket Subject"), 500),
            "description_redacted": _safe_excerpt(row.get("Ticket Description"), 1200),
            "review_reason_codes": "|".join(sorted(item["reasons"])),
            "evidence": " || ".join(item["reasons"][key] for key in sorted(item["reasons"])),
            "similar_ticket_ids_and_categories": " | ".join(sorted(item["neighbors"])),
            "automated_flags_are_not_proof_of_error": "true",
            "human_review_decision": "",
            "reviewer_notes": "",
        })
    reason_rows = Counter()
    split_counts = Counter()
    category_counts: dict[str, Counter[str]] = {split: Counter() for split in ("reference", "evaluation")}
    evidence_cross_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {split: Counter() for split in ("reference", "evaluation")}
    )
    for item in queue:
        split_counts[item["split"]] += 1
        category_counts[item["split"]][item["ground_truth_category_for_human_review_only"]] += 1
        for reason in item["review_reason_codes"].split("|"):
            reason_rows[reason] += 1
            evidence_cross_counts[reason][item["split"]][item["ground_truth_category_for_human_review_only"]] += 1
    summary = {
        "read_only": True,
        "models_fit_or_called": False,
        "supabase_accessed": False,
        "source_files_modified": False,
        "evaluation_labels_used_for_training": False,
        "evaluation_label_usage": "Used only to stratify diagnostic human-review examples and show ground truth to the reviewer.",
        "total_reference_rows": len(reference),
        "total_evaluation_rows": len(evaluation),
        "queue_rows": len(queue),
        "unique_ticket_ids": len({item["ticket_id"] for item in queue}),
        "queue_counts_by_split": {key: split_counts[key] for key in ("reference", "evaluation")},
        "queue_counts_by_category_and_split": {
            split: {label: category_counts[split][label] for label in LABELS}
            for split in ("reference", "evaluation")
        },
        "unique_ticket_count_by_reason": {key: reason_rows[key] for key in sorted(reason_rows)},
        "reason_membership_counts_by_split_and_category": {
            reason: {
                split: {label: evidence_cross_counts[reason][split][label] for label in LABELS}
                for split in ("reference", "evaluation")
            }
            for reason in sorted(evidence_cross_counts)
        },
        "reason_group_candidates_omitted_by_limit": {key: value for key, value in sorted(omitted.items()) if value},
        "cross_category_text_matches": {
            "method": "Exact normalized rag_text equality; approximate word-trigram Jaccard with rare-trigram blocking.",
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "blocking_max_document_frequency": max(2, int((len(reference) + len(evaluation)) * TRIGRAM_MAX_DOCUMENT_FREQUENCY)),
            "exact_cross_category_pairs": exact_cross,
            "approximate_cross_category_pairs": approximate_cross,
            "category_pair_counts": {key: cross_pair_summary[key] for key in sorted(cross_pair_summary)},
        },
        "selection_limits": {"limit_per_reason_and_category": limit_per_group, "balanced_evaluation_examples_per_category": balanced_per_category},
        "review_guidance": [
            "Review the original source context and routing policy before deciding a label is wrong.",
            "Automated similarity, generic wording, lexical-cue, and aggregate confusion flags indicate review candidates only.",
            "The saved aggregate confusion report has no per-ticket predictions; category-pair samples are not asserted misclassifications.",
            "Evaluation ground-truth labels appear only for diagnostic human review; do not use this queue to train or tune on the evaluation split.",
            "Record decisions and notes in the blank columns; this script does not alter source labels or split files.",
        ],
    }
    return queue, summary


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Ticket Routing Human Review Queue",
        "",
        f"Read-only queue with **{summary['queue_rows']:,} unique tickets** from {summary['total_reference_rows']:,} reference and {summary['total_evaluation_rows']:,} evaluation rows.",
        "",
        "## Queue counts",
        "",
        "| Evidence group | Unique tickets |",
        "|---|---:|",
    ]
    for reason, count in summary["unique_ticket_count_by_reason"].items():
        lines.append(f"| {reason} | {count:,} |")
    lines += ["", "### Queue membership by split", "", "| Split | Tickets |", "|---|---:|"]
    for split, count in summary["queue_counts_by_split"].items():
        lines.append(f"| {split} | {count:,} |")
    lines += ["", "### Queue membership by category", "", "| Split | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)]
    for split, counts in summary["queue_counts_by_category_and_split"].items():
        lines.append("| " + split + " | " + " | ".join(str(counts[label]) for label in LABELS) + " |")
    lines += ["", "### Evidence membership by split and category", "", "| Evidence group | Split | " + " | ".join(LABELS) + " |", "|---|---|" + "---:|" * len(LABELS)]
    for reason, split_map in summary["reason_membership_counts_by_split_and_category"].items():
        for split in ("reference", "evaluation"):
            counts = split_map[split]
            lines.append("| " + reason + " | " + split + " | " + " | ".join(str(counts[label]) for label in LABELS) + " |")
    matches = summary["cross_category_text_matches"]
    lines += [
        "", "## Cross-category text matches", "",
        f"Exact conflicting pairs: {matches['exact_cross_category_pairs']:,}. Approximate conflicting pairs: {matches['approximate_cross_category_pairs']:,} at word-trigram Jaccard ≥ {matches['threshold']:.2f}.",
        "These are candidate conflicts, not confirmed label errors. Category-pair counts: " + ("; ".join(f"{pair}: {count}" for pair, count in matches["category_pair_counts"].items()) or "none"),
        "", "## Human review procedure", "",
    ]
    lines.extend(f"- {tip}" for tip in summary["review_guidance"])
    lines += [
        "", "## Privacy and scope", "",
        "Ticket subject and description excerpts are redacted using the existing audit helper. Customer name/email/age/gender columns are not read into the queue. The script does not fit or call models, access Supabase, or modify source data, labels, split assignments, or embeddings.",
        "", "The CSV contains blank `human_review_decision` and `reviewer_notes` columns for annotation. The ground-truth category is included only for human diagnostic review.", "",
    ]
    return "\n".join(lines)


def write_outputs(queue: list[dict[str, str]], summary: dict[str, Any], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(queue)
    md_path.write_text(markdown_summary(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-group", type=int, default=30)
    parser.add_argument("--balanced-per-category", type=int, default=8)
    args = parser.parse_args()
    reference = read_split(REFERENCE_CSV, "reference")
    evaluation = read_split(EVALUATION_CSV, "evaluation")
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8")) if BASELINE_REPORT.exists() else {}
    queue, summary = build_review_queue(
        reference, evaluation, limit_per_group=args.limit_per_group,
        balanced_per_category=args.balanced_per_category, baseline=baseline,
    )
    write_outputs(queue, summary, REPORT_CSV, REPORT_MARKDOWN)
    print(json.dumps({"csv": str(REPORT_CSV), "markdown": str(REPORT_MARKDOWN), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
