#!/usr/bin/env python3
"""Create deterministic, duplicate-grouped reference/evaluation CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/customer_support_tickets_clean.csv"
REFERENCE = ROOT / "data/processed/rag_reference.csv"
EVALUATION = ROOT / "data/processed/rag_evaluation.csv"
REPORT = ROOT / "data/processed/rag_split_report.json"
LABELS = (
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SIMILARITY_THRESHOLD = 0.82


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _similarity_groups(rows: list[dict[str, str]]) -> tuple[list[str], int, int]:
    """Group exact duplicates and near duplicates using word-trigram Jaccard.

    Candidate pairs share at least one word trigram that occurs in no more than
    1% of the corpus. The deterministic Jaccard threshold limits generic
    boilerplate from joining otherwise unrelated records.
    """
    count = len(rows)
    uf = UnionFind(count)
    normalized = [row["normalized_duplicate_text"].casefold().strip() for row in rows]

    exact_buckets: dict[str, list[int]] = defaultdict(list)
    for i, value in enumerate(normalized):
        exact_buckets[value].append(i)
    exact_group_count = sum(len(indices) > 1 for indices in exact_buckets.values())
    for indices in exact_buckets.values():
        for i in indices[1:]:
            uf.union(indices[0], i)

    shingles: list[set[tuple[str, str, str]]] = []
    postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, value in enumerate(normalized):
        words = TOKEN_RE.findall(value)
        grams = set(zip(words, words[1:], words[2:]))
        shingles.append(grams)
        for gram in grams:
            postings[gram].append(i)

    max_document_frequency = max(2, math.floor(count * 0.01))
    candidates: set[tuple[int, int]] = set()
    for indexes in postings.values():
        if len(indexes) > max_document_frequency:
            continue
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1:]:
                candidates.add((left, right))

    near_edges = 0
    for left, right in candidates:
        if normalized[left] == normalized[right]:
            continue
        a, b = shingles[left], shingles[right]
        if not a or not b:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity >= SIMILARITY_THRESHOLD:
            uf.union(left, right)
            near_edges += 1

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(count):
        components[uf.find(i)].append(i)

    component_types: dict[int, str] = {}
    near_group_count = 0
    for root, members in components.items():
        is_exact = len(members) > 1 and len({normalized[i] for i in members}) == 1
        if len(members) > 1 and not is_exact:
            component_types[root] = "near"
            near_group_count += 1
        elif len(members) > 1:
            component_types[root] = "exact"
        else:
            component_types[root] = "single"

    group_ids = [""] * count
    for root, members in components.items():
        member_ids = sorted(f"ticket-{rows[i]['Ticket ID']}" for i in members)
        group_id = _stable_id("group", "\n".join(member_ids))
        for i in members:
            group_ids[i] = group_id
    return group_ids, exact_group_count, near_group_count


def _stratified_group_assignment(rows: list[dict[str, str]], group_ids: list[str]) -> dict[str, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        groups[group_id].append(i)
    totals = Counter(row["Ticket Type"] for row in rows)
    if set(totals) != set(LABELS):
        raise ValueError(f"Expected all five Ticket Type labels; found {sorted(totals)}")
    targets = {label: totals[label] * 0.20 for label in LABELS}

    def score(eval_counts: Counter[str], eval_total: int) -> float:
        label_error = sum(((eval_counts[label] - targets[label]) ** 2) / totals[label] for label in LABELS)
        total_error = ((eval_total - 0.20 * len(rows)) ** 2) / len(rows)
        return label_error + total_error

    order = sorted(
        groups,
        key=lambda group_id: (
            -len(groups[group_id]),
            hashlib.sha256(group_id.encode("utf-8")).hexdigest(),
        ),
    )
    eval_groups: set[str] = set()
    eval_counts: Counter[str] = Counter()
    eval_total = 0
    for group_id in order:
        counts = Counter(rows[i]["Ticket Type"] for i in groups[group_id])
        after_counts = eval_counts + counts
        if score(after_counts, eval_total + len(groups[group_id])) < score(eval_counts, eval_total):
            eval_groups.add(group_id)
            eval_counts = after_counts
            eval_total += len(groups[group_id])

    # Enforce all labels on both sides, moving the smallest suitable whole group.
    for label in LABELS:
        if eval_counts[label] == 0:
            choices = [g for g in order if g not in eval_groups and any(rows[i]["Ticket Type"] == label for i in groups[g])]
            chosen = min(choices, key=lambda g: (len(groups[g]), g))
            eval_groups.add(chosen)
            eval_counts.update(rows[i]["Ticket Type"] for i in groups[chosen])
            eval_total += len(groups[chosen])
    ref_counts = totals - eval_counts
    for label in LABELS:
        if ref_counts[label] == 0:
            choices = [g for g in order if g in eval_groups and any(rows[i]["Ticket Type"] == label for i in groups[g])]
            chosen = min(choices, key=lambda g: (len(groups[g]), g))
            eval_groups.remove(chosen)
            moved = Counter(rows[i]["Ticket Type"] for i in groups[chosen])
            eval_counts.subtract(moved)
            eval_total -= len(groups[chosen])

    # Deterministic single-group local improvements bring the result closer to 80/20.
    changed = True
    while changed:
        changed = False
        current = score(eval_counts, eval_total)
        for group_id in order:
            counts = Counter(rows[i]["Ticket Type"] for i in groups[group_id])
            size = len(groups[group_id])
            if group_id in eval_groups:
                candidate_counts = eval_counts - counts
                if any(candidate_counts[label] == 0 or totals[label] - candidate_counts[label] == 0 for label in LABELS):
                    continue
                candidate = score(candidate_counts, eval_total - size)
            else:
                candidate_counts = eval_counts + counts
                if any(totals[label] - candidate_counts[label] == 0 for label in LABELS):
                    continue
                candidate = score(candidate_counts, eval_total + size)
            if candidate + 1e-12 < current:
                if group_id in eval_groups:
                    eval_groups.remove(group_id)
                    eval_counts = candidate_counts
                    eval_total -= size
                else:
                    eval_groups.add(group_id)
                    eval_counts = candidate_counts
                    eval_total += size
                changed = True
                break

    return {group_id: ("evaluation" if group_id in eval_groups else "reference") for group_id in groups}


def create_split() -> dict:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"Ticket ID", "Ticket Type", "normalized_duplicate_text", "rag_text"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Cleaned CSV is missing required columns: {sorted(missing)}")
    bad_labels = sorted({row["Ticket Type"] for row in rows} - set(LABELS))
    if bad_labels:
        raise ValueError(f"Unexpected Ticket Type labels: {bad_labels}")

    group_ids, exact_group_count, near_group_count = _similarity_groups(rows)
    assignments = _stratified_group_assignment(rows, group_ids)
    out_fields = list(fieldnames)
    for name in ("rag_id", "duplicate_group_id", "split"):
        if name not in out_fields:
            out_fields.append(name)
    ref_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for row, group_id in zip(rows, group_ids):
        split = assignments[group_id]
        item = dict(row)
        item["rag_id"] = f"ticket-{row['Ticket ID']}"
        item["duplicate_group_id"] = group_id
        item["split"] = split
        if split == "evaluation":
            # Keep the true label column for scoring, but remove it from the
            # evaluation query text so it cannot leak into a RAG prompt.
            item["rag_text"] = "\n".join(
                line for line in row["rag_text"].splitlines()
                if not line.casefold().startswith("ticket type:")
            )
            eval_rows.append(item)
        else:
            ref_rows.append(item)

    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    for path, output_rows in ((REFERENCE, ref_rows), (EVALUATION, eval_rows)):
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(output_rows)

    ref_labels = Counter(row["Ticket Type"] for row in ref_rows)
    eval_labels = Counter(row["Ticket Type"] for row in eval_rows)
    ref_groups = {row["duplicate_group_id"] for row in ref_rows}
    eval_groups = {row["duplicate_group_id"] for row in eval_rows}
    crossing = sorted(ref_groups & eval_groups)
    total = len(rows)
    report = {
        "input_file": str(INPUT),
        "reference_file": str(REFERENCE),
        "evaluation_file": str(EVALUATION),
        "total_tickets": total,
        "reference_count": len(ref_rows),
        "evaluation_count": len(eval_rows),
        "reference_percentage": round(100 * len(ref_rows) / total, 4),
        "evaluation_percentage": round(100 * len(eval_rows) / total, 4),
        "ticket_type_counts_reference": {label: ref_labels[label] for label in LABELS},
        "ticket_type_counts_evaluation": {label: eval_labels[label] for label in LABELS},
        "duplicate_groups": exact_group_count,
        "near_duplicate_groups": near_group_count,
        "groups_crossing_split": len(crossing),
        "crossing_group_ids": crossing,
        "group_leakage_validation_passed": not crossing,
        "similarity_method": {
            "name": "word-trigram Jaccard similarity on normalized_duplicate_text",
            "threshold": SIMILARITY_THRESHOLD,
            "candidate_blocking": "shared trigram document frequency <= 1% of corpus",
            "external_api": False,
        },
        "evaluation_label_prompt_safety": "Ticket Type is retained as a scoring column; evaluation rag_text omits its Ticket Type line.",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(create_split(), indent=2, ensure_ascii=False))
