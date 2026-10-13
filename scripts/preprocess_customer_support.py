#!/usr/bin/env python3
"""Prepare the Kaggle customer support CSV for later retrieval experiments.

This is a local, dependency-free preprocessing step. It never writes to the
source CSV and deliberately does not create embeddings or call external APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/customer_support_tickets.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/customer_support_tickets_clean.csv"
PII_COLUMNS = {"Customer Name", "Customer Email", "Customer Age", "Customer Gender"}
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")
PRODUCT_PLACEHOLDER_RE = re.compile(re.escape("{product_purchased}"), re.IGNORECASE)


def normalize_whitespace(value: str | None) -> str:
    """Collapse all whitespace runs and trim the ends without changing words."""
    return " ".join((value or "").split())


def ticket_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def preprocess(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {input_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required = {"Ticket ID", "Product Purchased", "Ticket Type", "Ticket Subject", "Ticket Description", "Resolution"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    placeholder_counts: Counter[str] = Counter()
    processed: list[dict[str, str]] = []
    for raw in rows:
        product = normalize_whitespace(raw.get("Product Purchased"))
        subject = normalize_whitespace(raw.get("Ticket Subject"))
        description = normalize_whitespace(raw.get("Ticket Description"))
        description = PRODUCT_PLACEHOLDER_RE.sub(lambda _match: product, description)
        resolution = normalize_whitespace(raw.get("Resolution"))

        normalized_duplicate_text = f"{subject.casefold()}\n{description.casefold()}"
        rag_text = (
            f"Product: {product}\n"
            f"Subject: {subject}\n"
            f"Description: {description}"
        )
        has_resolution = bool(resolution)
        quality = "missing" if not has_resolution else ("very_short" if len(resolution) < 20 else "usable")

        cleaned = {name: raw.get(name, "") or "" for name in fieldnames if name not in PII_COLUMNS}
        cleaned["Ticket Subject"] = subject
        cleaned["Ticket Description"] = description
        cleaned["Resolution"] = resolution
        cleaned["rag_text"] = rag_text
        cleaned["normalized_duplicate_text"] = normalized_duplicate_text
        cleaned["has_resolution"] = str(has_resolution).lower()
        cleaned["resolution_quality"] = quality
        processed.append(cleaned)

    # Sort by Ticket ID before keeping a duplicate pair, so the smallest ID wins.
    processed.sort(key=lambda row: ticket_id_sort_key(row["Ticket ID"]))
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in processed:
        key = row["normalized_duplicate_text"]
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    for row in unique_rows:
        for match in PLACEHOLDER_RE.finditer(row["Ticket Description"]):
            placeholder_counts[match.group(0)] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [name for name in fieldnames if name not in PII_COLUMNS]
    output_fields += ["rag_text", "normalized_duplicate_text", "has_resolution", "resolution_quality"]
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique_rows)

    quality_counts = Counter(row["resolution_quality"] for row in unique_rows)
    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_row_count": len(rows),
        "cleaned_row_count": len(unique_rows),
        "exact_duplicates_removed": len(rows) - len(unique_rows),
        "usable_resolutions": quality_counts["usable"],
        "missing_resolutions": quality_counts["missing"],
        "very_short_resolutions": quality_counts["very_short"],
        "remaining_placeholder_patterns": dict(sorted(placeholder_counts.items())),
        "ticket_type_distribution": dict(sorted(Counter(row.get("Ticket Type", "") for row in unique_rows).items())),
        "product_purchased_distribution": dict(sorted(Counter(row.get("Product Purchased", "") for row in unique_rows).items())),
        "pii_columns_excluded_from_processed_output": sorted(PII_COLUMNS & set(fieldnames)),
        "near_duplicates_removed": 0,
    }
    report_path = output_path.parent / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.input, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
