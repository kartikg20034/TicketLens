#!/usr/bin/env python3
"""Read-only integrity summary for the synthetic routing experiment."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.teams import TEAMS  # noqa: E402
from scripts.ingest_synthetic_train import LABELS, load_dataset  # noqa: E402


def build_integrity_report() -> dict:
    rows = load_dataset()
    split_counts = Counter(row["split"] for row in rows)
    category_counts = {
        split: {
            label: sum(row["split"] == split and row["category"] == label for row in rows)
            for label in LABELS
        }
        for split in ("train", "validation", "test")
    }
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scenario_splits[row["scenario_group_id"]].add(row["split"])
    crossing_groups = sorted(group for group, splits in scenario_splits.items() if len(splits) > 1)

    team_names = [team["name"] for team in TEAMS]
    return {
        "dataset": "data/processed/realistic_app_support_tickets_1000.csv",
        "is_synthetic": all(row["is_synthetic"].strip().casefold() in {"true", "1", "yes"} for row in rows),
        "ticket_count": len(rows),
        "split_counts": {split: split_counts[split] for split in ("train", "validation", "test")},
        "category_counts_by_split": category_counts,
        "category_labels": list(LABELS),
        "scenario_group_count": len(scenario_splits),
        "groups_crossing_splits": crossing_groups,
        "group_isolation_passed": not crossing_groups,
        "demo_team_roster_count": len(team_names),
        "demo_team_names": team_names,
        "category_to_team_mapping": None,
        "team_routing_supported": False,
        "mapping_note": (
            "The experiment has five ticket categories. The separate demo roster has twelve teams; "
            "no validated category-to-team mapping is defined. Predictions therefore remain category recommendations."
        ),
    }


def main() -> int:
    report = build_integrity_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["group_isolation_passed"] and report["is_synthetic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
