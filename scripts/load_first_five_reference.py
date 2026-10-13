#!/usr/bin/env python3
"""Backward-compatible entry point for the local BGE smoke ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from smoke_test_ingestion import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
