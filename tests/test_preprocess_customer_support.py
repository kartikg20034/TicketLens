import csv
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/customer_support_tickets.csv"
SCRIPT = ROOT / "scripts/preprocess_customer_support.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreprocessingTest(unittest.TestCase):
    def test_pipeline_safety_and_rag_output(self):
        raw_before = sha256(SOURCE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clean.csv"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(SOURCE), "--output", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(raw_before, sha256(SOURCE), "pipeline modified the raw CSV")
        self.assertTrue(rows)
        self.assertTrue(all("{product_purchased}" not in row["rag_text"].casefold() for row in rows))
        self.assertTrue(all(not re.search(r"(?im)^\s*Ticket Type\s*:", row["rag_text"]) for row in rows))
        self.assertTrue(all(row["Ticket Type"] for row in rows), "ground-truth metadata was removed")
        email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
        with SOURCE.open(encoding="utf-8-sig", newline="") as file:
            customer_emails = {row["Customer Email"].casefold() for row in csv.DictReader(file) if row.get("Customer Email")}
        self.assertTrue(all(not (set(email_pattern.findall(row["rag_text"].casefold())) & customer_emails) for row in rows))
        keys = [row["normalized_duplicate_text"] for row in rows]
        self.assertEqual(len(keys), len(set(keys)), "exact normalized Subject + Description duplicates remain")


if __name__ == "__main__":
    unittest.main()
