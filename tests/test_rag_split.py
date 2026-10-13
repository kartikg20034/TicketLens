import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/processed/rag_reference.csv"
EVALUATION = ROOT / "data/processed/rag_evaluation.csv"
REPORT = ROOT / "data/processed/rag_split_report.json"
SCRIPT = ROOT / "scripts/create_rag_split.py"
LABELS = {"Billing inquiry", "Cancellation request", "Product inquiry", "Refund request", "Technical issue"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class RagSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Rebuild outputs from the checked-in cleaned dataset, using a temp cwd
        # only for stdout capture; the script writes the required project files.
        import subprocess
        subprocess.run([sys.executable, str(SCRIPT)], check=True, capture_output=True, text=True)
        cls.reference = read_csv(REFERENCE)
        cls.evaluation = read_csv(EVALUATION)

    def test_no_rag_id_or_group_crosses_splits(self):
        ref_ids = {row["rag_id"] for row in self.reference}
        eval_ids = {row["rag_id"] for row in self.evaluation}
        self.assertFalse(ref_ids & eval_ids)
        ref_groups = {row["duplicate_group_id"] for row in self.reference}
        eval_groups = {row["duplicate_group_id"] for row in self.evaluation}
        self.assertFalse(ref_groups & eval_groups)

    def test_all_labels_present_and_eval_label_is_not_prompt_context(self):
        self.assertEqual({row["Ticket Type"] for row in self.reference}, LABELS)
        self.assertEqual({row["Ticket Type"] for row in self.evaluation}, LABELS)
        self.assertTrue(all(row["split"] == "reference" for row in self.reference))
        self.assertTrue(all(row["split"] == "evaluation" for row in self.evaluation))
        self.assertTrue(all("Ticket Type:" not in row["rag_text"] for row in self.evaluation))

    def test_split_ratio_is_between_75_25_and_85_15(self):
        total = len(self.reference) + len(self.evaluation)
        ref_ratio = len(self.reference) / total
        self.assertGreaterEqual(ref_ratio, 0.75)
        self.assertLessEqual(ref_ratio, 0.85)

    def test_report_group_crossing_validation(self):
        import json
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["group_leakage_validation_passed"])
        self.assertEqual(report["groups_crossing_split"], 0)


if __name__ == "__main__":
    unittest.main()
