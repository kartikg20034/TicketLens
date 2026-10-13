import unittest

from scripts.ingest_synthetic_train import LABELS, load_dataset
from scripts.inspect_synthetic_integrity import build_integrity_report
from scripts.route_synthetic_ticket import make_query_text, safe_preview, summarize_retrieval


class SyntheticProjectIntegrityTests(unittest.TestCase):
    def test_integrity_report_confirms_split_and_reports_no_team_mapping(self):
        report = build_integrity_report()
        self.assertEqual(report["split_counts"], {"train": 700, "validation": 150, "test": 150})
        self.assertTrue(report["group_isolation_passed"])
        self.assertEqual(len(report["category_labels"]), 5)
        self.assertEqual(report["demo_team_roster_count"], 12)
        self.assertIsNone(report["category_to_team_mapping"])
        self.assertFalse(report["team_routing_supported"])

    def test_single_ticket_input_contains_no_target_field(self):
        text = make_query_text("The app crashes when opening the saved receipt.", "Receipt screen crash")
        self.assertIn("Subject: Receipt screen crash", text)
        self.assertIn("Description: The app crashes", text)
        self.assertNotIn("Category:", text)
        self.assertNotIn("Ticket Type:", text)
        with self.assertRaisesRegex(ValueError, "target field"):
            make_query_text("Category: Billing inquiry\nAn issue happened")

    def test_retrieval_summary_uses_category_vote_and_declines_team_mapping(self):
        examples = [
            {"subject": f"Example {index}", "category": label, "similarity": 0.9 - index * 0.01}
            for index, label in enumerate(
                [LABELS[0], LABELS[1], LABELS[1], LABELS[0], LABELS[2]]
            )
        ]
        summary = summarize_retrieval("Description: An example ticket", examples)
        self.assertEqual(summary["top1_category_recommendation"], LABELS[0])
        self.assertEqual(summary["top5_majority_category"], LABELS[0])
        self.assertIsNone(summary["team_recommendation"])
        self.assertIn("no validated", summary["team_warning"])
        self.assertEqual(len(summary["similar_training_tickets"]), 5)
        self.assertIn("nearest training example", summary["rationale"])

    def test_ticket_preview_redacts_email_and_common_secret_patterns(self):
        preview = safe_preview("Contact jane@example.test; api_key=abc123; call +1 (415) 555-0123", 160)
        self.assertNotIn("jane@example.test", preview)
        self.assertNotIn("abc123", preview)
        self.assertIn("[REDACTED]", preview)
        self.assertNotIn("555-0123", preview)
        self.assertIn("[PHONE REDACTED]", preview)

    def test_dataset_loader_remains_train_validation_test_only(self):
        rows = load_dataset()
        self.assertEqual(len(rows), 1000)
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
