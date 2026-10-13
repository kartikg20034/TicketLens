import unittest

from scripts.diagnose_rag_retrieval import (
    LABELS,
    comparison_texts,
    deterministic_comparison_subset,
    implementation_audit,
    majority_baseline,
    text_audit,
)
from scripts.evaluate_rag_retrieval import classification_metrics


class DiagnosticLogicTests(unittest.TestCase):
    def test_reference_text_audit_detects_ground_truth_type_in_embedding_text(self):
        rows = [
            {
                "rag_text": "Product: Phone\nTicket Type: Billing inquiry\nSubject: Invoice\nDescription: Wrong total",
                "Ticket Type": "Billing inquiry",
                "Ticket Subject": "Invoice",
                "Ticket Description": "Wrong total",
            }
        ]
        audit = text_audit(rows, is_reference=True)
        self.assertEqual(audit["ticket_type_lines_in_rag_text"], 1)
        self.assertEqual(audit["ticket_type_lines_matching_ground_truth"], 1)
        self.assertEqual(audit["subject_not_represented_in_rag_text"], 0)
        self.assertEqual(audit["description_not_represented_in_rag_text"], 0)

    def test_query_comparisons_do_not_add_ground_truth_label(self):
        row = {
            "query_text": "Product: Phone\nSubject: Invoice\nDescription: Wrong total",
            "Ticket Subject": "Invoice",
            "Ticket Description": "Wrong total",
            "Ticket Type": "Billing inquiry",
        }
        variants = comparison_texts(row)
        for text in variants.values():
            self.assertNotIn("Ticket Type:", text)
            self.assertNotIn(row["Ticket Type"], text)
        self.assertIn("Wrong total", variants["subject_plus_description"])
        self.assertNotIn("Wrong total", variants["subject_only"])

    def test_comparison_subset_is_repeatable_and_label_balanced(self):
        rows = []
        for label in LABELS:
            for index in range(12):
                rows.append({"Ticket Type": label, "Ticket ID": f"{label}-{index:02}"})
        first = deterministic_comparison_subset(rows)
        second = deterministic_comparison_subset(rows)
        self.assertEqual([r["Ticket ID"] for r in first], [r["Ticket ID"] for r in second])
        self.assertEqual(len(first), 50)
        self.assertEqual({label: sum(r["Ticket Type"] == label for r in first) for label in LABELS}, {label: 10 for label in LABELS})

    def test_majority_baseline_uses_deterministic_tie_breaking(self):
        baseline = majority_baseline(["Technical issue", "Billing inquiry"])
        self.assertEqual(baseline["label"], LABELS[0])
        self.assertEqual(baseline["accuracy"], 0.5)

    def test_top1_and_top5_metrics_are_scored_from_independent_predictions(self):
        truths = ["Billing inquiry", "Cancellation request"]
        top1 = classification_metrics(truths, ["Billing inquiry", "Billing inquiry"])
        top5 = classification_metrics(truths, ["Cancellation request", "Cancellation request"])
        self.assertEqual(top1["accuracy"], 0.5)
        self.assertEqual(top5["accuracy"], 0.5)
        self.assertNotEqual(top1["confusion_matrix"], top5["confusion_matrix"])

    def test_current_implementation_checks_are_present(self):
        checks = implementation_audit()
        self.assertTrue(checks["preprocessor_excludes_ticket_type_from_rag_text"])
        self.assertTrue(checks["split_script_removes_ticket_type_only_for_evaluation_text"])
        self.assertTrue(checks["ticket_type_preserved_as_database_metadata"])
        self.assertTrue(checks["top1_uses_nearest_neighbor_label"])
        self.assertTrue(checks["top5_uses_majority_vote_of_ranked_five"])
        self.assertTrue(checks["top1_and_top5_metrics_computed_separately"])
        self.assertTrue(checks["query_text_removes_ticket_type_line"])
        self.assertTrue(checks["reference_ingestion_validates_and_encodes_label_free_text"])
        self.assertTrue(checks["reference_resume_key_is_only_model_and_dimensions"])
        self.assertFalse(checks["reference_resume_checks_text_hash"])


if __name__ == "__main__":
    unittest.main()
