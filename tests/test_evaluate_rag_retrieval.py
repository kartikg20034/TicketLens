import unittest

from scripts.evaluate_rag_retrieval import (
    LABELS,
    classification_metrics,
    majority_vote,
    validate_matches,
)


class MetricTests(unittest.TestCase):
    def test_majority_vote_ties_follow_highest_retrieval_rank(self):
        self.assertEqual(
            majority_vote(["Refund request", "Billing inquiry", "Refund request", "Billing inquiry", "Technical issue"]),
            "Refund request",
        )

    def test_accuracy_class_metrics_and_confusion_matrix(self):
        metrics = classification_metrics(
            ["Billing inquiry", "Cancellation request"],
            ["Billing inquiry", "Billing inquiry"],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / len(LABELS))
        self.assertEqual(metrics["per_class"]["Billing inquiry"]["precision"], 0.5)
        self.assertEqual(metrics["per_class"]["Billing inquiry"]["recall"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"]["Cancellation request"]["Billing inquiry"], 1
        )

    def test_empty_success_set_is_reported_as_unavailable_not_zero_accuracy(self):
        metrics = classification_metrics([], [])
        self.assertIsNone(metrics["accuracy"])
        self.assertIsNone(metrics["macro_f1"])
        self.assertEqual(sum(sum(row.values()) for row in metrics["confusion_matrix"].values()), 0)
        self.assertIsNone(metrics["per_class"]["Technical issue"]["f1"])

    def test_unknown_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the five categories"):
            classification_metrics(["Billing inquiry"], ["unknown"])


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _query, _params):
        return None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)


class ResultValidationTests(unittest.TestCase):
    def test_accepts_only_verified_reference_bge_results(self):
        labels = list(LABELS)
        matches = [
            (f"id-{i}", f"Subject {i}", "Product", label, None, 0.9 - i * 0.01)
            for i, label in enumerate(labels)
        ]
        metadata = [(f"id-{i}", "reference", "BAAI/bge-base-en-v1.5", 768) for i in range(5)]
        verified = validate_matches(FakeConnection(metadata), matches)
        self.assertEqual(verified[0]["ticket_id"], "id-0")
        self.assertEqual(len(verified), 5)

    def test_rejects_wrong_split_or_embedding_model(self):
        matches = [
            (f"id-{i}", f"Subject {i}", "Product", LABELS[i], None, 0.9 - i * 0.01)
            for i in range(5)
        ]
        wrong_model = [(f"id-{i}", "reference", "gemini-embedding-2", 768) for i in range(5)]
        with self.assertRaisesRegex(ValueError, "required BGE embedding"):
            validate_matches(FakeConnection(wrong_model), matches)

        wrong_split = [(f"id-{i}", "evaluation", "BAAI/bge-base-en-v1.5", 768) for i in range(5)]
        with self.assertRaisesRegex(ValueError, "outside the reference split"):
            validate_matches(FakeConnection(wrong_split), matches)

    def test_requires_five_unique_results(self):
        with self.assertRaisesRegex(ValueError, "Expected 5 results"):
            validate_matches(FakeConnection([]), [])


if __name__ == "__main__":
    unittest.main()
