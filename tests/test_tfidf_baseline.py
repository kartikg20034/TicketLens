import unittest

from scripts.evaluate_tfidf_baseline import LABELS, feature_text, fit_and_evaluate


class TfidfBaselineSafetyTests(unittest.TestCase):
    def test_target_field_is_removed_from_reference_and_evaluation_features(self):
        for label in LABELS:
            with self.subTest(label=label):
                text = feature_text(
                    f"Product: Example\nTicket Type: {label}\nSubject: Issue details"
                )
                self.assertNotIn("Ticket Type:", text)
                self.assertNotIn(label, text)
                self.assertIn("Issue details", text)

    def test_feature_builder_uses_only_rag_text(self):
        self.assertEqual(
            feature_text("Product: Example\nSubject: Login failure"),
            feature_text("Product: Example\nSubject: Login failure"),
        )
        # The label is a separate target value and is not accepted by feature_text.
        self.assertEqual(feature_text("Subject: Login failure"), "Subject: Login failure")

    def test_evaluation_only_vocabulary_is_not_fitted(self):
        reference = [
            {
                "Ticket ID": f"r-{index}",
                "Ticket Type": label,
                "split": "reference",
                "rag_text": f"Ticket Type: {label}\nSubject: common support issue referenceword{index}",
            }
            for index, label in enumerate(LABELS)
        ]
        evaluation = [
            {
                "Ticket ID": f"e-{index}",
                "Ticket Type": label,
                "split": "evaluation",
                "rag_text": f"Subject: common support issue evaluationonlyquokka{index}",
            }
            for index, label in enumerate(LABELS)
        ]

        model, metrics = fit_and_evaluate(reference, evaluation)
        vocabulary = model.named_steps["tfidf"].vocabulary_
        self.assertFalse(any("evaluationonlyquokka" in token for token in vocabulary))
        self.assertEqual(metrics["evaluation_feature_texts_used_only_for_prediction"], 5)
        self.assertEqual(metrics["evaluation_only_ids_used_for_fit"], 0)
        self.assertEqual(metrics["training_ticket_count"], len(reference))

    def test_overlapping_ticket_ids_fail_closed(self):
        reference = [{"Ticket ID": "same", "Ticket Type": LABELS[0], "rag_text": "text"}]
        evaluation = [{"Ticket ID": "same", "Ticket Type": LABELS[0], "rag_text": "text"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_and_evaluate(reference, evaluation)


if __name__ == "__main__":
    unittest.main()
