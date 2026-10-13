import unittest

from scripts.audit_ticket_data_quality import LABELS
from scripts.build_ticket_review_queue import build_review_queue, find_text_matches


def row(ticket_id, label, split="evaluation", subject="A subject", description="A clear description"):
    return {
        "Ticket ID": str(ticket_id),
        "rag_id": f"ticket-{ticket_id}",
        "Ticket Type": label,
        "split": split,
        "Ticket Subject": subject,
        "Ticket Description": description,
        "rag_text": f"Subject: {subject}\nDescription: {description}",
    }


class TicketReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = [
            row(f"e{index}", label, subject=f"Ordinary subject {index}", description=f"Specific distinct description for {label} number {index}")
            for index, label in enumerate(LABELS)
        ]
        self.reference = [
            row(f"r{index}", label, "reference", subject=f"Reference subject {index}", description=f"Reference detail {index}.")
            for index, label in enumerate(LABELS)
        ]

    def test_build_is_deterministic_and_deduplicates_overlapping_reasons(self):
        target = self.evaluation[0]
        target["Ticket Description"] = "I'm having an issue. Please assist. Contact me at person@example.com {error_message}"
        target["rag_text"] = f"Subject: {target['Ticket Subject']}\nDescription: {target['Ticket Description']}"
        first, summary1 = build_review_queue(self.reference, self.evaluation, balanced_per_category=1)
        second, summary2 = build_review_queue(self.reference, self.evaluation, balanced_per_category=1)
        self.assertEqual(first, second)
        self.assertEqual(summary1, summary2)
        ids = [item["ticket_id"] for item in first]
        self.assertEqual(len(ids), len(set(ids)))
        item = next(item for item in first if item["ticket_id"] == target["Ticket ID"])
        self.assertIn("brace_placeholder", item["review_reason_codes"])
        self.assertIn("generic_description", item["review_reason_codes"])
        self.assertNotIn("person@example.com", item["description_redacted"])
        self.assertEqual(item["automated_flags_are_not_proof_of_error"], "true")

    def test_balanced_sample_covers_all_categories_and_unflagged_examples(self):
        queue, summary = build_review_queue(self.reference, self.evaluation, balanced_per_category=1)
        members = [item for item in queue if "balanced_category_sample" in item["review_reason_codes"]]
        self.assertEqual({item["ground_truth_category_for_human_review_only"] for item in members}, set(LABELS))
        self.assertEqual(summary["evaluation_labels_used_for_training"], False)
        self.assertFalse(summary["models_fit_or_called"])
        self.assertTrue(any("balanced_unflagged_sample" in item["review_reason_codes"] for item in queue))

    def test_exact_conflict_is_distinguished_from_approximate_match(self):
        details = " ".join(f"distinctiveword{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(40))
        left = row("1", LABELS[0], subject="Shared subject", description="same detailed issue " + details)
        exact = row("2", LABELS[3], subject="Shared subject", description="same detailed issue " + details)
        approximate = row("3", LABELS[4], subject="Shared subject", description="same detailed issue " + details.replace("distinctivewordau", "alteredwordau"))
        exact_matches = find_text_matches([left, exact])
        approximate_matches = find_text_matches([left, approximate])
        self.assertTrue(any(match["match_type"] == "exact" and match["cross_category"] for match in exact_matches))
        self.assertTrue(any(match["match_type"] == "approximate" for match in approximate_matches))

    def test_missing_or_empty_text_is_stable(self):
        self.evaluation[0]["Ticket Subject"] = ""
        self.evaluation[0]["Ticket Description"] = ""
        self.evaluation[0]["rag_text"] = ""
        queue, _ = build_review_queue(self.reference, self.evaluation, balanced_per_category=1)
        self.assertTrue(queue)
        self.assertTrue(all("error" not in item["review_reason_codes"] for item in queue))

    def test_split_boundary_is_validated_and_target_is_not_a_model_feature(self):
        queue, summary = build_review_queue(self.reference, self.evaluation, balanced_per_category=1)
        self.assertEqual(summary["total_reference_rows"], len(self.reference))
        self.assertEqual(summary["total_evaluation_rows"], len(self.evaluation))
        with self.assertRaisesRegex(ValueError, "non-evaluation"):
            build_review_queue(self.reference, [dict(self.evaluation[0], split="reference")])
        # This review tool has no training/vectorizer/model path; target is present
        # solely in the expressly named reviewer-facing field.
        self.assertTrue(all("ground_truth" not in key for key in queue[0] if key not in {"ground_truth_category_for_human_review_only"}))
        self.assertFalse(summary["evaluation_labels_used_for_training"])


if __name__ == "__main__":
    unittest.main()
