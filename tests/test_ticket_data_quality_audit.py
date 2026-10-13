import unittest

from scripts.audit_ticket_data_quality import (
    LABELS,
    _length_summary,
    build_report,
    evaluate_text_variants,
    safe_feature_text,
    similarity_audit,
    text_variants,
    _safe_excerpt,
)


def row(ticket_id, label, split, rag_text, subject="General request", description=None):
    return {
        "Ticket ID": str(ticket_id),
        "Ticket Type": label,
        "split": split,
        "Ticket Subject": subject,
        "Ticket Description": description if description is not None else rag_text,
        "rag_text": rag_text,
    }


class DataQualityAuditTests(unittest.TestCase):
    def test_target_label_and_pii_columns_never_enter_text_variants(self):
        record = row(
            "1", "Billing inquiry", "reference",
            "Product: Device\nTicket Type: Billing inquiry\nSubject: Help\nDescription: Payment failed user@example.com",
            subject="Help",
            description="Payment failed user@example.com",
        )
        variants = text_variants(record)
        self.assertTrue(all("Ticket Type:" not in text for text in variants.values()))
        self.assertTrue(all("Billing inquiry" not in text for text in variants.values()))
        self.assertTrue(all("user@example.com" not in text for text in variants.values()))
        changed_target = dict(record, **{"Ticket Type": "Refund request"})
        self.assertEqual(text_variants(record), text_variants(changed_target))

    def test_identical_text_with_conflicting_labels_is_reported_as_exact_conflict(self):
        records = [
            row("1", LABELS[0], "reference", "Product: X\nSubject: A\nDescription: same wording"),
            row("2", LABELS[3], "reference", " product: x  subject: a description: same wording "),
            row("3", LABELS[1], "evaluation", "Product: Y\nSubject: B\nDescription: distinct"),
        ]
        result = similarity_audit(records)
        self.assertEqual(result["exact"]["duplicate_groups"], 1)
        self.assertEqual(result["exact"]["cross_category_conflict_groups"], 1)
        self.assertEqual(result["exact"]["cross_category_conflicting_pairs"], 1)
        self.assertEqual(result["approximate"]["cross_category_conflicting_pairs"], 0)
        self.assertEqual(result["conflict_examples"][0]["match_type"], "exact")

    def test_near_duplicate_conflicts_are_separate_from_exact_matches(self):
        base = "Product device Subject issue Description " + " ".join(f"detail{i}" for i in range(36))
        near = base.replace("detail18", "detail18 changed")
        records = [
            row("1", LABELS[0], "reference", base),
            row("2", LABELS[4], "reference", near),
        ]
        result = similarity_audit(records)
        self.assertEqual(result["exact"]["cross_category_conflict_groups"], 0)
        self.assertEqual(result["approximate"]["cross_category_conflicting_pairs"], 1)
        self.assertEqual(result["conflict_examples"][0]["match_type"], "approximate")

    def test_evaluation_rows_are_scoring_only_and_vocabulary_is_reference_only(self):
        reference = [
            row(f"r-{index}", label, "reference", f"Product common Subject refword{index}")
            for index, label in enumerate(LABELS)
        ]
        evaluation = [
            row(f"e-{index}", label, "evaluation", f"Product common evalonlyquokka{index}")
            for index, label in enumerate(LABELS)
        ]
        results, vocabularies = evaluate_text_variants(reference, evaluation)
        for variant in results:
            self.assertEqual(results[variant]["vectorizer_fit_rows"], len(reference))
            self.assertEqual(results[variant]["evaluation_rows_used_for_fit"], 0)
            self.assertFalse(any("evalonlyquokka" in token for token in vocabularies[variant]))

        evaluation[0]["split"] = "reference"
        with self.assertRaisesRegex(ValueError, "evaluation split"):
            evaluate_text_variants(reference, evaluation)

    def test_empty_and_missing_text_have_stable_quality_counts(self):
        self.assertEqual(safe_feature_text(None), "")
        self.assertEqual(safe_feature_text("  \n  "), "")
        self.assertEqual(_length_summary([None, "", "brief"])["empty"], 2)

    def test_report_examples_redact_free_text_names_and_handles(self):
        excerpt = _safe_excerpt(
            "— Brian J. Jones (@BJJones) February 18. [3/1/2013 3:09 PM] Aaron Arth. My daughter needs stitches."
        )
        self.assertNotIn("Brian J. Jones", excerpt)
        self.assertNotIn("@BJJones", excerpt)
        self.assertNotIn("Aaron Arth", excerpt)
        self.assertNotIn("My daughter", excerpt)

    def test_report_generation_is_deterministic(self):
        reference = [
            row(f"r-{index}", label, "reference", f"Subject commonword{index} Description issue", subject=f"Subject {index}")
            for index, label in enumerate(LABELS)
        ]
        evaluation = [
            row(f"e-{index}", label, "evaluation", f"Subject commonword{index} Description issue", subject=f"Subject {index}")
            for index, label in enumerate(LABELS)
        ]
        combined = reference + evaluation
        first = build_report(combined, reference, evaluation)
        second = build_report(combined, reference, evaluation)
        self.assertEqual(first, second)
        self.assertEqual(len(first["balanced_evaluation_examples"]), len(LABELS))


if __name__ == "__main__":
    unittest.main()
