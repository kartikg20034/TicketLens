import csv
import tempfile
import unittest
from pathlib import Path

from scripts import ingest_synthetic_train as ingestion
from scripts import evaluate_synthetic_routing as evaluation


LABELS = ingestion.LABELS


def make_row(ticket_id, label, split="train", group=None, **overrides):
    row = {
        "ticket_id": ticket_id,
        "app_name": "Example App",
        "category": label,
        "subject": "A useful subject",
        "description": "The user reports an issue with a fictional transaction.",
        "rag_text": "App: Example App\nSubject: A useful subject\nDescription: A fictional issue.",
        "created_at": "2026-01-01T10:00:00+00:00",
        "city": "Pune",
        "channel": "Chat",
        "priority": "Medium",
        "status": "Open",
        "device_model": "Phone X",
        "os_version": "Android 15",
        "app_version": "1.0.0",
        "transaction_amount_inr": "100",
        "transaction_state": "pending",
        "trip_origin": "",
        "trip_destination": "",
        "event_time_local": "10:00",
        "first_response_minutes": "5",
        "resolution_minutes": "30",
        "scenario_group_id": group or f"group-{ticket_id}",
        "split": split,
        "source_type": "synthetic_public_workflow_inspired",
        "is_synthetic": "true",
    }
    row.update(overrides)
    return row


class SyntheticDatasetSafetyTests(unittest.TestCase):
    def test_real_dataset_schema_and_split_counts(self):
        rows = ingestion.load_dataset()
        self.assertEqual(len(rows), 1000)
        train = ingestion.select_training_rows(rows)
        self.assertEqual(len(train), 700)
        self.assertTrue(all(row["split"] == "train" for row in train))

    def test_embedding_input_never_contains_target_category(self):
        for index, label in enumerate(LABELS):
            row = make_row(str(index), label)
            text = ingestion.synthetic_embedding_text(row)
            self.assertEqual(text, row["rag_text"])
            self.assertNotIn(label.casefold(), text.casefold())
            self.assertNotIn("Category:", text)
            self.assertEqual(evaluation.tfidf_feature(row), text)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "leakage"):
                ingestion.synthetic_embedding_text({**row, "rag_text": f"Subject: issue\nCategory: {label}"})

    def test_nontraining_rows_never_reach_ingestion(self):
        rows = [make_row(f"train-{index}", label, "train") for index, label in enumerate(LABELS)]
        rows.append(make_row("held-out", LABELS[1], "validation"))
        train = ingestion.select_training_rows(rows)
        self.assertEqual(len(train), len(LABELS))
        self.assertTrue(all(row["split"] == "train" for row in train))
        with self.assertRaisesRegex(ValueError, "training ticket"):
            ingestion.ticket_values(rows[-1])
        with self.assertRaisesRegex(ValueError, "training rows only"):
            ingestion.ingest_training_rows(rows, "unused")

    def test_scenario_group_must_not_cross_splits(self):
        rows = [
            make_row("a", LABELS[0], "train", group="same-scenario"),
            make_row("b", LABELS[1], "validation", group="same-scenario"),
        ]
        with self.assertRaisesRegex(ValueError, "cross splits"):
            ingestion.validate_dataset_rows(rows)

    def test_namespaced_database_ids_avoid_existing_id_collisions(self):
        db_id = ingestion.namespaced_rag_id("SYN-01-001-1")
        self.assertEqual(db_id, "synthetic:SYN-01-001-1")
        self.assertFalse(ingestion.find_existing_id_collisions({"SYN-01-001-1"}, {db_id}, {"ticket-1", "100"}))
        self.assertEqual(
            ingestion.find_existing_id_collisions({"SYN-01-001-1"}, {db_id}, {"SYN-01-001-1"}),
            {"SYN-01-001-1"},
        )
        self.assertEqual(
            ingestion.find_existing_id_collisions({"SYN-01-001-1"}, {db_id}, {db_id}),
            {db_id},
        )

    def test_schema_and_label_validation_fail_closed(self):
        good = [make_row(str(index), label) for index, label in enumerate(LABELS)]
        ingestion.validate_dataset_rows(good)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            ingestion.validate_dataset_rows([{key: value for key, value in good[0].items() if key != "rag_text"}])
        with self.assertRaisesRegex(ValueError, "category outside"):
            ingestion.validate_dataset_rows([{**good[0], "category": "Other"}])
        with self.assertRaisesRegex(ValueError, "Duplicate ticket_id"):
            ingestion.validate_dataset_rows(good + [dict(good[0])])

    def test_test_split_requires_explicit_final_comparison(self):
        evaluation.validate_test_release("validation", False)
        with self.assertRaisesRegex(ValueError, "held back"):
            evaluation.validate_test_release("test", False)
        evaluation.validate_test_release("test", True)

    def test_tfidf_vocabulary_fits_training_rows_only(self):
        train = [
            make_row(f"train-{index}", label, "train", rag_text=f"Subject shared trainword{index}")
            for index, label in enumerate(LABELS)
        ]
        validation = [
            make_row(f"val-{index}", label, "validation", rag_text=f"Subject evalonlyquokka{index}")
            for index, label in enumerate(LABELS)
        ]
        # Use different category-free documents while preserving a fit-able corpus.
        for row in train:
            row["rag_text"] += " ordinary support wording"
        for row in validation:
            row["rag_text"] += " ordinary support wording"
        # Each row's category must not occur in text; the test-only vocabulary is absent in train.
        metrics = evaluation.tfidf_metrics(train, validation)
        self.assertEqual(metrics["training_ticket_count"], len(train))
        self.assertEqual(metrics["evaluation_rows_used_for_fit"], 0)
        self.assertEqual(metrics["feature_source"], "rag_text only; category is used only as y")

    def test_migration_is_additive_and_has_train_only_bge_retrieval(self):
        root = Path(__file__).resolve().parents[1]
        migration = (root / "supabase/migrations/003_synthetic_ticket_experiment.sql").read_text()
        self.assertIn("create table if not exists public.synthetic_tickets", migration)
        self.assertIn("create table if not exists public.synthetic_ticket_embeddings", migration)
        self.assertIn("where t.split = 'train'", migration)
        self.assertIn("e.embedding_model = 'BAAI/bge-base-en-v1.5'", migration)
        self.assertNotIn("drop table", migration.casefold())
        self.assertNotIn("delete from", migration.casefold())


class ExistingSyntheticStateValidationTests(unittest.TestCase):
    def setUp(self):
        self.incoming = make_row("SYN-001", LABELS[0], "train")
        self.rag_id = ingestion.namespaced_rag_id(self.incoming["ticket_id"])
        self.stored_ticket = (
            self.rag_id,
            self.incoming["ticket_id"],
            ingestion.DATASET_NAME,
            self.incoming["category"],
            self.incoming["rag_text"],
        )

    def state(self, ticket_rows, orphan=None, embedding_rows=()):
        connection = _FakeStateConnection(ticket_rows, orphan, embedding_rows)
        return ingestion.existing_synthetic_state(connection, [self.incoming])

    def test_matching_ticket_with_missing_embedding_is_verified_and_can_be_embedded(self):
        tickets, embeddings = self.state([self.stored_ticket])
        self.assertEqual(tickets, {self.rag_id})
        self.assertEqual(embeddings, set())

    def test_mismatched_rag_text_aborts_with_ticket_id_and_field(self):
        stored = (*self.stored_ticket[:4], "different rag text")
        with self.assertRaisesRegex(ValueError, f"{self.rag_id}: field rag_text differs"):
            self.state([stored])

    def test_mismatched_category_aborts_with_ticket_id_and_field(self):
        stored = (*self.stored_ticket[:3], LABELS[3], self.stored_ticket[4])
        with self.assertRaisesRegex(ValueError, f"{self.rag_id}: field category differs"):
            self.state([stored])

    def test_existing_embedding_is_not_considered_when_ticket_metadata_conflicts(self):
        stored = (*self.stored_ticket[:4], "different rag text")
        embedding = (self.rag_id, ingestion.MODEL_ID, ingestion.EMBEDDING_DIMENSIONS)
        with self.assertRaisesRegex(ValueError, "field rag_text differs"):
            self.state([stored], embedding_rows=[embedding])

    def test_orphaned_embedding_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "Orphaned synthetic embedding orphan-id"):
            self.state([], orphan=("orphan-id",))

    def test_matching_ticket_and_embedding_are_reused_only_with_model_and_dimension_match(self):
        embedding = (self.rag_id, ingestion.MODEL_ID, ingestion.EMBEDDING_DIMENSIONS)
        tickets, embeddings = self.state([self.stored_ticket], embedding_rows=[embedding])
        self.assertEqual(tickets, {self.rag_id})
        self.assertEqual(embeddings, {self.rag_id})
        for model, dimensions, expected in (
            ("other-model", 768, "embedding_model"),
            (ingestion.MODEL_ID, 384, "embedding_dimensions"),
        ):
            with self.subTest(model=model, dimensions=dimensions), self.assertRaisesRegex(ValueError, expected):
                self.state([self.stored_ticket], embedding_rows=[(self.rag_id, model, dimensions)])


class _FakeStateCursor:
    def __init__(self, ticket_rows, orphan, embedding_rows):
        self.ticket_rows = ticket_rows
        self.orphan = orphan
        self.embedding_rows = embedding_rows
        self.current = []
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, query, _params=None):
        self.query = query
        if "SELECT rag_id, source_ticket_id, dataset_name, category, rag_text" in query:
            self.current = self.ticket_rows
        elif "LEFT JOIN public.synthetic_tickets" in query:
            self.current = [self.orphan] if self.orphan else []
        elif "SELECT ticket_id, embedding_model, embedding_dimensions" in query:
            self.current = self.embedding_rows
        else:
            raise AssertionError(f"Unexpected query in test: {query}")

    def fetchall(self):
        return list(self.current)

    def fetchone(self):
        return self.current[0] if self.current else None


class _FakeStateConnection:
    def __init__(self, ticket_rows, orphan, embedding_rows):
        self.cursor_value = _FakeStateCursor(ticket_rows, orphan, embedding_rows)

    def cursor(self):
        return self.cursor_value


if __name__ == "__main__":
    unittest.main()
