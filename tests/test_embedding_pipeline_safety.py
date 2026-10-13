import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.local_embeddings import EMBEDDING_DIMENSIONS, MODEL_ID
from scripts import ingest_reference_tickets as ingestion
from scripts import test_rag_retrieval as retrieval


ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self):
        self.query = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchall(self):
        return [("bge-ticket",)]


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()

    def cursor(self):
        return self.fake_cursor


class EmbeddingPipelineSafetyTests(unittest.TestCase):
    def test_resume_lookup_only_accepts_current_model_and_dimensions(self):
        connection = FakeConnection()
        with patch.object(ingestion, "MODEL_ID", MODEL_ID), patch.object(
            ingestion, "EMBEDDING_DIMENSIONS", EMBEDDING_DIMENSIONS
        ):
            found = ingestion.existing_bge_embeddings(connection)

        self.assertEqual(found, {"bge-ticket"})
        self.assertIn("embedding_model = %s", connection.fake_cursor.query)
        self.assertIn("embedding_dimensions = %s", connection.fake_cursor.query)
        self.assertEqual(connection.fake_cursor.params, (MODEL_ID, EMBEDDING_DIMENSIONS))

    def test_upserts_are_idempotent(self):
        self.assertIn("ON CONFLICT (rag_id) DO UPDATE", ingestion.TICKET_UPSERT)
        self.assertIn("ON CONFLICT (ticket_id) DO UPDATE", ingestion.EMBEDDING_UPSERT)

    def test_retrieval_migration_filters_reference_split_and_bge_model(self):
        migration = (ROOT / "supabase/migrations/002_bge_retrieval_filter.sql").read_text()
        self.assertIn("t.split = 'reference'", migration)
        self.assertIn("e.embedding_model = 'BAAI/bge-base-en-v1.5'", migration)

    def test_ingestion_source_only_names_reference_csv(self):
        source = (ROOT / "scripts/ingest_reference_tickets.py").read_text()
        self.assertIn('data/processed/rag_reference.csv', source)
        self.assertNotIn('rag_evaluation.csv', source)
        for name in (
            "scripts/ingest_reference_tickets.py",
            "scripts/smoke_test_ingestion.py",
            "scripts/load_first_five_reference.py",
            "scripts/test_rag_retrieval.py",
        ):
            contents = (ROOT / name).read_text().casefold()
            self.assertNotIn("google.genai", contents)
            self.assertNotIn("gemini_api_key", contents)

    def test_evaluation_query_removes_ground_truth_ticket_type(self):
        labels = (
            "Billing inquiry", "Cancellation request", "Product inquiry",
            "Refund request", "Technical issue",
        )
        for label in labels:
            with self.subTest(label=label):
                query_text = retrieval.evaluation_query_text(
                    f"Product: Example\nTicket Type: {label}\nSubject: Issue"
                )
                self.assertEqual(query_text, "Product: Example\nSubject: Issue")
                self.assertNotIn(label, query_text)
        ticket = retrieval.read_first_evaluation_ticket()
        self.assertEqual(ticket["split"], "evaluation")
        self.assertTrue(ticket["Ticket Type"])
        self.assertNotIn("Ticket Type:", ticket["query_text"])
        self.assertNotIn("Ticket Type:", ticket["rag_text"])

    def test_reference_embedding_text_excludes_target_and_keeps_label_metadata(self):
        row = {
            "rag_id": "ticket-123",
            "Ticket ID": "123",
            "Product Purchased": "Example",
            "Ticket Type": "Refund request",
            "Ticket Subject": "Refund delay",
            "Ticket Description": "The refund is late.",
            "Resolution": "",
            "resolution_quality": "missing",
            "Ticket Status": "Open",
            "Ticket Priority": "High",
            "Ticket Channel": "Email",
            "duplicate_group_id": "group-123",
            "rag_text": "Product: Example\nSubject: Refund delay\nDescription: The refund is late.",
        }
        self.assertEqual(ingestion.reference_embedding_text(row), row["rag_text"])
        self.assertEqual(row["Ticket Type"], "Refund request")
        self.assertEqual(ingestion.ticket_values(row)[3], "Refund request")
        for label in (
            "Billing inquiry", "Cancellation request", "Product inquiry",
            "Refund request", "Technical issue",
        ):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "target Ticket Type"):
                ingestion.reference_embedding_text({**row, "rag_text": f"Ticket Type: {label}\nSubject: Issue"})

    def test_rebuild_mode_bypasses_existing_embedding_cache(self):
        with patch.object(ingestion, "load_reusable_embedding_ids") as lookup:
            self.assertEqual(
                ingestion.reusable_embedding_ids("unused", rebuild_embeddings=True), set()
            )
            lookup.assert_not_called()

    def test_reference_csv_loader_fails_closed_on_target_label_leakage(self):
        row = {
            "Ticket ID": "1", "Product Purchased": "Example", "Ticket Type": "Refund request",
            "Ticket Subject": "Refund delay", "Ticket Description": "Late refund",
            "Ticket Status": "Open", "Ticket Priority": "High", "Ticket Channel": "Email",
            "Resolution": "", "resolution_quality": "missing", "rag_text": "Ticket Type: Refund request",
            "split": "reference", "duplicate_group_id": "group-1", "rag_id": "ticket-1",
        }
        self._assert_reference_loader_rejects(row, "embeds a Ticket Type")

    def test_reference_loader_never_accepts_evaluation_rows(self):
        row = {
            "Ticket ID": "1", "Product Purchased": "Example", "Ticket Type": "Refund request",
            "Ticket Subject": "Refund delay", "Ticket Description": "Late refund",
            "Ticket Status": "Open", "Ticket Priority": "High", "Ticket Channel": "Email",
            "Resolution": "", "resolution_quality": "missing", "rag_text": "Subject: Refund delay",
            "split": "evaluation", "duplicate_group_id": "group-1", "rag_id": "ticket-1",
        }
        self._assert_reference_loader_rejects(row, "non-reference row")

    def _assert_reference_loader_rejects(self, row, error):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with patch.object(ingestion, "REFERENCE_CSV", path), patch.object(
                ingestion, "EXPECTED_ROWS", 1
            ), self.assertRaisesRegex(ValueError, error):
                ingestion.load_reference_rows()


if __name__ == "__main__":
    unittest.main()
