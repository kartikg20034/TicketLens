import math
import unittest

from backend.local_embeddings import (
    EMBEDDING_DIMENSIONS,
    MODEL_ID,
    encode_texts,
    is_reusable_embedding,
    retrieval_function_uses_model,
)


class DummyModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.kwargs = None

    def encode(self, texts, **kwargs):
        self.kwargs = kwargs
        return self.vectors


class LocalEmbeddingTests(unittest.TestCase):
    def test_encode_requests_normalized_numpy_vectors_and_checks_them(self):
        values = [1.0 / math.sqrt(EMBEDDING_DIMENSIONS)] * EMBEDDING_DIMENSIONS
        model = DummyModel([values])
        result = encode_texts(model, ["sample ticket"], batch_size=7)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), EMBEDDING_DIMENSIONS)
        self.assertTrue(model.kwargs["normalize_embeddings"])
        self.assertTrue(model.kwargs["convert_to_numpy"])
        self.assertEqual(model.kwargs["batch_size"], 7)

    def test_rejects_wrong_dimensions(self):
        with self.assertRaisesRegex(ValueError, "expected 768"):
            encode_texts(DummyModel([[1.0, 0.0]]), ["sample ticket"])

    def test_only_current_bge_vectors_are_reusable(self):
        self.assertTrue(is_reusable_embedding(MODEL_ID, 768))
        self.assertFalse(is_reusable_embedding("gemini-embedding-2", 768))
        self.assertFalse(is_reusable_embedding(MODEL_ID, 3072))

    def test_retrieval_function_requires_model_filter(self):
        self.assertTrue(
            retrieval_function_uses_model(
                "WHERE t.split = 'reference' AND e.embedding_model = 'BAAI/bge-base-en-v1.5'"
            )
        )
        self.assertFalse(
            retrieval_function_uses_model("WHERE split = 'reference' -- BAAI/bge-base-en-v1.5")
        )
        self.assertFalse(retrieval_function_uses_model(None))


if __name__ == "__main__":
    unittest.main()
