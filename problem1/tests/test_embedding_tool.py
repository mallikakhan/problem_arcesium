import unittest
from ..tools.embedding_tool import EmbeddingTool

class TestEmbeddingTool(unittest.TestCase):
    def setUp(self):
        self.embedding_tool = EmbeddingTool()

    def test_embedding_generation(self):
        texts = ["Test text for embedding generation."]
        result = self.embedding_tool({"texts": texts})
        self.assertIn("embeddings", result)
        self.assertEqual(len(result["embeddings"]), 1)
        self.assertEqual(len(result["embeddings"][0]), 384)  # Assuming the embedding dimension is 384

if __name__ == '__main__':
    unittest.main()
