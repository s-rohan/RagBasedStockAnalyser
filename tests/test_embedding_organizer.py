import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.fetch.EmbeddingOrganizer import EmbeddingOrganizer

class TestEmbeddingOrganizer(unittest.TestCase):
    def setUp(self):
        self.organizer = EmbeddingOrganizer()

    def test_preprocess(self):
        text = "Hello, this is a test transcript!"
        result = self.organizer.preprocess(text)
        self.assertIn("hello", result)
        self.assertNotIn(",", result)

    def test_getSemanticChunks(self):
        text = "This is a test. Another sentence."
        chunks = self.organizer.getSemanticChunks(text, .95)
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1)

    def test_storeLexicalData(self):
        docs = ["test document one", "another test document"]
        result = self.organizer.storeLexicalData(docs, url="https://www.investing.com/news/transcripts/earnings-call-transcript-apple-beats-q3-2025-forecasts-stock-dips-93CH-4164767", quater="Q1", year=2025, ticker="AAPL")
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
