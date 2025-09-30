import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.storeData.FetchData import TranscriptParser

class TestTranscriptParser(unittest.TestCase):
    def setUp(self):
        # Use a sample HTML file for testing
        self.html_path = os.path.join(os.path.dirname(__file__), '..\\src\\RagBasedStockAnalyser\\data\\html\\AAPL_2025_q3\\2025_q3.html')
        self.parser = TranscriptParser(url=None)

    def test_parse_html_file(self):
        self.assertTrue(os.path.exists(self.html_path), f"Test HTML file does not exist: {self.html_path}")
        self.parser.parse_html_file(self.html_path)
        self.assertIsNotNone(self.parser.soup)
        self.assertTrue(len(self.parser.soup.get_text()) > 0)
        # Optionally, test parse_blocks if the HTML is a real transcript
        self.parser.parse_blocks()
        self.assertIsInstance(self.parser.blocks, list)
        print(f"Parsed {len(self.parser.blocks)} blocks from HTML.")

if __name__ == "__main__":
    unittest.main()
