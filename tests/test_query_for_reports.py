
import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.fetch.QueryWithIDF import ReportQueryWithIDF
from RagBasedStockAnalyser.redis.VectorStore import VectorStore

class TestReportQueryWithIDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector_store = VectorStore()
        cls.query_report_with_idf = ReportQueryWithIDF(cls.vector_store)
        cls.query = "How did Tesla perform in Q2?"

    def test_query_tesla_q2(self):
        results = self.query_report_with_idf.fetch_and_lexical(self.query, top_k=10)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        print(f"Results for Tesla Q2: {results}")

if __name__ == "__main__":
    unittest.main()