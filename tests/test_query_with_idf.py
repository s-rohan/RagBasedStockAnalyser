import unittest
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.fetch.QueryWithIDF import QueryWithIDF
from RagBasedStockAnalyser.redis.VectorStore import VectorStore

class TestQueryWithIDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector_store = VectorStore()
        cls.query_with_idf = QueryWithIDF(cls.vector_store)
        cls.queries = [
            "How did Apple’s iPhone segment perform in Q3?",
            "What were the key drivers of revenue growth?",
            "Did Apple mention any challenges in the wearables category?"
        ]

    def test_query_iphone_segment(self):
        results = self.query_with_idf.fetch_and_lexical(self.queries[0], top_k=11)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        print(f"Results for query 1: {results}")
    
    def test_query_revenue_growth(self):
        results = self.query_with_idf.fetch_and_lexical(self.queries[1], top_k=11)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        print(f"Results for query 2: {results}")

    def test_query_wearables_challenges(self):
        results = self.query_with_idf.fetch_and_lexical(self.queries[2], top_k=11)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        print(f"Results for query 3: {results}")
       

if __name__ == "__main__":
    unittest.main()
