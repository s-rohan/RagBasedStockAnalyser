import unittest
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.model.EquityAgent import EquityAgent

class TestEquityAgent(unittest.TestCase):
    def setUp(self):
        self.agent = EquityAgent()
    
    def test_query_returns_string(self):
        # This test assumes the agent can run end-to-end with default setup
        query = "How did Apple perform in Q3 2025?"
        try:
            result = self.agent.query(query)
            self.assertIsInstance(result["answer"], str)

            print(f"Agent response: {result}")
        except Exception as e:
            self.fail(f"EquityAgent.query raised an exception: {e}")

    def test_merge_retrived_docs(self):
        # Simulate state with both transcript and report docs
        state = {
            "transcript_retrieved_documents": ["Transcript doc 1", "Transcript doc 2"],
            "report_retrieved_documents": ["Report doc 1"]
        }
        merged = self.agent.merge_retrived_docs(state)
        self.assertIn("retrieved_documents", merged)
        self.assertEqual(len(merged["retrieved_documents"]), 3)

    def test_route_retrievers(self):
        state = {"invoke_transcript": True, "invoke_report": True}
        paths = self.agent.route_retrievers(state)
        self.assertIn("retrieve_transcript", paths)
        self.assertIn("retrieve_report", paths)
    
    def test_error_handling(self):
        # Simulate a query that should not match any retriever
        query = "Show me the financials for XYZ Q1 2020"
        try:
            result = self.agent.query(query)
            self.assertIsInstance(result["answer"], str)
        except Exception as e:
            self.fail(f"EquityAgent.query raised an exception: {e.with_traceback()}")

if __name__ == "__main__":
    unittest.main()
