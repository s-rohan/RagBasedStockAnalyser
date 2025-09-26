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

if __name__ == "__main__":
    unittest.main()
