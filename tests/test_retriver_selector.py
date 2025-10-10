import unittest
import asyncio
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.model.RetriverSelector import RetriverSelector

class TestRetriverSelector(unittest.TestCase):
    def setUp(self):
        self.selector = RetriverSelector()
    
    def test_call_name(self):
        query="How did Apple and Tesla do in 2025"
        response=asyncio.run(self.selector.call(query=query))
        self.assertTrue("transcript" in response["response"])
        self.assertTrue("report" in response["response"])
    
    
    def test_call(self):
        query="How did AAPL do in 2025"
        response=asyncio.run(self.selector.call(query=query))
        self.assertTrue("transcript" in response["response"])
    
        
    def test_call_exception(self):
        query="Show me the financials for XYZ Q1 2020"
        response=asyncio.run(self.selector.call(query=query))
        self.assertTrue("No retriever found." in response["response"])
      



if __name__ == "__main__":
    unittest.main()
