import unittest
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.pipeline.EarningCallPipeLine import EarningCallPipleLine

class TestEarningCallPipeLine(unittest.TestCase):
    def setUp(self):
        self.pipeline = EarningCallPipleLine()

    def test_read_and_store_from_earningcall(self):
        # This will attempt to read data/earninngcall/AAPL_2025_q3.jsonl and store to Redis
        bool=self.pipeline.read_and_store_from_earningcall('AAPL', '2025', 'q3')
        # No assertion here: success is no exception and log output
        # Optionally, you could add checks for side effects if Redis is mocked
        assert bool
        print("read_and_store_from_earningcall executed for AAPL, 2025, q3")
if __name__ == "__main__":
    unittest.main()
