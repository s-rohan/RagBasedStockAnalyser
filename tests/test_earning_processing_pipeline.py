import unittest
import importlib
import os
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.fetch.FetchFilingData import FetchFilingData
from RagBasedStockAnalyser.equity.storeData.db.DocumentManager import DocumentManager
from RagBasedStockAnalyser.equity.pipeline.EarningProcessingPipeline import EarningProcessingPipeline

# ensure src is on path when running tests in isolation
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))




class TestEarningProcessingPipeline(unittest.TestCase):
    def setUp(self):
        self.sec_filing_data = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "RagBasedStockAnalyser",
            "data",
            "tickers",
            "company_tickers.json"
        )
        # import the module so we can patch its attributes
        self.pipeline = EarningProcessingPipeline(sec_filing_data=self.sec_filing_data)
        # patch the dependencies on the module

    def test_process_earnings_calls_db(self):
       data= self.pipeline.process_earnings()
       self.assertTrue( data is not None)
       self.assertTrue(data.size > 0)


if __name__ == '__main__':
    unittest.main()
