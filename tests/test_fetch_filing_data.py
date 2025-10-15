import unittest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from RagBasedStockAnalyser.equity.fetch.FetchFilingData import FetchFilingData

class TestFetchFilingData(unittest.TestCase):
    def setUp(self):
        self.ticker = "AAPL"
        self.year = 2023
        self.quarter = 2
        self.sec_filing_data = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "RagBasedStockAnalyser",
            "data",
            "tickers",
            "company_tickers.json"
        )
        self.fetcher = FetchFilingData(SEC_FILING_DATA=self.sec_filing_data)

    def test_ticker_cif_mapping(self):
        cik = self.fetcher.ticker_cif_mapping(ticker=self.ticker)
        self.assertIsInstance(cik, str)
        self.assertEqual(len(cik), 10)

    def test_fetch_submissions(self):
        cik = self.fetcher.ticker_cif_mapping(ticker=self.ticker)
        if cik:
            result = self.fetcher.fetch_submissions(cik=cik)
            # The function stores data and logs, but doesn't return a value
            # Just check no exception and log output
            self.assertIsNone(result)
        else:
            self.fail("CIK not found for ticker.")

if __name__ == "__main__":
    unittest.main()
