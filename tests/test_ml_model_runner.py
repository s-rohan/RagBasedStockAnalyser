import unittest
from unittest.mock import patch
import os, sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


class TestMlModelRunner(unittest.TestCase):
    def test_run_model_and_evaluate_with_real_csv(self):
        # Use the existing processed CSV in repo and patch S3Store to copy it
        repo_csv = os.path.join(os.path.dirname(__file__), '..\\src\\RagBasedStockAnalyser\\data\\earningcall\\processed_earnings_NFLX_AAPL_MSFT_GOOGL_AMZN_TSLA.csv')
        # Ensure file exists
        self.assertTrue(os.path.isfile(repo_csv), f"Expected CSV at {repo_csv}")

        # Import the module under test and the XGBoostEarningModel module to patch S3Store
        from RagBasedStockAnalyser.equity.pipeline import XGBoostEarningModel as xmod
        from RagBasedStockAnalyser.equity.pipeline.MlModelRunner import RunModelAndEvaluate

        # Patch S3Store used by XGBoostEarningModel so download_File copies the repo CSV to the requested path
        class FakeS3Store:
            def __init__(self, bucket_name=None):
                self.bucket_name = bucket_name

            def download_File(self, object_name: str, file_path: str):
                # The model expects dataFilePath like processed_earnings_<tickers>.csv
                # If object_name matches the basename of repo_csv, copy it
                if object_name == os.path.basename(repo_csv):
                    shutil.copyfile(repo_csv, file_path)
                    return True
                # Also support when requested name matches expected pattern for the default ticker list
                expected_name = "processed_earnings_AAPL_MSFT_GOOGL_AMZN_TSLA.csv"
                if object_name == expected_name:
                    shutil.copyfile(repo_csv, file_path)
                    return True
                return False

        with patch.object(xmod, 'S3Store', FakeS3Store) :
            # Call runner with the tickers that match the CSV naming
            metrics=["Revenues_growth", "NetIncomeLoss_growth",'ResearchAndDevelopmentExpense_growth']
            preds = RunModelAndEvaluate(ticker=["AAPL","MSFT","GOOGL","AMZN","TSLA"],targetMetrics=metrics, s3=FakeS3Store(bucket_name="earnings")  )

        # Validate output structure
        self.assertIsInstance(preds, dict)
        print(f"scores :{preds}")
        for metric in metrics:
            self.assertIn(metric, preds)

        # Predictions should be iterable (numpy array or list)
        for arr in preds.values():
            self.assertTrue(arr>0.90 or arr==0)


if __name__ == '__main__':
    unittest.main()

