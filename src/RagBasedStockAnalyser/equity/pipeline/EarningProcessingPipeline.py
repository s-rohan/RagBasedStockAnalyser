import os, sys
from RagBasedStockAnalyser.equity.fetch.FetchFilingData import FetchFilingData
from RagBasedStockAnalyser.equity.storeData.db.DocumentManager import DocumentManager
from RagBasedStockAnalyser.equity.storeData.S3Store import S3Store   
import pandas as pd
from RagBasedStockAnalyser.common.logging_config import setup_logging
from typing import Tuple
logger = setup_logging(logger_name=__name__)
class EarningProcessingPipeline:
    def __init__(self, sec_filing_data: str, ticker=None):
        """Initialize the earnings processing pipeline.

        Args:
            sec_filing_data: Path to the JSON file containing SEC ticker metadata.
            ticker: Optional list of tickers to process. If omitted, a default list is used.
        """
        self.ticker = ticker
        self.fetcher = FetchFilingData(SEC_FILING_DATA=sec_filing_data)
        self.db = DocumentManager()
        self.s3=S3Store(bucket_name="earnings") 

    def process_earnings(self):
        """Primary pipeline entry point.

        Fetches company facts data for configured tickers, aggregates selected
        financial concepts, computes lagged features and fills missing values.

        Returns:
            pd.DataFrame: Processed financial dataset with lagged and growth features.
        """
        cik = [self.fetcher.ticker_cif_mapping(ticker=ticker) for ticker in self.ticker]
        concepts = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities", "ResearchAndDevelopmentExpense"]
        data = self._getDataFromDB(cik, concepts)
        data = self._drop_empty_concept_rows(data, concepts)
        data,derived_concepts = self._add_lagged_features(data, concepts)
        data = self._complete_incomplete_lags(data, derived_concepts)
        fileName="processed_earnings_"+"_".join(self.ticker)+".csv"
        data.to_csv(fileName,index=False)
        loaded=self.s3.upload_file(file_path=fileName,object_name=fileName)
        logger.info(f"Uploaded processed earnings {fileName}to S3: {loaded}")
        return data
    

    def _getDataFromDB(self, cik: list, concepts: list) -> pd.DataFrame:
        """Query the document DB for company facts and aggregate requested concepts.

        Args:
            cik: list of zero-padded CIK strings to include.
            concepts: list of financial concepts to aggregate (e.g., Revenues).

        Returns:
            pd.DataFrame: Aggregated results from the database.
        """
        match = {'fp': {'$ne': 'FY'}, 'cik': {'$in': cik}}
        group_fields=["cik", "frame"]
        group_id = {field: f"${field}" for field in group_fields}
        group_stage = {
                "$group": {
                    "_id": group_id,
                    "fy": {"$first": "$fy"},
                    "fp": {"$first": "$fp"},
                }
            }
        for concept in concepts:
            group_stage["$group"][concept] = {"$sum": f"${concept}"}
            query = [
                {"$match": match},
                group_stage,
                self._build_project_stage(concepts=concepts,include_meta=True)
            ]
        data = pd.DataFrame(self.db.query_and_parse(query=query,collection_name="company_facts",use_aggregation=True))
            
        logger.info(f"Earnings processing completed.")
        return data
    
    def _build_project_stage(self, concepts: list, include_meta: bool = True) -> dict:
        """Create a MongoDB aggregation $project stage for the requested concepts.

        Args:
            concepts: list of financial concepts.
            include_meta: whether to include metadata columns (cik, frame, fy, fp).

        Returns:
            dict: Aggregation $project stage.
        """
        project = {"_id": 0}
        if include_meta:
            project.update({
                "cik": "$_id.cik",
                "frame": "$_id.frame",
                "fy": 1,
                "fp": 1
            })
        for concept in concepts:
            project[concept] = f"${concept}"
        return {"$project": project}
    
    def _add_lagged_features(self, df: pd.DataFrame, concepts: list) -> Tuple[pd.DataFrame,list]:
        """Add lagged, next, growth, and acceleration features for each concept.

        Adds the following columns per concept:
            - {concept}_t-1 : previous value for the same cik
            - {concept}_next : next value for the same cik
            - {concept}_growth : growth from previous to current
            - {concept}_growth_next : growth from current to next
            - {concept}_accel : acceleration of growth

        Returns the DataFrame with new columns appended.
        """
        df = df.sort_values(["cik", "frame"])
        derived_cols = []
        for concept in concepts:
            df[f"{concept}_t-1"] = df.groupby("cik")[concept].shift(1)
            df[f"{concept}_next"] = df.groupby("cik")[concept].shift(-1)
            df[f"{concept}_growth"] = df[concept] / df[f"{concept}_t-1"] - 1
            df[f"{concept}_growth_next"] = df[f"{concept}_next"] / df[concept] - 1
            df[f"{concept}_accel"] = df[f"{concept}_growth_next"] - df[f"{concept}_growth"]
            derived_cols.extend([
                f"{concept}_t-1",
                f"{concept}_next",
                f"{concept}_growth",
                f"{concept}_growth_next",
                f"{concept}_accel"
            ])
        return df,derived_cols
    
    def _complete_incomplete_lags(self, df: pd.DataFrame, derived_concepts: list) -> pd.DataFrame:
        """Fill missing lagged features using forward/backward fill.

        This mutates the DataFrame in-place and returns it for convenience.
        """

        df[derived_concepts] = df[derived_concepts].ffill()
        df[derived_concepts] = df[derived_concepts].bfill()
        return df
    
    def _drop_empty_concept_rows(self,df: pd.DataFrame, concepts: list) -> pd.DataFrame:
         df.dropna(subset=concepts, how="all",inplace=True)
         return df




            

