# Configure logging

from RagBasedStockAnalyser.equity.fetch.EmbeddingOrganizer import EmbeddingOrganizer
import glob
import os,json
from  RagBasedStockAnalyser.equity.storeData.ReadStore import  ReadStore
from RagBasedStockAnalyser.common.logging_config import setup_logging
logger = setup_logging()
class EarningCallPipleLine:
    def __init__(self):
        self.readStore=ReadStore()

        logger.info("EarningCallPipleLine initialized.")
    def read_and_store_from_earningcall(self,ticker: str, year: str, quater: str):
        """
        Reads files from data/earninngcall based on ticker, year, quater.
        If any value is '*', reads all matching files.
        Uses EmbeddingOrganizer.store to write to Redis.
        """

        pattern = f"{ticker}_{year}_{quater}"
        folder = "earningcall"
        files =[
                f for f in self.readStore.list_files_in_folder(folder)
                 if pattern in os.path.basename(f)
                ]
        logger.info(f"Files matched for pattern {pattern}: {files}")
        
        if not files:
            logger.warning(f"No files found for pattern: {pattern}")
            return False
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            
            blocks = self.readStore.read_file(file_path,processor=lambda x: json.loads(x))
            # Extract actual values from filename for store method
            fname = os.path.basename(file_path)
            parts = fname.replace('.jsonl', '').split('_')
            t, y, q = parts[0], parts[1], parts[2].split('.')[0]
            org = EmbeddingOrganizer()
            org.store(blocks, t, int(y), q)
            logger.info(f"Stored blocks from {file_path} with  {t}, {int(y)}, {q} to Redis.")
            return True