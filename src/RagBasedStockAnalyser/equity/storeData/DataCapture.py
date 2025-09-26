import sys
import os

# Dynamically add src/ to sys.path
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '../../../src'))
sys.path.insert(0, src_path)

from RagBasedStockAnalyser.equity.storeData.FetchData import TranscriptParser
from RagBasedStockAnalyser.equity.storeData.StoreToFile import DataStore


def capture_and_store_transcript():
    url = 'https://www.investing.com/news/transcripts/earnings-call-transcript-apple-beats-q3-2025-forecasts-stock-dips-93CH-4164767'
    html_path = os.path.join(os.path.dirname(__file__), '..\\..\\data\\html\\AAPL_2025_q3\\2025_q3.html')
    trans = TranscriptParser(url=None)
    trans.parse_html_file(html_path)
    trans.parse_blocks()
    blocks = trans.blocks
    # Store blocks to file under data/earninngcall/APPL_2025_q3.jsonl
    ds=DataStore()
    savePath=ds.store_json_to_file(blocks, 'earninngcall', 'APPL_2025_q3')
    

if __name__ == "__main__":
    capture_and_store_transcript()
