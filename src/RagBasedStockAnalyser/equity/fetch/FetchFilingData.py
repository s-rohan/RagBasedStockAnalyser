import requests
from datetime import datetime,timezone
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import json,time
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List
from RagBasedStockAnalyser.common.logging_config import setup_logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from RagBasedStockAnalyser.equity.storeData.db.DocumentManager import DocumentManager
from collections import defaultdict
from playwright.sync_api import sync_playwright
load_dotenv()
logger = setup_logging(logger_name=__name__)
class TickerEntry(BaseModel):
    cik_str: str
    ticker: str
    title: str
    @field_validator("cik_str", mode="before")
    @classmethod
    def coerce_cik(cls, v):
        return str(v)

class FilingChunk(BaseModel):
    chunk_id: str
    accession_number: str
    CIK_number: str
    filing_date: datetime
    fiscal_year: int
    fiscal_quarter: str
    content: str
    source_url: str

class FilingMetadata(BaseModel):
    CIK_number: str
    accession_number: str
    filing_date: datetime
    form_type: str
    fiscal_year: int
    fiscal_quarter: str
    source_url: str
    chunk_count: int
    
class FetchFilingData():
    def __init__(self,**kargs):
        sec_filing_data_json =kargs.get("SEC_FILING_DATA")
        self._ticker_cif_mapping=self.load_ticker_data(sec_filing_data_json)
        self.doc_db=DocumentManager()
     
    def ticker_cif_mapping(self,ticker:str)->str:
       return self._ticker_cif_mapping.get(ticker).cik_str.zfill(10)
   
    def load_ticker_data(self,filepath: str) -> dict[str:TickerEntry]:
        """Loads and validates SEC ticker data from a local JSON file."""
        raw_data=None
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        entries = list(raw_data.values())
        validated_dict = {
            entry["ticker"]: TickerEntry(**entry)
            for entry in entries
            if "ticker" in entry and "cik_str" in entry and "title" in entry
        }
        return validated_dict
    @classmethod
    def get_quarter_date_range(cls,year: int, quarter: int):
        """Returns start and end dates for a given year and quarter."""
        start_month = 3 * (quarter - 1) + 1
        end_month = start_month + 2
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month + 1, 1) if end_month < 12 else datetime(year, 12, 31)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    
   
    def fetch_submissions(self, cik: str,ticker:str, form_type: str = "10-Q",past_years:int=5) -> bool:
        logger.info(f"Fetching submissions for CIK: {cik}, Ticker: {ticker}, Form Type: {form_type}")
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        logger.info(f"Fetching URL: {url}")

        data = self.get_sec_content(url).json()
        fy_metrics = self.extract_fy_metrics(data,cik=cik)
        ret=self.doc_db.store_company_facts_data(df=fy_metrics, repo="company_facts")
        return ret
    
    def print_fy_metrics(self,fy_data: pd.DataFrame):
        logger.info(f"{'Fiscal Year':<12} {'Net Income':>15} {'Assets':>15} {'Liabilities':>15}")
        for _, row in fy_data.sort_values(by="fy", ascending=False).iterrows():
            fy = row["fy"]
            ni = row.get("NetIncomeLoss", 0)
            assets = row.get("Assets", 0)
            liabilities = row.get("Liabilities", 0)
            logger.info(f"{fy:<12} {ni:>15,.2f} {assets:>15,.2f} {liabilities:>15,.2f}")
        
    def derive_fiscal_tags(self,filing_date: datetime) -> tuple:
        ''' Takes foling date and returns the year and quarter'''
        month = filing_date.month
        year = filing_date.year
        quarter = (
            "Q1" if month <= 3 else
            "Q2" if month <= 6 else
            "Q3" if month <= 9 else
            "Q4"
        )
        return year, quarter

    def chunk_html_content(self,html: str, max_chars: int = 50000) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        paragraphs = text.split("\n")
        
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += para + "\n"
            else:
                chunks.append(current.strip())
                current = para + "\n"
        if current:
            chunks.append(current.strip())
        return chunks
        

    def extract_fy_metrics(self,data: dict,cik:str, concepts: list = None)->pd.DataFrame:
        if concepts is None:
            concepts = [
    # Profit & Loss
    "Revenues",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "ShareBasedCompensation",
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsOfDividends",

    # Balance Sheet
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "LongTermDebt",
    "ShortTermDebt",
    "InventoryNet",
    "AccountsReceivableNetCurrent",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",

    # Cash Flow
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "GoodwillImpairmentLoss"
]

        rows=[]

        for concept in concepts:
            try:
                entries = data["facts"]["us-gaap"][concept]["units"]["USD"]
            except KeyError:
                continue

            for entry in entries:
                if "end" not in entry or "val" not in entry:
                    continue

                try:
                    fy = datetime.strptime(entry["end"], "%Y-%m-%d").year
                    fp = entry.get("fp", "FY")  # Default to full year if missing
                    row = {
                    "cik": cik,
                    "concept": concept,
                    "val": entry["val"],
                    "end": entry["end"],
                    "fy": entry.get("fy",fy),
                    "fp": fp,
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "filed": entry.get("filed"),
                    "accn": entry.get("accn")
                }
                    rows.append(row)
                except ValueError:
                    logger.info(f"Skipping entry due to ValueError: {entry}")
                    continue
            df = pd.DataFrame(rows)
            df_wide = df.pivot_table(
            index=["cik", "fy", "fp", "frame"],
            columns="concept",
            values="val",
            aggfunc="max"
        ).reset_index()

        return df_wide
    def get_sec_content(self,url):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[403, 404, 429])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        headers = {
            "User-Agent": "Rohan Sharma techroitleads@gmail.com",
            "Accept-Encoding": "gzip, deflate"
        }

        response = session.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch submissions for CIK: {url} → {response.status_code}")
            return None
        return response
