import requests
from datetime import datetime,timezone
from typing import Optional
import time
from dotenv import load_dotenv
import json
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List
from RagBasedStockAnalyser.common.logging_config import setup_logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from RagBasedStockAnalyser.equity.storeData.db.DocumentManager import DocumentManager
load_dotenv()
logger = setup_logging()
class TickerEntry(BaseModel):
    cik_str: str
    ticker: str
    title: str
    @field_validator("cik_str", mode="before")
    @classmethod
    def coerce_cik(cls, v):
        return str(v)

class DocumentMetadata(BaseModel):
    form_type: str
    filing_date: str
    accession_number: str
    source_url: str
    content: Optional[str] = None  # Optional parsed content

class SECSubmission(BaseModel):
    CIK_number: str
    date_fetched: datetime
    document: List[DocumentMetadata]
    
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

    def get_quarter_date_range(year: int, quarter: int):
        """Returns start and end dates for a given year and quarter."""
        start_month = 3 * (quarter - 1) + 1
        end_month = start_month + 2
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month + 1, 1) if end_month < 12 else datetime(year, 12, 31)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')




        return None

    def fetch_submissions(self, cik: str, form_type: str = "10-Q") -> bool:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        data = self.get_sec_content(url).json()
        filings = data.get("filings", {}).get("recent", {})
        results = []

        for i in range(len(filings.get("accessionNumber", []))):
            if filings["form"][i] != form_type:
                continue

            accession = filings["accessionNumber"][i]
            filing_date = filings["filingDate"][i]
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.html"
            content = self.fetch_filing_content(cik, accession)


            document=DocumentMetadata(
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=accession,
                    source_url=doc_url,
                    content=content
                )
            results.append(document)

        submission =SECSubmission(
                CIK_number=cik,
                date_fetched=datetime.now(timezone.utc),
                document=results

            )
            
        is_stored=self.doc_db.storeData(submission=submission,repo="sec_filing")
        logger.info("Data stored for cif {cik}")
        return is_stored
    


    def fetch_filing_content(self, cik: str, accession: str) -> Optional[str]:
        base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.html"
        index_response =self.get_sec_content(base_url)


        soup = BeautifulSoup(index_response.text, "html.parser")
        table = soup.find("table", class_="tableFile")
        if not table:
            print("Could not find document table.")
            return None

        # Find the first .htm or .html document link
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            doc_link = cols[2].find("a")
            if doc_link and doc_link["href"].endswith((".htm", ".html")):
                doc_url = f"https://www.sec.gov{doc_link['href']}"
                doc_response =self.get_sec_content(doc_url)

                if doc_response.status_code == 200:
                    return doc_response.text
                else:
                    print(f"Failed to fetch document: {doc_url}")
                    return None

        print("No HTML document found.")
        return None



    def fetch_quarterly_10q(self, cik: str, year: int, quarter: int):
        """Fetches 10-Q filings metadata for a given CIK and quarter."""
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
        data = self.get_sec_content(url).json()
        start_date, end_date = self.get_quarter_date_range(year, quarter)
        filings = data.get("filings", {}).get("recent", {})
        results = []

        for i, form in enumerate(filings.get("form", [])):
            if form == "10-Q":
                filed_date = filings["filingDate"][i]
                if start_date <= filed_date <= end_date:
                    results.append({
                        "date": filed_date,
                        "accession": filings["accessionNumber"][i],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{filings['accessionNumber'][i].replace('-', '')}/{filings['primaryDocument'][i]}"
                    })

        return results
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
