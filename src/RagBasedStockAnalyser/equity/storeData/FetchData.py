#Fetch Data 
from bs4 import BeautifulSoup
import time
import requests
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from selenium.common.exceptions import WebDriverException
from requests.exceptions import Timeout, ReadTimeout, ConnectTimeout
import socket
import time
import urllib3


class TranscriptParser:
    def __init__(self, url):
        self.url = url
        self.soup = None
        self.blocks = []
        self.rawtext=None

    def fetch_earnings_data(self,ticker, date_range=None):
        if date_range is None:
            today = pd.to_datetime('today')
            date_range = (today, today + pd.DateOffset(months=3)) # Fetch 3 months of data from today
        start_date = date_range[0].strftime("%Y-%m-%d")
        end_date = date_range[1].strftime("%Y-%m-%d")
        url = f"https://finance.yahoo.com/calendar/earnings?symbol={ticker}&from={start_date}&to={end_date}"
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)  # Wait for JS to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        # Parse earnings rows
        rows = soup.select("table tbody tr")
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                data.append({
                    "Date": cols[1].text.strip(),
                    "Time": cols[2].text.strip(),
                    "EPS Estimate": cols[3].text.strip(),
                    "Reported EPS": cols[4].text.strip(),
                    "Surprise (%)": cols[5].text.strip()
                })

        
        return data

    def safe_get(self,driver, url, retries=2, delay=5):
        for attempt in range(retries):
            try:
                driver.get(url)
                return True
            except(ConnectionResetError, socket.error, WebDriverException,Timeout, ReadTimeout,ConnectTimeout) as e:
                print(f"[Attempt {attempt+1}] Connection reset: {e}")
                time.sleep(delay)
                return True
        return False  # All attempts failed

    def fetch(self):
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.google.com/"
        }
        response = requests.get(self.url, headers=headers)
        self.soup = BeautifulSoup(response.text, "html.parser")
        if len(self.soup.get_text())<100:
            options = uc.ChromeOptions()
            
            ua = UserAgent()
            options.add_argument(f"user-agent={ua.chrome}")          
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--start-maximized")
            options.add_argument("--lang=en-US,en")
            options.add_argument("--disable-notifications")
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0.0.0 Safari/537.36")

        driver = uc.Chrome(options=options, headless=False, use_subprocess=True)
        if self.safe_get(driver, self.url):
            self.soup = BeautifulSoup(driver.page_source, "html.parser")
            print("Page loaded successfully.")
    # Proceed with parsing
        else:
            print("Failed to load page after retries. Proceeding with fallback logic.")
            
        driver.quit()

    def parse_blocks(self):
        self.raw_text = self.soup.get_text(separator="\n")
        self.raw_text = self.raw_text.replace("\\n:", ":").replace("\n:", ":")
        lines = self.raw_text.split("\n")
        current_speaker = None
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect speaker change
            match = re.search(r"^(.*?)(, CEO|, CFO|, Director of Investor Relations).*?:", line)
            if match:
                if current_speaker and current_block:
                    self.blocks.append({
                        "speaker": current_speaker,
                        "text": "\n".join(current_block)
                    })
                current_speaker = match.group(1)
                current_block = [line]
            else:
                current_block.append(line)

        # Add final block
        if current_speaker and current_block:
            self.blocks.append({
                "speaker": current_speaker,
                "text": "\n".join(current_block)
            })

    def get_blocks(self):
        return self.blocks
    
    def parse_html_file(self, html_path):
        """
        Parse a local HTML file and set self.soup for further processing.
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        self.soup = BeautifulSoup(html_content, "html.parser")
