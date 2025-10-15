from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
from RagBasedStockAnalyser.common.logging_config import setup_logging
logger=setup_logging()
# Connect to DocumentDB Local
class DocumentManager:
    def __init__(self,url:str=None,username:str=None,password:str=None,**kwargs):
        if url is None:
            url= os.getenv("DOCUMENT_DB_URL")
        if password is None:
            password= os.getenv("DOCUMENT_DB_password")
        if username is None :
            username= os.getenv("DOCUMENT_DB_username")
        if "<username>" in url:
            url=url.replace("<username>",username)
        if "<password>" in url:
            url=url.replace("<password",password)
        self.url=url
        self._client = MongoClient(
            url,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000
        )

    def storeData(self,submission:dict,repo:str)->bool:
    # Access database and collection
        db = self._client["edgar_data"]
        collection = db[repo]

        if(isinstance(submission,dict)):
        # Insert into collection
            result = collection.insert_one(submission)
            logger.info(f"Inserted document with ID: {result.inserted_id}")
        else:
            result = collection.insert_one(submission.dict())
            logger.info(f"Inserted document with ID: {result.inserted_id}")

