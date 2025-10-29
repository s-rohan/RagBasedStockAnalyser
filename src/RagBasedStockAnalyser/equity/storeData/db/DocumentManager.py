from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Callable, List, Union, Dict, Any
import atexit
load_dotenv()
from RagBasedStockAnalyser.common.logging_config import setup_logging
logger=setup_logging(logger_name=__name__)
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
            url=url.replace("<password>",password)
        self.db_name=kwargs.get("db_name","edgar_data")
        self.url=url
        self._client = MongoConnector(
            uri=url,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,
            db_name=self.db_name
        )
    
    def store_company_facts_data(self,df: pd.DataFrame, repo:str)->True:
        collection = self._client.get_collection(repo)
        records = df.to_dict(orient="records")
        for record in records:
            # Optional: add provenance tag
            record["source"] = "SEC CompanyFacts API"
            collection.update_one(
                {"cik": record["cik"], "fy": record["fy"], "fp": record["fp"], "frame": record["frame"]},
                {"$set": record},
                upsert=True
            )
        return True

    def storeData(self,submission:dict,repo:str)->bool:
        """
        Stores a document in the specified repository (collection) within the MongoDB database.

        Args:
            submission: The document to store, either as a dict or a Pydantic model instance.
            repo: The name of the collection to store the document in."""
    # Access database and collection

        collection = self._client.get_collection(repo)
        try:
            if(isinstance(submission,dict)):
            # Insert into collection
                result = collection.insert_one(submission)
                logger.info(f"Inserted in {repo} document with ID: {result.inserted_id}")
            else:
                result = collection.insert_one(submission.dict())
                logger.info(f"Inserted in {repo} document with ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.exception(f"Error inserting document into {repo}: {e}")
            return False
    
    def query_and_parse(
    self,
    collection_name: str,
    query: Union[Dict, List[Dict]],
    parser: Callable[[dict], Any] = None,
    limit: int = 100,
    use_aggregation: bool = False
) -> List[Any]:
        """
        Connects to MongoDB, runs a find or aggregate query on the specified collection,
        and applies a parser function to each result.

        Args:
            collection_name: Name of the collection
            query: MongoDB query dict (for find) or list of pipeline stages (for aggregate)
            parser: Function to parse each document
            limit: Max number of documents to return (0 = no limit)
            use_aggregation: If True, runs aggregation pipeline instead of find

        Returns:
            List of parsed documents
        """
        collection = self._client.get_collection(collection_name)

        if use_aggregation:
            pipeline = query if isinstance(query, list) else [query]
            if limit > 0:
                pipeline.append({"$limit": limit})
            cursor = collection.aggregate(pipeline)
        else:
            cursor = collection.find(query)
            if limit > 0:
                cursor = cursor.limit(limit)

        if parser:
            return [parser(doc) for doc in cursor]
        else:
            return list(cursor)

from threading import Lock

class MongoConnector:
    _instance = None
    _lock = Lock()

    def __new__(cls, uri: str, db_name: str, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(uri, db_name, **kwargs)
                    atexit.register(cls._instance._cleanup)
        return cls._instance

    def _init(self, uri: str, db_name: str, **kwargs):
        self.client = MongoClient(uri, **kwargs)
        self.db = self.client[db_name]

    def get_collection(self, name: str):
        return self.db[name]

    def _cleanup(self):
        try:
            if  self.client:
                self.client.close()
                logger.info("MongoClient closed on GC.")
        except Exception as e:
            logger.exception(f"Error closing MongoClient: {e}")


