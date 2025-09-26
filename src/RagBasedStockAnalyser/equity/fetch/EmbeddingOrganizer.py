from RagBasedStockAnalyser.redis.CreateRedisStore import *
from RagBasedStockAnalyser.redis.RedisQueryRunner import *
from RagBasedStockAnalyser.redis.VectorStore import Document, LexicalDocument, LexicalDocuments, VectorStore
from ..storeData.FetchData import TranscriptParser
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,TfidfVectorizer
# Use the already extracted classes from redis module
#Semantic chuncking applied to data broken down per user .
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class EmbeddingOrganizer:
# Preprocess function
    @staticmethod
    def preprocess(doc):
        logging.debug(f"Preprocessing document: {doc[:30]}...")
        doc = doc.lower()
        doc = re.sub(r"[^\w\s]", "", doc)
        tokens = doc.split()
        result = " ".join([t for t in tokens if t not in ENGLISH_STOP_WORDS])
        logging.debug(f"Preprocessed result: {result[:30]}...")
        return result
    @staticmethod
    def getSemanticChunks(text: str,breakpoint_threshold_amount=.75) -> List[str]:
        logging.info(f"Splitting text into semantic chunks (threshold={breakpoint_threshold_amount})")
        embeddings = OpenAIEmbeddings()
        splitter = SemanticChunker(
            embeddings,
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
        chunks = splitter.split_text(text)
        logging.info(f"Split into {len(chunks)} chunks.")
        return chunks
    
    def storeLexicalData(self,allDocs,**kargs) -> bool:
        logging.info("Storing lexical data using TF-IDF vectorizer.")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(allDocs)
        features = vectorizer.get_feature_names_out()
        idf_scores = dict(zip(features, vectorizer.idf_))
        idf_score=json.dumps(idf_scores)

        storedDocs = []
        quater=kargs.get("quater")
        year=int(kargs.get("year"))
        ticker=kargs.get("ticker")

        for i, row in enumerate(matrix.toarray()):
            tfidf_dict = {
                features[j]: float(row[j])
                for j in range(len(row))
                if row[j] > 0
            }
            doc = LexicalDocument(
                id=f"lexical_{ticker}_{year}_{quater}_{i}",
                content=json.dumps(tfidf_dict),
                year=2025,
                doc_name=f"transcript_{ticker}_{quater}_{year}",
            )
            storedDocs.append(doc)

        store=LexicalDocuments(
            **{"documents":storedDocs,
            "idf_score":idf_score,
            "id":f"idfscore_{ticker}_{year}_{quater}"
            }
        )

        if storedDocs:
            vs = VectorStore()
            vs.storeLexicalData(store)
            logging.info(f"Stored {len(storedDocs)} lexical documents in VectorStore.")
            return True

        logging.warning("No documents to store.")
        return False

    def store(self,blocks:list,ticker:str,year:int,quater:str):

        logging.info(f"First block: {blocks[0]}")
        cleaned_docs = []
        vs = VectorStore()
        i = 0
        for b in blocks:
            ls = []
            for c in self.getSemanticChunks(b['text'].replace("\n", ""), .95):
                d = {}
                d["id"] = f"transcript_{ticker}_{year}_{quater}_{i}"
                d["content"] = c
                d["year"] = 2025
                d["doc_name"] = f"transcript_{ticker}_{year}_{quater}"
                d["chunk_id"] = i
                d["speaker"] = b['speaker']
                ls.append(Document(**d))
                i += 1
                cleaned_docs.append(self.preprocess(f'{b["speaker"]}:{c}'))
            vs.store(ls)
        self.storeLexicalData(cleaned_docs, quater=quater, year=year, ticker=ticker)
        logging.info(f"EmbeddingOrganizer.store completed.{len(blocks)} blocks processed.")

   