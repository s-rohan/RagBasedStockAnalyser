from RagBasedStockAnalyser.redis.CreateRedisStore import *
from RagBasedStockAnalyser.redis.RedisQueryRunner import *
from RagBasedStockAnalyser.redis.VectorStore import Document, LexicalDocument, LexicalDocuments, VectorStore
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,TfidfVectorizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List
import json
import logging
import collections
import numpy as np

# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
class EmbeddingOrganizer:
    logger = logging.getLogger(__name__)
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
    def getSemanticChunks(text: str,breakpoint_threshold_amount:float=.75) -> List[str]:
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
      
    def storeLexicalData(self,allDocs:List[str],**kargs) -> bool:
        '''Stores lexical data using TF-IDF vectorizer into the VectorStore.
        **kargs must contain 'quater', 'year', 'ticker'.
        lexical_id_docs: Optional[list] = None
        doc_name: Optional[str] = None
        '''

        logging.info("Storing lexical data using TF-IDF vectorizer.")
        quater=kargs.get("quater")
        year=int(kargs.get("year"))
        ticker=kargs.get("ticker")
        doc_name=kargs.get("doc_name",self.formatDefault(quater, year, ticker))
        id_parts=kargs.get("lexical_id_docs",collections.defaultdict(list))

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(allDocs)
        features = vectorizer.get_feature_names_out()
        idf_scores = dict(zip(features, vectorizer.idf_))
        #idf_score=json.dumps(idf_scores)

        storedDocs = []


        for i, row in enumerate(matrix.toarray()):
            tfidf_dict = {
                features[j]: float(row[j])
                for j in range(len(row))
                if row[j] > 0
            }
            if len(id_parts)==0:
                id_parts = ["lexical",ticker,year,quater,i]
            else:
                id_parts.append(i)
            doc = LexicalDocument(
                id=self.create_lexical_doc_id(id_parts),
                content=json.dumps(tfidf_dict),
                year=year,
                doc_name=doc_name,
            )
            storedDocs.append(doc)

        docs_id = self.get_lex_alldocs_id(quater, year, ticker)
        store=LexicalDocuments(
            **{"documents":storedDocs,
            "idf_score":idf_scores,
            "id":docs_id
            }
        )

        if storedDocs:
            vs = VectorStore()
            vs.storeLexicalData(store)
            logging.info(f"Stored {len(storedDocs)} lexical documents in VectorStore.")
            return True

        logging.warning("No documents to store.")
        return False
    
        
    def get_lex_alldocs_id(self, quater:str, year:int|str, ticker:str,doc_type="idfscore"):
        return self.formatDefault(quater, year, ticker,doc_type=doc_type)
    
    def create_lexical_doc_id(self, params:list[str|int],sperator:str="_" ):
        return sperator.join(f"{p}" for p in params)

    def formatDefault(self, quater:str, year:int|str, ticker:str,doc_type="transcript"):
        return f"{doc_type}_{ticker}_{quater}_{year}"

    def store(self,blocks:list,ticker:str,year:int,quater:str):
        '''Stores the given blocks of text into the VectorStore with embeddings and lexical data.'''

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
                d["year"] = year
                d["doc_name"] = f"transcript_{ticker}_{year}_{quater}"
                d["chunk_id"] = i
                d["speaker"] = b['speaker']
                ls.append(Document(**d))
                i += 1
                cleaned_docs.append(self.preprocess(f'{b["speaker"]}:{c}'))
            vs.store(ls)
        self.storeLexicalData(cleaned_docs, quater=quater, year=year, ticker=ticker)
        logging.info(f"EmbeddingOrganizer.store completed.{len(blocks)} blocks processed.")

   