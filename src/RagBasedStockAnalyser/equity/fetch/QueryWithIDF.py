from RagBasedStockAnalyser.redis.VectorStore import VectorStore, LexicalDocuments,LexicalDocument
from RagBasedStockAnalyser.redis.RedisQueryRunner import RedisQueryRunner
from typing import List, Dict, Any
import numpy as np
import json
from itertools import chain
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from functools import lru_cache
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryWithIDF:
    def __init__(self, vector_store: VectorStore):
        logger.info("Initializing QueryWithIDF")
        self.vector_store = vector_store
        self.runner = RedisQueryRunner(vector_store)
    def _get_idf_data(self,retrivedDocs:list[LexicalDocument]):
        logger.info(f"Entering _get_idf_data with {len(retrivedDocs)} documents.")
        doc_vectors = []
        all_terms=set(chain.from_iterable(json.loads(doc.content).keys() for doc in retrivedDocs))
        feature_names=sorted(all_terms)

        for document in retrivedDocs:
            doc=json.loads(document.content)
            vec = np.array([doc.get(term, 0.0) for term in feature_names])
            doc_vectors.append(vec)
        doc_matrix = np.array(doc_vectors)

        return doc_matrix,feature_names
   
    def _get_lexical_cosine_scores(self, query: str, docs: list[LexicalDocument],top_k) -> list[dict]:
        logger.info(f"Calculating lexical cosine scores for query: {query} and {len(docs)} docs.")
        """
        Given a list of LexicalDocument (with .content as json tfidf dict),
        returns a dict of doc_id -> cosine similarity score for each doc, using the query string.
        """
        tfidf_matrix,feature_names=self._get_idf_data(docs)
        vectorizer = TfidfVectorizer(vocabulary=feature_names)
        q_vec = vectorizer.fit_transform([query]).toarray()
        similarities = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_k = min(top_k, len(similarities)) 
        top_indices = similarities.argsort()[-top_k:][::-1]
        retDocs=list()
        
        for i in top_indices:
            if i < len(docs):  # Extra safety check
                doc={"score":similarities[i],
                "content":docs[i].content,
                "id":docs[i].id
                }
            retDocs.append(doc)
        return retDocs
    
    def fetch_and_lexical(self, query: str, top_k: int = 11) -> list:
        """
        Calls fetch_vector_results, extracts all unique (ticker, year, quater) from doc_name or id using EmbeddingOrganizer patterns,
        fetches lexical results for each, merges idf scores, and returns merged/ranked results for each doc group.
        """
        
        logger.info(f"fetch_and_lexical called with query='{query}', top_k={top_k}")
        vector_results = self.fetch_vector_results(query, top_k=top_k)
        if not vector_results:
            return []
        # Extract all unique (ticker, year, quater) from doc_name or id
        doc_keys = set()
        doc_key_to_results = defaultdict(list)
        doc_key_lexical_results=defaultdict(list)
        idf_scores_dict=defaultdict(dict)
        for doc in vector_results:
            doc_name = doc.get('doc_name', '')
            doc_id = doc.get('id', '')
            ticker = year = quater = None
            m = re.match(r"transcript_([A-Za-z0-9]+)_(\d{4})_([Qq][1-4])", doc_name)
            if m:
                ticker, year, quater = m.group(1), m.group(2), m.group(3)
            else:
                m2 = re.match(r"transcript_([A-Za-z0-9]+)_(\d{4})_([Qq][1-4])_", doc_id)
                if m2:
                    ticker, year, quater = m2.group(1), m2.group(2), m2.group(3)
            key =(ticker, int(year), quater)
            ticker, year, quater =  key
              
            if len(doc_key_lexical_results[key])==0:
                all_lexical_docs = self.fetch_lexical_for_doc(ticker, year, quater)
                lexical_docs=self._get_lexical_cosine_scores(query,all_lexical_docs.documents,top_k)

                doc_key_lexical_results[key].extend(lexical_docs)

            # Get idf_score dict if available
            idf_scores = idf_scores_dict[key]
            if len(idf_scores) ==0 and hasattr(all_lexical_docs, 'idf_score'):
                if not isinstance(all_lexical_docs.idf_score, dict):
                    idf_scores = json.loads(all_lexical_docs.idf_score)
                else:
                    idf_scores = all_lexical_docs.idf_score
                idf_scores_dict[key]=idf_scores
            doc_key_to_results[key].append(doc)

        final_results = []
        for key in doc_key_to_results.keys():
            ticker, year, quater = key
            idf_scores = idf_scores_dict[key]
            logger.info(f"Merging {len(doc_key_to_results[key])} vector results with {len(doc_key_lexical_results[key])} lexical docs for {key}")    
            merged = self.merge_and_rank(query,doc_key_to_results[key], idf_scores, doc_key_lexical_results[key], top_k=len(doc_key_to_results[key]),key=key)
            if len(merged)>0:
                final_results.append({
                    'ticker': ticker,
                    'year': year,
                    'quater': quater,
                    'results': merged
                })
        return final_results


    def fetch_vector_results(self, query: str, top_k: int = 11) -> List[Dict[str, Any]]:
        logger.info(f"Fetching vector results for query: {query}, top_k: {top_k}")
        # Embed the query and use RedisQueryRunner to search
        query_embedding = self.vector_store.embed(query)
        results_dict = self.runner.run_query(query, query_embedding, top_k)
        all_results = []
        idx =self.runner.index_a
        for doc in results_dict.get(idx, {}).get('results', []):
                # Convert to dict if needed
            if hasattr(doc, '__dict__'):
                all_results.append(doc.__dict__)
            else:
                all_results.append(doc)
        return all_results
    
    #@lru_cache(maxsize=None)
    def fetch_lexical_for_doc(self, ticker: str, year: int, quater: str) -> LexicalDocuments:
        logger.info(f"Fetching lexical docs for ticker={ticker}, year={year}, quater={quater}")
        # Build idf_key using provided values, use '*' as wildcard if any are None
        t = ticker if ticker is not None else '*'
        y = year if year is not None else '*'
        q = quater if quater is not None else '*'
        idf_key = f"{t}_{y}_{q}"
        lexical_docs: LexicalDocuments = self.vector_store.retriveLexicalData(f"lexical_{idf_key}_*", f"{idf_key}")
        return lexical_docs
    def normalize(self, scores: dict) -> dict:
        logger.info(f"Normalizing scores for {len(scores)} items.")
        values = list(scores.values())
        if not values:
            return {k: 0 for k in scores}
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-8:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val + 1e-8) for k, v in scores.items()}

    def merge_and_rank(self, query: str, vector_results: List[Dict[str, Any]], idf_scores: Dict[str, float], lexicalList: list[dict],key:tuple, top_k: int = 5, alpha: float = 0.7) -> List[str]:
        logger.info(f"Merging and ranking: query='{query}', {len(vector_results)} vector_results, {len(lexicalList)} docs, top_k={top_k}, alpha={alpha}")
        # Prepare semantic (vector) and lexical (idf) scores
        semantic_scores = {str(res.get('id')).split("_")[-1]: float(res.get('score', 0)) for res in vector_results}
        # Use helper to get lexical cosine scores
        #lexical_scores_list = self._get_lexical_cosine_scores(query,docs,top_k)
        lexical_scores={s["id"].split("_")[-1]:s["score"] for s in lexicalList}

        doc_map = {res["id"]:res["content"] for res in lexicalList}
        # Normalize both
        semantic_norm = self.normalize(semantic_scores)
        lexical_norm = self.normalize(lexical_scores)
        # Compute hybrid score for sorting only
        all_ids = set(semantic_norm) | set(lexical_norm)
        hybrid_scores = {}
        for doc_id in all_ids:
            sem = semantic_norm.get(doc_id, 0)
            lex = lexical_norm.get(doc_id, 0)
            hybrid_score = alpha * sem + (1 - alpha) * lex
            hybrid_scores[doc_id] = hybrid_score
        # Attach original score (semantic if present, else lexical) and use hybrid for sorting
        id_to_res = {str(res.get('id')): res["content"] for res in vector_results}
        merged = []
        for doc_id in sorted(hybrid_scores, key=lambda x: hybrid_scores[x], reverse=True)[:top_k]:
            transcript_id=f'transcript_{"_".join(str(k) for k in key)}_{doc_id}'
            lexical_id=f'lexical_{"_".join(str(k) for k in key)}_{doc_id}'
            if transcript_id in id_to_res:
                res = id_to_res[transcript_id]
            elif lexical_id in doc_map:
                # Convert LexicalDocument to dict if needed
                res = self.vector_store.retrieve(transcript_id).content
                logger.info(f"Using lexical doc for id {lexical_id} :res {res}")
            else:
                continue
            # Use semantic score if present, else lexical
            #orig_score = semantic_scores.get(doc_id, lexical_scores.get(doc_id, 0))
            merged.append(res)
        return merged


