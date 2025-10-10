# Compare query and get similar chunks
from redis.commands.search.query import Query
import numpy as np
import time
from typing import Optional
from pydantic import BaseModel, field_validator
from RagBasedStockAnalyser.redis.VectorStore import VectorStore
import logging
from  typing import List

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class StoredDocument(BaseModel):
    content:Optional[str]=None
    doc_name:Optional[str]=None
    id:str
    score:float
    year:Optional[int]=0
    embedding:Optional[np.ndarray]=None
    others:Optional[dict]=None
    @field_validator("embedding", mode="before")
    def normalize_embedding(cls, v):
        if isinstance(v, (bytes, bytearray, memoryview)):
            return np.frombuffer(v, dtype=np.float32)
        elif isinstance(v, list):
            return np.array(v, dtype=np.float32)
        elif isinstance(v, np.ndarray):
            return v.astype(np.float32)
        raise TypeError(f"Unsupported embedding format: {type(v)}")
    model_config = {
        "arbitrary_types_allowed": True
    }
    


class RedisQueryRunner:
    def __init__(self, vs:VectorStore, index_a:str=None):
        self.vs = vs
        self.store = vs.r
        if index_a is None:
            index_a = "transcript_idx"
        self.index_a = index_a
        logger.info(f"RedisQueryRunner initialized with index: {self.index_a}")

    def lexicalSearch(self, queryStr, index='idf_lexical_idx'):
        logger.info(f"Performing lexical search on index '{index}' with query: {queryStr}")
        q = Query(queryStr) \
            .sort_by("idf_score", asc=False) \
            .return_fields("content", "doc_name", "idf_score", "id") \
            .paging(0, 10)
        start = time.time()
        res = self.store.ft(index).search(q)
        latency = time.time() - start
        results = [doc for doc in res.docs]
        logger.info(f"Lexical search returned {len(results)} results in {latency:.4f} seconds.")
        return results, latency


    def build_vector_query(
        self,
        top_k: int,
        query_vec: bytes,
        return_fields: Optional[List[str]] = None,
        queryStr: Optional[str] = None,
        vector_field: str = "embedding",
        sort_by: str = "score"
    ) -> Query:
            """
            Constructs a RediSearch vector query with dialect 2 and optional field filtering.

            Args:
                top_k (int): Number of nearest neighbors to retrieve.
                query_vec (bytes): Serialized vector embedding (FLOAT32).
                return_fields (Optional[List[str]]): Fields to return. If None, returns all.
                queryStr (Optional[str]): Custom query string. Defaults to KNN clause.
                vector_field (str): Name of the vector field in the index.
                sort_by (str): Field to sort results by.

            Returns:
                Query: Configured RediSearch Query object.
            """

            # Validate vector format
            assert isinstance(query_vec, bytes), f"query_vec must be bytes, got {type(query_vec)}"
            expected_dim = 1536  # Customize if needed
            assert len(query_vec) == expected_dim * 4, f"query_vec length mismatch: expected {expected_dim*4}, got {len(query_vec)}"

            # Default query string
            if queryStr is None:
                queryStr = f"*=>[KNN {top_k} @{vector_field} ${vector_field} AS {sort_by}]"

            # Log provenance
            logger.info(f"Constructing vector query: {queryStr}")
            logger.info(f"Return fields: {return_fields or 'ALL'}, dialect=2, sort_by={sort_by}")

            # Build query
            q = Query(queryStr).sort_by(sort_by, asc=False).dialect(2)

            if return_fields:
                q = q.return_fields(*return_fields)

            return q


    def search(self, index, top_k, query_vec, queryStr=None, return_fields: Optional[list] = None):
        # Ensure query_vec is bytes and correct length
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec, dtype=np.float32).tobytes()
        elif isinstance(query_vec, np.ndarray):
            query_vec = query_vec.astype(np.float32).tobytes()
        elif not isinstance(query_vec, bytes):
            raise TypeError(f"query_vec must be bytes, got {type(query_vec)}")

        expected_dim = 1536
        if len(query_vec) != expected_dim * 4:
            logger.error(f"Embedding length mismatch: expected {expected_dim*4}, got {len(query_vec)}")
            raise ValueError(f"Embedding length mismatch: expected {expected_dim*4}, got {len(query_vec)}")

        logger.info(f"Performing vector search on index '{index}' with top_k={top_k} and embedding type {type(query_vec)}, length {len(query_vec)}")
        q = self.build_vector_query(top_k=top_k,
                                   query_vec=query_vec,
                                   queryStr=queryStr,
                                   return_fields=return_fields)

        start = time.time()
        res = self.store.ft(index).search(q, query_params={"embedding": query_vec})
        latency = time.time() - start

        results = []
        for doc in res.docs:
            raw_embedding = doc.embedding
            try:
                if isinstance(raw_embedding, memoryview):
                    raw_embedding = bytes(raw_embedding)
                elif isinstance(raw_embedding, str):
                    raw_embedding = raw_embedding.encode('latin1')
                elif isinstance(raw_embedding, list):
                    raw_embedding = np.array(raw_embedding, dtype=np.float32).tobytes()
                elif not isinstance(raw_embedding, (bytes, bytearray)):
                    raise TypeError(f"Unexpected embedding type: {type(raw_embedding)}")

                doc.embedding = np.frombuffer(raw_embedding, dtype=np.float32)

            except Exception as e:
                try:
                    fallback_raw = self.vs.retrieve(doc.id)
                    doc.embedding = fallback_raw.embedding
                except Exception as fallback_error:
                    doc.embedding = None

            results.append(doc)

        return results, latency


    def run_query(self, query_text, query_vec_emb, top_k=10, return_fields: list = None):
        # Accept list, np.ndarray, or bytes for embedding
        if isinstance(query_vec_emb, list):
            query_vec = np.array(query_vec_emb, dtype=np.float32).tobytes()
        elif isinstance(query_vec_emb, np.ndarray):
            query_vec = query_vec_emb.astype(np.float32).tobytes()
        elif isinstance(query_vec_emb, bytes):
            query_vec = query_vec_emb
        else:
            raise TypeError(f"query_vec_emb must be list, np.ndarray, or bytes, got {type(query_vec_emb)}")

        results_a, latency_a = self.search(self.index_a, top_k, query_vec)
        return {
            "query": query_text,
            self.index_a: {
                "results": results_a,
                "latency_ms": latency_a
            }
        }
    
