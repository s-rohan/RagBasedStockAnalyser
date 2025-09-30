# Compare query and get similar chunks
from redis.commands.search.query import Query
import numpy as np
import time
from typing import Optional
from pydantic import BaseModel, field_validator
from RagBasedStockAnalyser.redis.VectorStore import VectorStore
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
    def __init__(self, vs:VectorStore, index_a="transcript_idx"):
        self.vs = vs
        self.store=vs.r
    
        self.index_a = index_a
        
    def lexicalSearch(self,queryStr,index='idf_lexical_idx'):
        q = Query(queryStr) \
                .sort_by("idf_score", asc=False) \
                .return_fields("content", "doc_name","idf_score", "id") \
                .paging(0, 10)
        start = time.time()
        res =  self.store.ft(index).search(q)
        latency = time.time() - start
        results=[doc for doc in res.docs]
        return results,latency

    def search(self,index, top_k,query_vec,queryStr=None,return_fields:Optional[list]=None,getAllFields:bool=False):
            if queryStr is None:
                queryStr=f"*=>[KNN {top_k} @embedding $vec AS score]"
            if return_fields is None:
                return_fields=self.getDefaultReturnFields()
            q = None
            if getAllFields:
                q=Query(queryStr) \
                .sort_by("score", asc=False) \
                .dialect(2)
            else:
                q=Query(queryStr) \
                    .return_fields(*return_fields) \
                    .sort_by("score", asc=False) \
                    .dialect(2)

            start = time.time()
            res =  self.store.ft(index).search(q, query_params={"vec": query_vec})
            latency = time.time() - start

            results = []
            for doc in res.docs:
                raw_embedding = doc.embedding
                try:
                    if isinstance(raw_embedding, memoryview):
                        raw_embedding = bytes(raw_embedding)
                    elif isinstance(raw_embedding, str):
                        raw_embedding = raw_embedding.encode('latin1')
                    elif not isinstance(raw_embedding, (bytes, bytearray)):
                        raise TypeError(f"Unexpected embedding type: {type(raw_embedding)}")

                    doc.embedding = np.frombuffer(raw_embedding, dtype=np.float32)

                except Exception as e:
                    #print(f"Failed to decode embedding for doc {doc.id} from search: {e}")
                    try:
                        fallback_raw = self.vs.retrieve(doc.id)
                        doc.embedding = fallback_raw.embedding
                        #print(f"Recovered embedding for doc {doc.id} via fallback")
                    except Exception as fallback_error:
                        #print(f"Fallback failed for doc {doc.id}: {fallback_error}")
                        doc.embedding = None

                results.append(doc)

            return results, latency

    def getDefaultReturnFields(self):
        return ["content", "doc_name", "chunk_id", "score", "id", "embedding"]

    def run_query(self, query_text,query_vec_emb:np.array, top_k=10):
        query_vec = query_vec_emb.astype(np.float32).tobytes()      
        results_a, latency_a = self.search(self.index_a,top_k,query_vec)
        
        return {
            "query": query_text,
            self.index_a: {
                "results": results_a,
                "latency_ms": latency_a
            }
        }
