#Vector store to store and read from redis
import redis
import numpy as np
import openai
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition  import IndexDefinition, IndexType
import os
from pydantic import BaseModel,field_validator
from typing import Optional
from dotenv import load_dotenv
from functools import lru_cache
import json
load_dotenv()
class Document(BaseModel):
    id:str
    content:str
    chunk_id:Optional[int]=0
    score:Optional[int]=0
    doc_name:str
    year:int
    embedding:Optional[np.ndarray]=None
    speaker:Optional[str]=None
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

class LexicalDocument(BaseModel):
    id:str
    content:str
    doc_name:str
    year:int
class LexicalDocuments(BaseModel):
     documents:list[LexicalDocument]
     id:str
     idf_score:dict
     model_config = {
        "arbitrary_types_allowed": True
    }
         
    


class VectorStore:
    def __init__(self, host: str = "host.docker.internal", port: int = 6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=False)
    @lru_cache(maxsize=1000)
    def embed(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    def storeLexicalData(self, store:LexicalDocuments):
        docs=store.documents

        for doc in docs:
                    self.r.hset(doc.id, mapping={
                    "content": doc.content,
                    "year": doc.year,
                    "doc_name": doc.doc_name
                
                })
        self.r.hset(store.id,mapping=store.idf_score)
    def retriveLexicalData(self, contentKey: str, idfscoreKey: str) -> LexicalDocuments:
        docs: list[LexicalDocument] = []

        for key in self.r.scan_iter(contentKey):
            raw_fields = self.r.hgetall(key)
            fields = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_fields.items()}
            if "year" in fields:
                fields["year"] = int(fields["year"])
            fields["id"]=key
            doc = LexicalDocument(**fields)
            docs.append(doc)

        raw_idf_fields = self.r.hgetall(idfscoreKey)
        idf_fields = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_idf_fields.items()}

        lexical_bundle = {
            "documents": docs,
            "id": idfscoreKey,
            "idf_score":idf_fields
        }

        return LexicalDocuments(**lexical_bundle)


         


    def store(self, docs: list[Document]):
        for doc in docs:
            embedding = self.embed(doc.content).tobytes()
            if doc.speaker is None:
                    self.r.hset(doc.id, mapping={
                    "content": doc.content,
                    "year": doc.year,
                    "embedding": embedding,
                    "doc_name": doc.doc_name,
                    "chunk_id": doc.chunk_id
                
                })
            else:
                self.r.hset(doc.id, mapping={
                    "content": doc.content,
                    "year": doc.year,
                    "embedding": embedding,
                    "doc_name": doc.doc_name,
                    "chunk_id": doc.chunk_id,
                    "speaker":doc.speaker
                })
            
    def retrieve(self, doc_id: str) -> Document:
        raw_fields = self.r.hgetall(doc_id)
        fields = {k.decode(): v.decode() for k, v in raw_fields.items() if k != b'embedding'}
        if len(fields)==0:
            return None
        if "chunk_id" in fields:
            fields["chunk_id"] = int(fields["chunk_id"])
        if "year" in fields:
            fields["year"] = int(fields["year"])
        if "score" in fields:
            fields["score"] = int(fields["score"])
        
        fields["id"] = doc_id
        


        raw_embedding = self.r.hget(doc_id, "embedding")
        if raw_embedding is None and "lexical" not in doc_id:
            print(f"{raw_fields} raw feilds")
            raise ValueError(f"Missing 'embedding' for doc_id: {doc_id}")

        fields["embedding"] = np.frombuffer(raw_embedding, dtype=np.float32)
        return Document(**fields)
