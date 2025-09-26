from .VectorStore import VectorStore
import redis
import numpy as np
import openai
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition  import IndexDefinition, IndexType
import os
from pydantic import BaseModel,field_validator
from typing import Optional
class ReportDoc(BaseModel):
    id:str
    content:str
    chunk_id:Optional[int]=0
    score:Optional[int]=0
    page_no:Optional[int]=0
    doc_name:str
    year:int
    content_type:Optional[str]=""
    embedding:Optional[np.ndarray]=None
    heading:Optional[str]=""
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
class ReportVectorStore(VectorStore):
    def storeReports(self, docs: list[ReportDoc]):
        for doc in docs:
            embedding = self.embed(doc.content).tobytes()
            self.r.hset(doc.id, mapping={
                "content": doc.content,
                "year": doc.year,
                "embedding": embedding,
                "doc_name": doc.doc_name,
                "chunk_id": doc.chunk_id,
                "heading":doc.heading,
                "page_no":doc.page_no,
                "content_type":doc.content_type
            })
            
    def retrieveReport(self, doc_id: str) -> ReportDoc:
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
        if "page_no" in fields:
            fields["page_no"] = int(fields["page_no"])
        
        fields["id"] = doc_id
        raw_embedding = self.r.hget(doc_id, "embedding")
        fields["embedding"] = np.frombuffer(raw_embedding, dtype=np.float32)
        return ReportDoc(**fields)
