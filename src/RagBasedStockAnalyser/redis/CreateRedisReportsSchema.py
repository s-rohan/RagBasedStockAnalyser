import redis
import numpy as np
import openai
from redis.commands.search.field import TextField, TagField, VectorField

from redis.commands.search.index_definition  import IndexDefinition, IndexType
from dotenv import load_dotenv
load_dotenv()
# Initialize Redis client
r = redis.Redis(host="host.docker.internal", port=6379, decode_responses=False)

# Set your OpenAI API key
#openai.api_key =  os.getenv("OPENAI_API_KEY")


def embed(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


try:
    print(r.ping())  # Should return True
except Exception as e:
    print(f"Connection failed: {e}")




def create_redis_reports_schema():
    # Drop existing index if it exists. dont do in production
    try:
         r.ft("report_idx").dropindex(delete_documents=True)
    except redis.exceptions.ResponseError:
        pass
    schema_reports = (
        TextField("content"),
        TagField("Year"),
        TagField("doc_name"),
        TagField("content_type"),
        TagField("heading"),
        TagField("chunk_id"),
        TagField("page_no"),
        VectorField("embedding", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE"
        })
    )
   
   


    r.ft("report_idx").create_index(
        fields=schema_reports,
        definition=IndexDefinition(prefix=["report_"], index_type=IndexType.HASH)
    )
    
    print("Redis vector index created.")


if __name__ == "__main__":
    create_redis_reports_schema()
