import os
from dotenv import load_dotenv
import redis
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory
from langchain.callbacks.tracers import LangChainTracer

from langchain.globals import set_llm_cache
from langchain_core.messages import BaseMessage
from langchain.schema import Generation
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache


# Load environment variables from .env file
load_dotenv()


class BaseRedisMemoryAgent:
    def __init__(self,**kwargs):
        # Load config from environment variables
        self.initializeRedisAgent(**kwargs)
        self.tracer = LangChainTracer()



    def initializeRedisAgent(self,**kwargs):
        try:
            redis_host = os.getenv("REDIS_HOST")
            if not redis_host:
                raise EnvironmentError("REDIS_HOST environment variable is not set.")
            redis_port = int(os.getenv("REDIS_PORT"))
            redis_password = os.getenv("REDIS_PASSWORD")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            redis_host = os.getenv("REDIS_HOST")

            try:
                ttl_env = os.getenv("REDIS_TTL_SECONDS", "3600")
                self.ttl_seconds = int(ttl_env)
            except ValueError:
                self.ttl_seconds = 3600  # fallback to default
            except ValueError:
                redis_port = 6379  # fallback to default
            redis_password = os.getenv("REDIS_PASSWORD")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if kwargs==None:
                self.openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            else:
                self.openai_model =kwargs.get("OPENAI_MODEL",os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            self.ttl_seconds = int(os.getenv("REDIS_TTL_SECONDS", 3600))
            redis_username = os.getenv("REDIS_USERNAME","AppUser")
            addSematicCaching=kwargs.get("Add_Sematic_Caching",True)
            redis_password
            try:
                redis_url = os.getenv("REDIS_URL")
                if redis_url:
                    self.r = redis.from_url(redis_url, decode_responses=True)
                else:
                    pool = redis.ConnectionPool(
                        host=redis_host,
                        port=redis_port,
                        password=redis_password,
                        decode_responses=True,
                        username=redis_username,
                        max_connections=5 # Adjust as needed for your workload
                    )

                    self.r = redis.Redis(connection_pool=pool)
                self.r.ping()  # Test the connection
                print("Connected to Redis successfully!")
            except redis.exceptions.ConnectionError as e:
                print(f"Failed to connect to Redis: {e}")
                raise

            
            embeddings = OpenAIEmbeddings()
            class VerboseSemanticCache(RedisSemanticCache):
                def lookup(self, prompt: str, llm_string: str):
                    result = super().lookup(prompt, llm_string)
                    print("🔁 Cache hit!" if result else "🆕 Cache miss.")
                    return result
            
            if addSematicCaching:
                semantic_cache = RedisSemanticCache(
                    embeddings=embeddings,
                    redis_client= self.r,
                    distance_threshold=0.2,
                    prefix='cache_openAI'
                )

                set_llm_cache(VerboseSemanticCache(semantic_cache))
            modelParam=kwargs.get("Model_Param",{"temperature":0.1})
            
          

            self.model = ChatOpenAI(
                model=self.openai_model,
                api_key=openai_api_key,
                **modelParam

            )
        except ValueError:
            print(ValueError)


    



    



    class MergedRedisChatHistory(BaseChatMessageHistory):
        
        def __init__(self, history_store, summary_store):
            self.history_store = history_store
            self.summary_store = summary_store

        @property
        def messages(self) -> list[BaseMessage]:
            return sorted(
                self.history_store.messages + self.summary_store.messages,
                key=lambda m: getattr(m, "timestamp", 0)
            )

        def add_message(self, message: BaseMessage) -> None:
            self.history_store.add_message(message)

        def clear(self) -> None:
            self.history_store.clear()
            self.summary_store.clear()


    def get_redis_history(self,session_id: str) -> BaseChatMessageHistory:
            return RedisChatMessageHistory(session_id=session_id, redis_client=self.r,key_prefix="chat_history:")
    def get_redis_summary(self,session_id: str) -> BaseChatMessageHistory:
            return RedisChatMessageHistory(session_id=session_id, redis_client=self.r,key_prefix="chat_summary:")
    def get_merged_messages(self, session_id: str) -> BaseMessage:
        history = self.get_redis_history(session_id)
        summary = self.get_redis_summary(session_id)
        merged = self.MergedRedisChatHistory(history_store=history ,summary_store=summary)
        return merged

   

