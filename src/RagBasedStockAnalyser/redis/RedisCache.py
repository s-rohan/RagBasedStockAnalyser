import redis
import time
import hashlib
import pickle
from RagBasedStockAnalyser.common.logging_config import setup_logging
logger = setup_logging(logger_name=__name__)

# Connect to the Redis server
redis_client = redis.StrictRedis(host="host.docker.internal", port=6379,db=0, decode_responses=True)
redis_client_binary = redis.StrictRedis(host="host.docker.internal", port=6379,db=0, decode_responses=False)

def redis_cache(ttl=0):
    """
    A decorator to cache function results in Redis.
    :param ttl: Time-to-live for the cache in seconds.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a unique cache key based on the function name and arguments
            key = f"{func.__name__}:{hashlib.sha256(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Check if the result is already cached
            cached_result = redis_client.get(key)
            if cached_result:
                logger.info("Cache hit!")
                return eval(cached_result)  # Convert the string back to a Python object
            
            logger.info("Cache miss. Computing result...")
            # Compute the result and cache it
            result = func(*args, **kwargs)
            if ttl>0:
                redis_client.set(name=key, ex=ttl,value= repr(result))  # Store the result with TTL
            else:
                redis_client.set(name=key, value=repr(result))
            return result
        return wrapper
    return decorator

def llm_redis_cache(ttl=0):
    """
    A decorator to cache function results in Redis.
    :param ttl: Time-to-live for the cache in seconds.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a unique cache key based on the function name and arguments
            key = f"{func.__name__}:{hashlib.sha256(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Check if the result is already cached
            cached_result = redis_client_binary.get(key)
            if cached_result:
                logger.info("Cache hit!")
                return pickle.loads(cached_result)  # Convert the string back to a Python object
            
            logger.info("Cache miss. Computing result...")
            # Compute the result and cache it
            result = func(*args, **kwargs)
            if ttl>0:
                redis_client_binary.set(name=key, ex=ttl,value= pickle.dumps(result))  # Store the result with TTL
            else:
                redis_client_binary.set(name=key, value=pickle.dumps(result))
            return result
        return wrapper
    return decorator