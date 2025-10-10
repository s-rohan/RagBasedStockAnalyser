import redis
r = redis.Redis(host="host.docker.internal", port=6379, decode_responses=False)
# Scan for keys matching the pattern
def update_keys(oldKey:str="*_AAPL_*.json1_*"):
    for old_key in r.scan_iter("*_AAPL_*.json1_*"):
        # Generate new key name
        old_key_str = old_key.decode() 
        new_key = old_key_str.replace(".json1", "")
        
        # Rename the key
        r.rename(old_key_str, new_key)
        print(f"Renamed: {old_key_str} → {new_key}")

def update_feild(field_to_update:str,new_value_str:str):
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="lexical*", count=500)
        for key in keys:
            r.hset(key, field_to_update, new_value_str)
        if cursor == 0:
            break
def delete_keys(key_pattern:str=None):
    for key in r.scan_iter(match=key_pattern):
        r.delete(key)

delete_keys("lexical_TSLA*")
