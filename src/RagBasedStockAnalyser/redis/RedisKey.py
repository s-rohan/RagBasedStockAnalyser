import redis
r = redis.Redis(host="host.docker.internal", port=6379, decode_responses=False)
# Scan for keys matching the pattern
for old_key in r.scan_iter("*_AAPL_*.json1_*"):
    # Generate new key name
    old_key_str = old_key.decode() 
    new_key = old_key_str.replace(".json1", "")
    
    # Rename the key
    r.rename(old_key_str, new_key)
    print(f"Renamed: {old_key_str} → {new_key}")
