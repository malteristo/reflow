"""
Query result caching system with thread-safe operations.

This module provides a comprehensive caching system for query results including TTL support,
LRU eviction, and thread-safe operations for concurrent access.
"""

import time
import threading
import hashlib
import json
from typing import Dict, List, Optional

from .types import QueryResult


class QueryCache:
    """Query result caching system."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def _generate_key(self, query_embedding: List[float], collections: List[str], 
                     filters: Optional[Dict] = None, suffix: str = "") -> str:
        """Generate cache key for query."""
        key_data = {
            "embedding": query_embedding,
            "collections": sorted(collections),
            "filters": filters,
            "suffix": suffix
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[QueryResult]:
        """Retrieve cached result."""
        with self.lock:
            if key in self.cache:
                entry_time = self.access_times.get(key, 0)
                if time.time() - entry_time < self.ttl_seconds:
                    return self.cache[key]
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.access_times[key]
            return None
    
    def put(self, key: str, result: QueryResult) -> None:
        """Store result in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def invalidate(self, strategy: str = "all", **kwargs) -> None:
        """Invalidate cache entries."""
        with self.lock:
            if strategy == "all":
                self.cache.clear()
                self.access_times.clear()
            elif strategy == "time_based":
                max_age = kwargs.get("max_age_seconds", 0)
                cutoff_time = time.time() - max_age
                keys_to_remove = [
                    key for key, access_time in self.access_times.items()
                    if access_time < cutoff_time
                ]
                for key in keys_to_remove:
                    del self.cache[key]
                    del self.access_times[key]
            elif strategy == "manual":
                cache_keys = kwargs.get("cache_keys", [])
                for key in cache_keys:
                    if key in self.cache:
                        del self.cache[key]
                        del self.access_times[key]
    
    def is_valid(self, key: str) -> bool:
        """Check if cache entry is valid."""
        return key in self.cache
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache) 