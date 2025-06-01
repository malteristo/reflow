"""
Query result caching system with thread-safe operations.

This module provides a comprehensive caching system for query results including TTL support,
LRU eviction, and thread-safe operations for concurrent access.
"""

import time
import threading
import hashlib
import json
from typing import Dict, List, Optional, Any
from copy import deepcopy

from .types import QueryResult


class QueryCache:
    """Query result caching system."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.collection_metadata = {}  # Maps cache keys to collections they involve
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
        
        # Performance tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'cache_puts': 0,
            'invalidations': 0,
            'response_times': []
        }
    
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
        start_time = time.time()
        
        with self.lock:
            self._stats['total_requests'] += 1
            
            if key in self.cache:
                entry_time = self.access_times.get(key, 0)
                if time.time() - entry_time < self.ttl_seconds:
                    # Cache hit - add from_cache attribute
                    cached_result = deepcopy(self.cache[key])
                    if hasattr(cached_result, '__dict__'):
                        cached_result.from_cache = True
                    else:
                        # Handle mock objects and other types
                        try:
                            cached_result.from_cache = True
                        except AttributeError:
                            # Create a wrapper object if we can't set attribute
                            class CachedResultWrapper:
                                def __init__(self, result):
                                    self._result = result
                                    self.from_cache = True
                                
                                def __getattr__(self, name):
                                    return getattr(self._result, name)
                            
                            cached_result = CachedResultWrapper(cached_result)
                    
                    self._stats['hits'] += 1
                    response_time = time.time() - start_time
                    self._stats['response_times'].append(response_time)
                    return cached_result
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.access_times[key]
                    self._stats['misses'] += 1
            else:
                self._stats['misses'] += 1
            
            return None
    
    def put(self, key: str, result: QueryResult, collections: Optional[List[str]] = None) -> None:
        """Store result in cache."""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                if oldest_key in self.collection_metadata:
                    del self.collection_metadata[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = time.time()
            if collections:
                self.collection_metadata[key] = sorted(collections)  # Store sorted for consistent comparison
            self._stats['cache_puts'] += 1
    
    def invalidate_by_content_change(self, collection_name: str, changed_documents: List[str]) -> int:
        """
        Invalidate cache entries that might be affected by document changes.
        
        Args:
            collection_name: Name of the collection where documents changed
            changed_documents: List of document IDs that changed
            
        Returns:
            Number of cache entries invalidated
        """
        with self.lock:
            invalidated_count = 0
            keys_to_remove = []
            
            # For this implementation, we'll invalidate all entries that involve the collection
            # In a more sophisticated version, we could track which documents contributed to each result
            for key, collections in self.collection_metadata.items():
                if collection_name in collections:
                    keys_to_remove.append(key)
                
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
                    del self.collection_metadata[key]
                    invalidated_count += 1
            
            self._stats['invalidations'] += invalidated_count
            return invalidated_count
    
    def invalidate_by_collections(self, collection_names: List[str]) -> int:
        """
        Invalidate cache entries that involve specific collections.
        
        Args:
            collection_names: List of collection names to invalidate
            
        Returns:
            Number of cache entries invalidated
        """
        with self.lock:
            invalidated_count = 0
            keys_to_remove = []
            
            # Use stored collection metadata to identify which entries to invalidate
            for key, collections in self.collection_metadata.items():
                # Check if any of the collections to invalidate are involved in this cache entry
                if any(collection_name in collections for collection_name in collection_names):
                    keys_to_remove.append(key)
            
            # Remove identified cache entries
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
                    del self.collection_metadata[key]
                    invalidated_count += 1
            
            self._stats['invalidations'] += invalidated_count
            return invalidated_count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            total_requests = self._stats['total_requests']
            hits = self._stats['hits']
            misses = self._stats['misses']
            
            hit_rate = hits / total_requests if total_requests > 0 else 0.0
            miss_rate = misses / total_requests if total_requests > 0 else 0.0
            
            response_times = self._stats['response_times']
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            return {
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'cache_hits': hits,
                'cache_misses': misses,
                'cache_size': len(self.cache),
                'max_cache_size': self.max_size,
                'cache_puts': self._stats['cache_puts'],
                'invalidations': self._stats['invalidations'],
                'average_response_time': avg_response_time,
                'ttl_seconds': self.ttl_seconds
            }
    
    def invalidate(self, strategy: str = "all", **kwargs) -> None:
        """Invalidate cache entries."""
        with self.lock:
            if strategy == "all":
                invalidated_count = len(self.cache)
                self.cache.clear()
                self.access_times.clear()
                self.collection_metadata.clear()
                self._stats['invalidations'] += invalidated_count
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
                    if key in self.collection_metadata:
                        del self.collection_metadata[key]
                self._stats['invalidations'] += len(keys_to_remove)
            elif strategy == "manual":
                cache_keys = kwargs.get("cache_keys", [])
                invalidated_count = 0
                for key in cache_keys:
                    if key in self.cache:
                        del self.cache[key]
                        del self.access_times[key]
                        if key in self.collection_metadata:
                            del self.collection_metadata[key]
                        invalidated_count += 1
                self._stats['invalidations'] += invalidated_count
    
    def is_valid(self, key: str) -> bool:
        """Check if cache entry is valid."""
        return key in self.cache
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache) 