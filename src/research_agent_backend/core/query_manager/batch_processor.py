"""
Batch query processing engine with parallel and sequential execution.

This module provides batch processing capabilities for multiple queries with support for
both parallel and sequential execution strategies.
"""

import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchQueryProcessor:
    """Batch query processing engine."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_batch(self, queries: List[Dict], query_manager, **kwargs) -> Dict[str, Any]:
        """Process a batch of queries."""
        strategy = kwargs.get("strategy", "sequential")
        parallel_config = kwargs.get("parallel_config", {})
        
        if strategy == "parallel" or parallel_config.get("enable_parallel"):
            return self._process_parallel(queries, query_manager, parallel_config)
        else:
            return self._process_sequential(queries, query_manager)
    
    def _process_sequential(self, queries: List[Dict], query_manager) -> Dict[str, Any]:
        """Process queries sequentially."""
        start_time = time.time()
        results = []
        failed_count = 0
        
        for query in queries:
            try:
                result = query_manager.similarity_search(
                    query_embedding=query["query_embedding"],
                    collections=query["collections"]
                )
                results.append({
                    "query_id": query.get("query_id"),
                    "result": result
                })
            except Exception as e:
                failed_count += 1
                results.append({
                    "query_id": query.get("query_id"),
                    "error": str(e)
                })
        
        return {
            "results": results,
            "total_processed": len(queries),
            "failed_count": failed_count,
            "batch_execution_time": time.time() - start_time,
            "strategy_used": "sequential"
        }
    
    def _process_parallel(self, queries: List[Dict], query_manager, config: Dict) -> Dict[str, Any]:
        """Process queries in parallel."""
        start_time = time.time()
        results = []
        failed_count = 0
        max_workers = min(config.get("max_workers", 4), len(queries))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {}
            for query in queries:
                future = executor.submit(
                    query_manager.similarity_search,
                    query_embedding=query["query_embedding"],
                    collections=query["collections"]
                )
                future_to_query[future] = query
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    results.append({
                        "query_id": query.get("query_id"),
                        "result": result
                    })
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "query_id": query.get("query_id"),
                        "error": str(e)
                    })
        
        return {
            "results": results,
            "total_processed": len(queries),
            "failed_count": failed_count,
            "batch_execution_time": time.time() - start_time,
            "parallel_execution": True,
            "workers_used": max_workers,
            "chunks_processed": max_workers,
            "strategy_used": "parallel"
        } 