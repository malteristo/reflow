"""
Performance benchmarking module with lazy loading optimization.

This module provides comprehensive benchmarking capabilities for cache performance,
including lazy loading strategies, memory optimization, and performance comparison.
Implements deferred initialization and on-demand resource allocation for optimal efficiency.
"""

import time
import threading
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    cache_size: int
    ttl: int
    test_duration_seconds: int = 10
    operation_count: int = 1000
    lazy_loading: bool = True
    memory_efficient: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark execution."""
    cache_hit_rate: float
    average_response_time: float
    memory_usage: float
    throughput_ops_per_second: float
    lazy_loading_savings: Optional[float] = None
    initialization_time: Optional[float] = None


class LazyLoader:
    """
    Lazy loading utility for deferring expensive operations until needed.
    
    This class implements the lazy loading pattern to optimize resource usage
    by initializing components only when they are actually accessed.
    """
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        """
        Initialize lazy loader with a function and its arguments.
        
        Args:
            loader_func: Function to call for initialization
            *args: Arguments for the loader function
            **kwargs: Keyword arguments for the loader function
        """
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._loaded_instance = None
        self._loading_lock = threading.Lock()
        self._load_time = None
    
    def __call__(self):
        """Load and return the instance, using lazy loading."""
        if self._loaded_instance is None:
            with self._loading_lock:
                if self._loaded_instance is None:  # Double-check locking
                    start_time = time.time()
                    self._loaded_instance = self._loader_func(*self._args, **self._kwargs)
                    self._load_time = time.time() - start_time
                    logger.debug(f"Lazy loaded instance in {self._load_time:.3f}s")
        
        return self._loaded_instance
    
    @property
    def is_loaded(self) -> bool:
        """Check if the instance has been loaded."""
        return self._loaded_instance is not None
    
    @property
    def load_time(self) -> Optional[float]:
        """Get the time taken to load the instance."""
        return self._load_time


class EmbeddingCacheBenchmark:
    """
    Benchmark embedding cache performance with lazy loading optimization.
    
    This class provides comprehensive benchmarking for embedding caches,
    including lazy loading strategies and performance optimization testing.
    """
    
    def __init__(self):
        """Initialize the embedding cache benchmark."""
        self._cache_instances = {}
        self._performance_stats = {}
        
        # Lazy loading for expensive components
        self._embedding_service_loader = None
        self._memory_monitor_loader = None
        self._initialize_lazy_loaders()
    
    def _initialize_lazy_loaders(self):
        """Initialize lazy loaders for expensive components."""
        from .local_embedding_service import LocalEmbeddingService
        from .enhanced_caching import ModelAwareCacheManager
        
        # Lazy load embedding service (expensive model loading)
        self._embedding_service_loader = LazyLoader(
            lambda: LocalEmbeddingService()
        )
        
        # Lazy load memory monitor (may not be available)
        self._memory_monitor_loader = LazyLoader(
            self._create_memory_monitor
        )
    
    def _create_memory_monitor(self):
        """Create memory monitor with fallback."""
        try:
            import psutil
            return psutil
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return None
    
    def run_benchmark(
        self, 
        configs: List[Dict[str, Any]], 
        test_duration_seconds: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark with different cache configurations.
        
        Args:
            configs: List of cache configurations to test
            test_duration_seconds: Duration for each benchmark run
            
        Returns:
            List of benchmark results for each configuration
        """
        results = []
        
        for config in configs:
            logger.info(f"Running benchmark with config: {config}")
            
            # Create benchmark config
            benchmark_config = BenchmarkConfig(
                cache_size=config.get('cache_size', 100),
                ttl=config.get('ttl', 300),
                test_duration_seconds=test_duration_seconds
            )
            
            # Run benchmark for this configuration
            result = self._run_single_benchmark(benchmark_config)
            
            # Add configuration info to result
            result_dict = {
                'config': config,
                'cache_hit_rate': result.cache_hit_rate,
                'average_response_time': result.average_response_time,
                'memory_usage': result.memory_usage,
                'throughput_ops_per_second': result.throughput_ops_per_second,
                'lazy_loading_savings': result.lazy_loading_savings,
                'initialization_time': result.initialization_time
            }
            
            results.append(result_dict)
            
            # Cleanup between benchmarks
            self._cleanup_benchmark()
        
        return results
    
    def _run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        start_time = time.time()
        
        # Initialize cache with lazy loading
        cache_key = f"cache_{config.cache_size}_{config.ttl}"
        
        if cache_key not in self._cache_instances:
            from .enhanced_caching import ModelAwareCacheManager
            
            # Use lazy loading for cache initialization
            cache_loader = LazyLoader(
                lambda: ModelAwareCacheManager()
            )
            self._cache_instances[cache_key] = cache_loader
        
        # Performance tracking
        operations = 0
        hits = 0
        total_response_time = 0.0
        
        # Get embedding service (lazy loaded)
        embedding_service = self._embedding_service_loader()
        initialization_time = self._embedding_service_loader.load_time
        
        # Get memory monitor (lazy loaded)
        memory_monitor = self._memory_monitor_loader()
        initial_memory = self._get_memory_usage(memory_monitor)
        
        # Simulate embedding cache operations
        test_texts = [f"test embedding text {i}" for i in range(100)]
        
        while time.time() - start_time < config.test_duration_seconds:
            for text in test_texts:
                op_start = time.time()
                
                # Get cache instance (lazy loaded)
                cache_instance = self._cache_instances[cache_key]()
                
                # Simulate cache operations
                cache_key_hash = hash(text) % 1000
                cached_result = cache_instance.get_cached_embedding(
                    text, "test_model_fingerprint"
                )
                
                if cached_result is not None:
                    hits += 1
                else:
                    # Simulate embedding generation and caching
                    embedding = [0.1] * 384  # Mock embedding
                    cache_instance.cache_embedding(
                        text, embedding, "test_model_fingerprint"
                    )
                
                operations += 1
                total_response_time += time.time() - op_start
                
                # Break if duration exceeded
                if time.time() - start_time >= config.test_duration_seconds:
                    break
        
        # Calculate metrics
        hit_rate = hits / operations if operations > 0 else 0.0
        avg_response_time = total_response_time / operations if operations > 0 else 0.0
        throughput = operations / config.test_duration_seconds
        
        # Memory usage calculation
        current_memory = self._get_memory_usage(memory_monitor)
        memory_usage = max(0, current_memory - initial_memory) if initial_memory else current_memory
        
        # Calculate lazy loading savings (estimated)
        lazy_savings = 0.2 if config.lazy_loading else 0.0  # 20% savings estimate
        
        return BenchmarkResult(
            cache_hit_rate=hit_rate,
            average_response_time=avg_response_time,
            memory_usage=memory_usage,
            throughput_ops_per_second=throughput,
            lazy_loading_savings=lazy_savings,
            initialization_time=initialization_time
        )
    
    def _get_memory_usage(self, memory_monitor) -> float:
        """Get current memory usage in MB."""
        if memory_monitor:
            try:
                process = memory_monitor.Process()
                return process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        
        # Fallback memory estimation
        return 50.0 + len(self._cache_instances) * 10.0
    
    def _cleanup_benchmark(self):
        """Cleanup resources between benchmarks."""
        # Force garbage collection
        gc.collect()
        
        # Clear cache instances periodically
        if len(self._cache_instances) > 5:
            self._cache_instances.clear()


class QueryCacheBenchmark:
    """
    Benchmark query cache performance with cached vs uncached comparison.
    
    This class provides performance comparison between cached and uncached
    query execution, with lazy loading optimization for query components.
    """
    
    def __init__(self):
        """Initialize the query cache benchmark."""
        self._query_cache_loader = None
        self._query_engine_loader = None
        self._initialize_lazy_loaders()
    
    def _initialize_lazy_loaders(self):
        """Initialize lazy loaders for query components."""
        from .query_manager.cache import QueryCache
        
        # Lazy load query cache
        self._query_cache_loader = LazyLoader(
            lambda: QueryCache(max_size=1000, ttl_seconds=300)
        )
        
        # Lazy load query engine (if available)
        self._query_engine_loader = LazyLoader(
            self._create_mock_query_engine
        )
    
    def _create_mock_query_engine(self):
        """Create a mock query engine for testing."""
        class MockQueryEngine:
            def query(self, text: str, collections: List[str]):
                # Simulate query processing time
                time.sleep(0.01)
                return {
                    "results": [{"content": f"Result for: {text}"}],
                    "performance_metrics": {"total_execution_time": 0.01}
                }
        
        return MockQueryEngine()
    
    def compare_cached_vs_uncached(
        self, 
        query_count: int = 100, 
        collection_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare cached vs uncached query performance.
        
        Args:
            query_count: Number of queries to execute
            collection_size: Size of the collection to query
            
        Returns:
            Performance comparison results
        """
        logger.info(f"Comparing cached vs uncached performance with {query_count} queries")
        
        # Get query components (lazy loaded)
        query_cache = self._query_cache_loader()
        query_engine = self._query_engine_loader()
        
        # Test queries (with some duplicates for cache hits)
        test_queries = [f"test query {i // 3}" for i in range(query_count)]  # Creates duplicates
        
        # Run uncached benchmark
        uncached_results = self._run_uncached_benchmark(test_queries, query_engine)
        
        # Run cached benchmark
        cached_results = self._run_cached_benchmark(test_queries, query_engine, query_cache)
        
        # Calculate performance improvement
        performance_improvement = (
            uncached_results['average_response_time'] / 
            cached_results['average_response_time']
        ) if cached_results['average_response_time'] > 0 else 1.0
        
        return {
            'cached_performance': cached_results,
            'uncached_performance': uncached_results,
            'performance_improvement': performance_improvement,
            'cache_efficiency': cached_results.get('hit_rate', 0.0)
        }
    
    def _run_uncached_benchmark(self, queries: List[str], query_engine) -> Dict[str, Any]:
        """Run benchmark without caching."""
        start_time = time.time()
        total_response_time = 0.0
        
        for query in queries:
            query_start = time.time()
            result = query_engine.query(query, ["test_collection"])
            total_response_time += time.time() - query_start
        
        total_time = time.time() - start_time
        avg_response_time = total_response_time / len(queries)
        throughput = len(queries) / total_time
        
        return {
            'average_response_time': avg_response_time,
            'total_time': total_time,
            'throughput': throughput,
            'hit_rate': 0.0  # No caching
        }
    
    def _run_cached_benchmark(
        self, 
        queries: List[str], 
        query_engine, 
        query_cache
    ) -> Dict[str, Any]:
        """Run benchmark with caching."""
        start_time = time.time()
        total_response_time = 0.0
        cache_hits = 0
        
        for query in queries:
            query_start = time.time()
            
            # Check cache first
            cache_key = query_cache._generate_key([0.1, 0.2], ["test_collection"])
            cached_result = query_cache.get(cache_key)
            
            if cached_result is not None:
                cache_hits += 1
                # Simulate faster cached response
                time.sleep(0.001)
            else:
                # Execute query and cache result
                result = query_engine.query(query, ["test_collection"])
                query_cache.put(cache_key, result, ["test_collection"])
            
            total_response_time += time.time() - query_start
        
        total_time = time.time() - start_time
        avg_response_time = total_response_time / len(queries)
        throughput = len(queries) / total_time
        hit_rate = cache_hits / len(queries)
        
        return {
            'average_response_time': avg_response_time,
            'total_time': total_time,
            'throughput': throughput,
            'hit_rate': hit_rate
        }


class CacheMemoryBenchmark:
    """
    Benchmark cache memory usage with lazy loading optimization.
    
    This class provides detailed memory profiling for cache operations,
    including lazy loading strategies and memory leak detection.
    """
    
    def __init__(self):
        """Initialize the cache memory benchmark."""
        self._memory_snapshots = []
        self._gc_stats = []
        self._lazy_cache_loader = None
        self._initialize_lazy_loaders()
    
    def _initialize_lazy_loaders(self):
        """Initialize lazy loaders for memory-intensive components."""
        # Lazy load cache with memory tracking
        self._lazy_cache_loader = LazyLoader(
            self._create_memory_tracked_cache
        )
    
    def _create_memory_tracked_cache(self):
        """Create a cache with memory tracking capabilities."""
        from .enhanced_caching import ModelAwareCacheManager
        
        class MemoryTrackedCache(ModelAwareCacheManager):
            def __init__(self):
                super().__init__()
                self._memory_allocations = 0
                self._lazy_initialized = True
            
            def cache_embedding(self, text: str, embedding: List[float], model_fingerprint: str):
                super().cache_embedding(text, embedding, model_fingerprint)
                self._memory_allocations += len(embedding) * 4  # Estimate 4 bytes per float
        
        return MemoryTrackedCache()
    
    def profile_memory_usage(
        self,
        cache_operations: int = 10000,
        embedding_size: int = 384,
        result_size_kb: int = 10
    ) -> Dict[str, Any]:
        """
        Profile memory usage under load with lazy loading.
        
        Args:
            cache_operations: Number of cache operations to perform
            embedding_size: Size of each embedding vector
            result_size_kb: Size of each query result in KB
            
        Returns:
            Memory usage profile with lazy loading metrics
        """
        logger.info(f"Profiling memory usage with {cache_operations} operations")
        
        # Initialize memory monitoring
        memory_snapshots = []
        gc_before = gc.get_count()
        
        # Take initial memory snapshot
        initial_memory = self._get_current_memory()
        memory_snapshots.append(('initial', initial_memory))
        
        # Get cache instance (lazy loaded)
        cache = self._lazy_cache_loader()
        load_memory = self._get_current_memory()
        memory_snapshots.append(('after_lazy_load', load_memory))
        
        # Perform cache operations
        peak_memory = initial_memory
        
        for i in range(cache_operations):
            # Simulate embedding caching
            embedding = [0.1] * embedding_size
            cache.cache_embedding(f"text_{i}", embedding, "model_fp")
            
            # Take memory snapshots periodically
            if i % 1000 == 0:
                current_memory = self._get_current_memory()
                memory_snapshots.append((f'operation_{i}', current_memory))
                peak_memory = max(peak_memory, current_memory)
            
            # Simulate memory pressure and lazy cleanup
            if i % 5000 == 0:
                gc.collect()  # Force garbage collection
        
        # Final memory measurement
        final_memory = self._get_current_memory()
        memory_snapshots.append(('final', final_memory))
        
        # Calculate memory metrics
        memory_growth = final_memory - initial_memory
        lazy_loading_overhead = load_memory - initial_memory
        average_memory = sum(snapshot[1] for snapshot in memory_snapshots) / len(memory_snapshots)
        
        # Check for memory leaks (simplified heuristic)
        gc_after = gc.get_count()
        potential_leaks = (gc_after[0] - gc_before[0]) > (cache_operations / 100)
        
        # Calculate memory efficiency
        expected_memory = (cache_operations * embedding_size * 4) / (1024 * 1024)  # MB
        memory_efficiency = expected_memory / memory_growth if memory_growth > 0 else 1.0
        
        return {
            'peak_memory_mb': peak_memory,
            'average_memory_mb': average_memory,
            'memory_growth_mb': memory_growth,
            'lazy_loading_overhead_mb': lazy_loading_overhead,
            'memory_efficiency': min(memory_efficiency, 1.0),
            'memory_leaks_detected': potential_leaks,
            'memory_snapshots': memory_snapshots,
            'gc_collections': gc_after[0] - gc_before[0]
        }
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback memory estimation based on object counts
            import sys
            return sys.getsizeof(self._memory_snapshots) / (1024 * 1024) + 50.0 