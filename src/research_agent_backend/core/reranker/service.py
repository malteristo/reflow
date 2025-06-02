"""
Cross-encoder re-ranking service for improving search result precision.

Implements cross-encoder model-based re-ranking to improve the precision
of vector search results through semantic similarity scoring, with enhanced
features for keyword highlighting, source attribution, and relevance analysis.

Implements FR-RQ-005, FR-RQ-008: Core query processing pipeline with re-ranking.
"""

import time
import hashlib
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from functools import lru_cache
from collections import OrderedDict

import numpy as np
from sentence_transformers import CrossEncoder

from .config import RerankerConfig
from .models import RankedResult
from .utils import KeywordHighlighter, SourceAttributionExtractor, RelevanceAnalyzer
from ..integration_pipeline.models import SearchResult
from ...utils.config import ConfigManager


logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation for query-document pairs."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[float]:
        """Get cached score, updating LRU order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: float) -> None:
        """Cache score, evicting least recently used if needed."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class RerankerService:
    """
    Cross-encoder re-ranking service for improving search precision.
    
    Uses cross-encoder models to re-rank vector search results based on
    semantic similarity, providing more accurate relevance scoring with
    enhanced features for keyword highlighting, source attribution, and
    relevance confidence analysis.
    
    Implements FR-RQ-005: Core query processing pipeline with re-ranking.
    Implements FR-RQ-008: Enhanced result presentation with highlighting and attribution.
    """
    
    def __init__(
        self, 
        config: Optional[RerankerConfig] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize re-ranking service.
        
        Args:
            config: Direct configuration object
            config_manager: Configuration manager for loading settings
        """
        # Initialize configuration
        if config is not None:
            self.config = config
        elif config_manager is not None:
            self.config = self._load_config_from_manager(config_manager)
        else:
            self.config = RerankerConfig()
        
        # Initialize model
        try:
            self.model = CrossEncoder(self.config.model_name)
            if hasattr(self.model, 'to') and self.config.device != 'cpu':
                self.model.to(self.config.device)
            logger.info(f"Loaded cross-encoder model: {self.config.model_name} on {self.config.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
        
        # Initialize enhanced cache if enabled
        self.cache = LRUCache(self.config.cache_size) if self.config.enable_caching else None
        
        # Initialize enhanced utilities for FR-RQ-008
        self.keyword_highlighter = KeywordHighlighter()
        self.source_extractor = SourceAttributionExtractor()
        self.relevance_analyzer = RelevanceAnalyzer(self.keyword_highlighter)
        
        # Enhanced metrics tracking
        self.metrics = {
            'total_queries': 0,
            'total_candidates_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time_ms': 0.0,
            'average_batch_size': 0.0,
            'last_operation': {}
        }
        
        logger.info(f"RerankerService initialized with config: {self.config.to_dict()}")
    
    def _load_config_from_manager(self, config_manager: ConfigManager) -> RerankerConfig:
        """Load configuration from ConfigManager with enhanced defaults."""
        return RerankerConfig(
            model_name=config_manager.get('reranker.model_name', 'cross-encoder/ms-marco-MiniLM-L6-v2'),
            batch_size=config_manager.get('reranker.batch_size', 32),
            max_length=config_manager.get('reranker.max_length', 512),
            device=config_manager.get('reranker.device', 'cpu'),
            cache_size=config_manager.get('reranker.cache_size', 1000),
            enable_caching=config_manager.get('reranker.enable_caching', True),
            temperature=config_manager.get('reranker.temperature', 1.0),
            top_k_candidates=config_manager.get('reranker.top_k_candidates', 100),
            use_fp16=config_manager.get('reranker.use_fp16', False),
            num_threads=config_manager.get('reranker.num_threads', None)
        )
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name
    
    @property
    def device(self) -> str:
        """Get device."""
        return self.config.device
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.config.batch_size
    
    @property
    def max_length(self) -> int:
        """Get max length."""
        return self.config.max_length
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair with enhanced caching.
        
        Args:
            query: Search query
            document: Document content
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(query, document)
            cached_score = self.cache.get(cache_key)
            if cached_score is not None:
                self.metrics['cache_hits'] += 1
                return cached_score
            self.metrics['cache_misses'] += 1
        
        # Score with model
        start_time = time.time()
        try:
            pair = [query, document]
            scores = self.model.predict([pair])
            normalized_score = self._normalize_scores(scores)[0]
            
            # Cache result
            if self.cache is not None:
                cache_key = self._get_cache_key(query, document)
                self.cache.put(cache_key, float(normalized_score))
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics['total_processing_time_ms'] += processing_time
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"Error scoring pair: {e}")
            # Return neutral score on error
            return 0.5
    
    def rerank_single(self, query: str, search_result: SearchResult, include_enhancements: bool = True) -> RankedResult:
        """
        Re-rank a single search result with enhanced features.
        
        Args:
            query: Search query
            search_result: SearchResult to re-rank
            include_enhancements: Whether to include keyword highlighting and attribution
            
        Returns:
            RankedResult with enhanced features
        """
        # Get re-ranking score
        rerank_score = self.score_pair(query, search_result.content)
        
        # Create base ranked result
        ranked_result = RankedResult(
            original_result=search_result,
            rerank_score=rerank_score,
            original_score=search_result.relevance_score,
            rank=1,
            metadata={
                'processing_method': 'single',
                'model_name': self.config.model_name,
                'timestamp': time.time()
            }
        )
        
        # Add enhanced features if requested (FR-RQ-008)
        if include_enhancements:
            # Extract keywords and add highlighting
            keywords = self.keyword_highlighter.extract_query_keywords(query)
            highlighted_content = self.keyword_highlighter.highlight_keywords(
                search_result.content, keywords
            )
            ranked_result.highlighted_content = highlighted_content
            
            # Extract source attribution
            source_attribution = self.source_extractor.extract_attribution(search_result)
            ranked_result.source_attribution = source_attribution
            
            # Analyze relevance indicators
            relevance_indicators = self.relevance_analyzer.analyze_relevance(
                query, search_result, rerank_score
            )
            ranked_result.relevance_indicators = relevance_indicators
        
        return ranked_result
    
    def rerank_results(
        self, 
        query: str, 
        candidates: List[SearchResult], 
        top_n: Optional[int] = None,
        collect_metrics: bool = False,
        include_enhancements: bool = True
    ) -> List[RankedResult]:
        """
        Re-rank multiple search results with enhanced batch processing and features.
        
        Args:
            query: Search query
            candidates: List of search results to re-rank
            top_n: Number of top results to return
            collect_metrics: Whether to collect performance metrics
            include_enhancements: Whether to include enhanced features (FR-RQ-008)
            
        Returns:
            List of RankedResult sorted by re-ranking score with enhanced features
        """
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Limit candidates if configured
        working_candidates = candidates[:self.config.top_k_candidates] if len(candidates) > self.config.top_k_candidates else candidates
        
        # Enhanced batch processing
        try:
            scores = self._batch_score_pairs(query, working_candidates)
            
            # Pre-extract keywords for batch processing efficiency
            keywords = None
            if include_enhancements:
                keywords = self.keyword_highlighter.extract_query_keywords(query)
            
            # Create ranked results with enhanced features
            ranked_results = []
            for i, (result, score) in enumerate(zip(working_candidates, scores)):
                # Create base ranked result
                ranked_result = RankedResult(
                    original_result=result,
                    rerank_score=float(score),
                    original_score=result.relevance_score,
                    rank=i + 1,  # Will be updated after sorting
                    metadata={
                        'processing_method': 'batch',
                        'model_name': self.config.model_name,
                        'batch_size': len(working_candidates),
                        'original_rank': i + 1,
                        'timestamp': time.time()
                    }
                )
                
                # Add enhanced features if requested (FR-RQ-008)
                if include_enhancements:
                    # Add keyword highlighting
                    highlighted_content = self.keyword_highlighter.highlight_keywords(
                        result.content, keywords
                    )
                    ranked_result.highlighted_content = highlighted_content
                    
                    # Extract source attribution
                    source_attribution = self.source_extractor.extract_attribution(result)
                    ranked_result.source_attribution = source_attribution
                    
                    # Analyze relevance indicators
                    relevance_indicators = self.relevance_analyzer.analyze_relevance(
                        query, result, float(score)
                    )
                    ranked_result.relevance_indicators = relevance_indicators
                
                ranked_results.append(ranked_result)
            
            # Sort by rerank_score (descending)
            ranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Update ranks and apply top_n limit
            for i, result in enumerate(ranked_results):
                result.rank = i + 1
            
            if top_n is not None:
                ranked_results = ranked_results[:top_n]
            
            # Enhanced metrics collection
            processing_time = (time.time() - start_time) * 1000
            batch_count = max(1, len(working_candidates) // self.config.batch_size)
            
            # Update global metrics
            self.metrics['total_queries'] += 1
            self.metrics['total_candidates_processed'] += len(working_candidates)
            self.metrics['total_processing_time_ms'] += processing_time
            
            if collect_metrics or True:  # Always collect basic metrics
                self.metrics['last_operation'] = {
                    'processing_time_ms': processing_time,
                    'batch_count': batch_count,
                    'candidates_processed': len(working_candidates),
                    'results_returned': len(ranked_results),
                    'cache_hit_rate': self._get_cache_hit_rate(),
                    'average_score_improvement': self._calculate_score_improvement(ranked_results),
                    'enhancements_enabled': include_enhancements,
                    'keywords_extracted': len(keywords) if keywords else 0
                }
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in batch re-ranking: {e}")
            # Return original results with neutral scores on error
            return [
                RankedResult(
                    original_result=result,
                    rerank_score=0.5,
                    original_score=result.relevance_score,
                    rank=i + 1,
                    metadata={'error': str(e)}
                )
                for i, result in enumerate(working_candidates[:top_n] if top_n else working_candidates)
            ]
    
    def _batch_score_pairs(self, query: str, candidates: List[SearchResult]) -> List[float]:
        """
        Score multiple query-document pairs efficiently in batches.
        
        Args:
            query: Search query
            candidates: List of search results to score
            
        Returns:
            List of normalized scores
        """
        all_scores = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            
            # Check cache for batch items
            batch_pairs = []
            batch_indices = []
            cached_scores = [None] * len(batch_candidates)
            
            for j, result in enumerate(batch_candidates):
                if self.cache is not None:
                    cache_key = self._get_cache_key(query, result.content)
                    cached_score = self.cache.get(cache_key)
                    if cached_score is not None:
                        cached_scores[j] = cached_score
                        self.metrics['cache_hits'] += 1
                        continue
                    self.metrics['cache_misses'] += 1
                
                batch_pairs.append([query, result.content])
                batch_indices.append(j)
            
            # Score uncached pairs
            if batch_pairs:
                try:
                    batch_scores = self.model.predict(batch_pairs)
                    normalized_batch_scores = self._normalize_scores(batch_scores)
                    
                    # Fill in batch results and cache them
                    for idx, (batch_idx, score) in enumerate(zip(batch_indices, normalized_batch_scores)):
                        cached_scores[batch_idx] = float(score)
                        
                        # Cache the result
                        if self.cache is not None:
                            cache_key = self._get_cache_key(query, batch_candidates[batch_idx].content)
                            self.cache.put(cache_key, float(score))
                            
                except Exception as e:
                    logger.warning(f"Error in batch scoring: {e}")
                    # Fill remaining with neutral scores
                    for batch_idx in batch_indices:
                        if cached_scores[batch_idx] is None:
                            cached_scores[batch_idx] = 0.5
            
            # Fill any remaining None values (shouldn't happen, but safety check)
            for j in range(len(cached_scores)):
                if cached_scores[j] is None:
                    cached_scores[j] = 0.5
            
            all_scores.extend(cached_scores)
        
        return all_scores
    
    def enhance_search_results(
        self, 
        query: str, 
        search_results: List[SearchResult]
    ) -> List[RankedResult]:
        """
        Enhance search results with re-ranking scores.
        
        Args:
            query: Search query
            search_results: Original search results
            
        Returns:
            Enhanced results with re-ranking information
        """
        return self.rerank_results(query, search_results, collect_metrics=True)
    
    def get_last_operation_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last re-ranking operation."""
        return self.metrics['last_operation'].copy()
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_queries'] > 0:
            metrics['average_processing_time_ms'] = metrics['total_processing_time_ms'] / metrics['total_queries']
            metrics['average_candidates_per_query'] = metrics['total_candidates_processed'] / metrics['total_queries']
        
        if self.cache:
            metrics['cache_size'] = self.cache.size()
            metrics['cache_utilization'] = self.cache.size() / self.config.cache_size
        
        return metrics
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to 0-1 range with temperature scaling.
        
        Args:
            scores: Raw model scores
            
        Returns:
            Normalized scores between 0.0 and 1.0
        """
        # Apply temperature scaling then sigmoid
        scaled_scores = scores / self.config.temperature
        normalized = 1 / (1 + np.exp(-scaled_scores))
        return normalized
    
    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        content = f"{query}|||{document[:1000]}"  # Limit document length for key
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_requests == 0:
            return 0.0
        return self.metrics['cache_hits'] / total_requests
    
    def _calculate_score_improvement(self, ranked_results: List[RankedResult]) -> float:
        """Calculate average score improvement from re-ranking."""
        if not ranked_results:
            return 0.0
        
        improvements = [result.get_score_improvement() for result in ranked_results]
        return sum(improvements) / len(improvements)
    
    def clear_cache(self) -> None:
        """Clear the scoring cache and reset cache metrics."""
        if self.cache is not None:
            self.cache.clear()
            self.metrics['cache_hits'] = 0
            self.metrics['cache_misses'] = 0
            logger.info("Scoring cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if self.cache is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'size': self.cache.size(),
            'max_size': self.config.cache_size,
            'utilization': self.cache.size() / self.config.cache_size,
            'hit_rate': self._get_cache_hit_rate(),
            'total_hits': self.metrics['cache_hits'],
            'total_misses': self.metrics['cache_misses']
        }
    
    def warmup_cache(self, queries: List[str], documents: List[str]) -> None:
        """
        Warm up the cache with common query-document pairs.
        
        Args:
            queries: List of common queries
            documents: List of common documents
        """
        if not self.cache:
            return
        
        logger.info(f"Warming up cache with {len(queries)} queries and {len(documents)} documents")
        start_time = time.time()
        
        for query in queries:
            for document in documents:
                self.score_pair(query, document)
        
        warmup_time = time.time() - start_time
        logger.info(f"Cache warmup completed in {warmup_time:.2f}s. Cache size: {self.cache.size()}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.metrics = {
            'total_queries': 0,
            'total_candidates_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time_ms': 0.0,
            'average_batch_size': 0.0,
            'last_operation': {}
        }
        logger.info("Metrics reset") 