"""
Enhanced search engine with integrated re-ranking capabilities.

Provides a wrapper around IntegratedSearchEngine that adds re-ranking
functionality for improved search result precision and relevance.
"""

import logging
from typing import List, Optional, Dict, Any

from .processor import RerankerPipelineProcessor
from .config import PipelineConfig
from ..config import RerankerConfig
from ...integration_pipeline import IntegratedSearchEngine
from ...integration_pipeline.models import SearchResult
from ....utils.config import ConfigManager


logger = logging.getLogger(__name__)


class EnhancedSearchEngine:
    """
    Enhanced search engine with integrated re-ranking capabilities.
    
    Wraps IntegratedSearchEngine to provide re-ranking functionality
    for improved search result precision and relevance scoring.
    
    Implements Task 6.4: Integration with Retrieval Pipeline
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        search_engine_config: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        reranker_config: Optional[RerankerConfig] = None
    ):
        """
        Initialize enhanced search engine.
        
        Args:
            config_manager: Configuration manager for loading settings
            search_engine_config: Configuration for underlying search engine
            pipeline_config: Pipeline configuration for re-ranking
            reranker_config: Re-ranking service configuration
        """
        self.config_manager = config_manager
        
        # Initialize base search engine
        if search_engine_config is not None:
            self.search_engine = IntegratedSearchEngine(search_engine_config)
        elif config_manager is not None:
            # Extract search engine config from manager
            search_config = self._extract_search_config(config_manager)
            self.search_engine = IntegratedSearchEngine(search_config)
        else:
            # Use default configuration
            self.search_engine = IntegratedSearchEngine({})
        
        # Initialize pipeline processor
        self.pipeline_processor = RerankerPipelineProcessor(
            config_manager=config_manager,
            pipeline_config=pipeline_config,
            reranker_config=reranker_config
        )
        
        logger.info("EnhancedSearchEngine initialized with re-ranking capabilities")
    
    def _extract_search_config(self, config_manager: ConfigManager) -> Dict[str, Any]:
        """Extract search engine configuration from ConfigManager."""
        return {
            "search": {
                "default_top_k": config_manager.get('search.default_top_k', 10),
                "min_relevance": config_manager.get('search.min_relevance', 0.1)
            },
            "vector_store": {
                "type": config_manager.get('vector_store.type', 'chromadb'),
                "settings": config_manager.get('vector_store.settings', {})
            }
        }
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform search without re-ranking (standard IntegratedSearchEngine behavior).
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects from base search engine
        """
        return await self.search_engine.search(query=query, top_k=top_k, filters=filters)
    
    def search_with_reranking(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_n: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform search with integrated re-ranking for enhanced precision.
        
        Args:
            query: Search query
            top_k: Number of candidates to retrieve for re-ranking
            rerank_top_n: Number of results to return after re-ranking
            filters: Optional metadata filters
            rerank_threshold: Minimum re-ranking score threshold
            
        Returns:
            List of re-ranked SearchResult objects
        """
        try:
            # Perform initial search with higher top_k for re-ranking candidates
            import asyncio
            search_results = asyncio.run(self.search_engine.search(query=query, top_k=top_k, filters=filters))
            
            # Apply re-ranking pipeline
            pipeline_result = self.pipeline_processor.process_search_results(
                query=query,
                search_results=search_results
            )
            
            # Extract final results
            if isinstance(pipeline_result.reranked_results[0] if pipeline_result.reranked_results else None, SearchResult):
                return pipeline_result.reranked_results
            else:
                # Convert RankedResult back to SearchResult for compatibility
                converted_results = []
                for ranked_result in pipeline_result.reranked_results:
                    if hasattr(ranked_result, 'original_result'):
                        # Update relevance score with re-ranking score
                        result = ranked_result.original_result
                        result.relevance_score = ranked_result.rerank_score
                        converted_results.append(result)
                    else:
                        converted_results.append(ranked_result)
                return converted_results
                
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to standard search
            import asyncio
            return asyncio.run(self.search_engine.search(query=query, top_k=min(top_k, rerank_top_n or 5), filters=filters))
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get current pipeline configuration."""
        return self.pipeline_processor.get_current_config()
    
    def update_pipeline_config(self, config: PipelineConfig) -> None:
        """Update pipeline configuration."""
        self.pipeline_processor.update_config(config)
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        return self.pipeline_processor.get_reranker_metrics()
    
    def reset_pipeline_metrics(self) -> None:
        """Reset pipeline metrics."""
        self.pipeline_processor.reset_metrics()
    
    def warmup_pipeline_cache(self, queries: List[str], documents: List[str]) -> None:
        """Warm up the pipeline cache."""
        self.pipeline_processor.warmup_cache(queries, documents)
    
    def clear_pipeline_cache(self) -> None:
        """Clear the pipeline cache."""
        self.pipeline_processor.clear_cache() 