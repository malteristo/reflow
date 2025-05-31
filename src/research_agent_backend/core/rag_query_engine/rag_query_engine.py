"""
RAG Query Engine - Core query processing and orchestration.

This module implements the main RAG (Retrieval-Augmented Generation) query engine
that orchestrates the complete query processing pipeline including context parsing,
embedding generation, vector search, metadata filtering, re-ranking, and result formatting.

Implements FR-RQ-005, FR-RQ-008: Core query processing and re-ranking pipeline.
"""

import logging
from typing import Dict, List, Any, Optional

from .constants import DEFAULT_TOP_K, DEFAULT_DISTANCE_THRESHOLD
from .query_context import QueryContext, QueryIntent, ContextualFilter
from .query_parsing import QueryParser, QueryEnhancer
from .feedback_generation import FeedbackGenerator

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """
    Main RAG Query Engine for processing and orchestrating queries.
    
    Coordinates the complete RAG pipeline from query parsing through
    result formatting, integrating with existing QueryManager, embedding
    services, and re-ranking components.
    """
    
    def __init__(self, query_manager, embedding_service, reranker):
        """Initialize RAG Query Engine with required components."""
        self.query_manager = query_manager
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.logger = logging.getLogger(__name__)
    
    def parse_query_context(self, query: str) -> QueryContext:
        """
        Parse a natural language query to extract context and intent.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryContext with parsed intent, terms, filters, and preferences
            
        Raises:
            ValueError: If query is empty or None
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        # Main parsing pipeline using utility classes
        intent = QueryIntent.classify(query)
        key_terms = QueryParser.extract_key_terms(query, intent)
        filters = ContextualFilter.extract_from_text(query)
        preferences = QueryParser.extract_preferences(query, intent)
        entities = QueryParser.extract_entities(query)
        temporal_context = QueryParser.extract_temporal_context(query)
        
        return QueryContext(
            original_query=query,
            intent=intent,
            key_terms=key_terms,
            filters=filters,
            preferences=preferences,
            entities=entities,
            temporal_context=temporal_context
        )
    
    def generate_query_embedding(self, query_context: QueryContext) -> List[float]:
        """
        Generate vector embedding for a query context.
        
        Args:
            query_context: Parsed query context with intent, terms, and metadata
            
        Returns:
            List of floats representing the query embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Enhance the query with context for better embedding
            enhanced_query = QueryEnhancer.enhance_query_for_embedding(query_context)
            
            # Generate embedding using the embedding service
            embedding = self.embedding_service.generate_embeddings(enhanced_query)
            
            self.logger.debug(f"Generated embedding of dimension {len(embedding)} for query: {query_context.original_query}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for query '{query_context.original_query}': {e}")
            raise Exception(f"Failed to generate query embedding: {e}")
    
    def execute_vector_search(
        self, 
        query_embedding: List[float], 
        collections: List[str],
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = DEFAULT_DISTANCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search using the query embedding.
        
        Args:
            query_embedding: Vector representation of the query
            collections: List of collection names to search in
            top_k: Maximum number of results to return (default: 20)
            filters: Optional metadata filters to apply
            distance_threshold: Optional maximum distance for results
            
        Returns:
            List of search results with metadata and relevance scores
            
        Raises:
            Exception: If vector search execution fails
        """
        try:
            self.logger.debug(f"Executing vector search across {len(collections)} collections with top_k={top_k}")
            
            # Prepare search parameters for QueryManager
            search_params = self._prepare_search_parameters(collections, top_k, filters)
            
            # Execute search through QueryManager
            results = self.query_manager.query(query_embedding, **search_params)
            
            # Apply post-processing filters and limits
            processed_results = self._post_process_search_results(results, distance_threshold, top_k)
            
            self.logger.debug(f"Vector search returned {len(processed_results)} results")
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Vector search execution failed: {e}")
            raise
    
    def apply_metadata_filters(
        self, 
        candidates: List[Dict[str, Any]], 
        filters: Optional[List[ContextualFilter]]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata-based filters to candidate search results.
        
        Args:
            candidates: List of candidate results with metadata
            filters: List of ContextualFilter objects to apply
            
        Returns:
            Filtered list of results that match all filter criteria
            
        Raises:
            Exception: If filtering operation fails
        """
        try:
            # Return all candidates if no filters provided
            if not filters:
                return candidates
            
            # Return empty list if no candidates
            if not candidates:
                return []
            
            filtered_results = []
            
            for candidate in candidates:
                if self._candidate_matches_filters(candidate, filters):
                    filtered_results.append(candidate)
            
            self.logger.debug(f"Metadata filtering: {len(candidates)} -> {len(filtered_results)} results")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Metadata filtering failed: {e}")
            raise
    
    def apply_reranking(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply re-ranking to candidate search results using cross-encoder model.
        
        Args:
            query: Original search query for semantic relevance scoring
            candidates: List of candidate results from vector search and filtering
            top_n: Maximum number of results to return after re-ranking
            
        Returns:
            List of re-ranked results in dictionary format with enhanced scoring
            
        Raises:
            Exception: If re-ranking fails completely (falls back to original ranking)
        """
        try:
            # Handle empty candidates early
            if not candidates:
                self.logger.debug("No candidates provided for re-ranking")
                return []
            
            self.logger.debug(f"Re-ranking {len(candidates)} candidates with query: {query[:50]}...")
            
            # Convert dictionary candidates to SearchResult objects for reranker
            search_result_candidates = self._convert_to_search_results(candidates)
            
            # Use injected reranker to score and rank results
            ranked_results = self.reranker.rerank_results(
                query=query,
                candidates=search_result_candidates,
                top_n=top_n,
                collect_metrics=True
            )
            
            # Convert back to dictionary format for pipeline compatibility
            reranked_dict_results = self._convert_ranked_results_to_dict(ranked_results, candidates)
            
            self.logger.debug(f"Re-ranking completed: {len(ranked_results)} results returned")
            
            return reranked_dict_results
            
        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            # Graceful fallback: return original results with neutral rerank scores
            return self._create_fallback_reranked_results(candidates, top_n)
    
    def generate_result_feedback(
        self, 
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]], 
        top_k: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feedback for search results.
        
        Args:
            query_context: The parsed query context with intent and filters
            search_results: List of search results with metadata
            top_k: Number of top results requested
            
        Returns:
            Dictionary containing feedback components including explanations,
            suggestions, and metrics
        """
        return FeedbackGenerator.generate_result_feedback(query_context, search_results, top_k)
    
    def collect_user_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and process user feedback on search results.
        
        Args:
            feedback_data: User feedback including ratings and comments
            
        Returns:
            Dictionary with feedback processing results
        """
        # Basic feedback collection interface
        return {
            "success": True,
            "feedback_id": f"feedback_{hash(str(feedback_data))}"
        }
    
    # Private helper methods for core pipeline operations
    
    def _prepare_search_parameters(
        self, 
        collections: List[str], 
        top_k: int, 
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare search parameters for QueryManager."""
        search_params = {
            "collections": collections,
            "top_k": top_k
        }
        
        # Add filters if provided
        if filters:
            search_params["filters"] = filters
        
        return search_params
    
    def _post_process_search_results(
        self, 
        results: List[Dict[str, Any]], 
        distance_threshold: Optional[float], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Apply post-processing to search results."""
        # Apply distance threshold filtering if specified
        if distance_threshold is not None:
            results = [
                result for result in results 
                if result.get("distance", float('inf')) <= distance_threshold
            ]
        
        # Ensure we don't exceed top_k limit
        if len(results) > top_k:
            results = results[:top_k]
        
        return results
    
    def _candidate_matches_filters(
        self, 
        candidate: Dict[str, Any], 
        filters: List[ContextualFilter]
    ) -> bool:
        """Check if a candidate matches all provided filters (AND logic)."""
        metadata = candidate.get("metadata", {})
        
        for filter_obj in filters:
            if not self._evaluate_single_filter(metadata, filter_obj):
                return False  # Fail fast on first non-match (AND logic)
        
        return True  # All filters passed
    
    def _evaluate_single_filter(
        self, 
        metadata: Dict[str, Any], 
        filter_obj: ContextualFilter
    ) -> bool:
        """Evaluate a single filter against metadata."""
        try:
            # Get field value from metadata (supports nested fields)
            field_value = self._get_nested_field_value(metadata, filter_obj.field)
            
            # Handle missing fields
            if field_value is None:
                return False
            
            # Apply operator-specific logic
            return self._apply_filter_operator(field_value, filter_obj.value, filter_obj.operator)
            
        except Exception as e:
            self.logger.warning(f"Error evaluating filter {filter_obj.field}: {e}")
            return False
    
    def _get_nested_field_value(self, metadata: Dict[str, Any], field_path: str) -> Any:
        """Get value from metadata, supporting nested field access with dot notation."""
        try:
            # Split field path by dots for nested access
            path_parts = field_path.split('.')
            current_value = metadata
            
            for part in path_parts:
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]
                else:
                    return None  # Field not found
            
            return current_value
            
        except Exception:
            return None
    
    def _apply_filter_operator(self, field_value: Any, filter_value: Any, operator: str) -> bool:
        """Apply operator-specific comparison logic."""
        try:
            if operator == "equals":
                return self._equals_comparison(field_value, filter_value)
            elif operator == "contains":
                return self._contains_comparison(field_value, filter_value)
            elif operator == "greater_than":
                return self._greater_than_comparison(field_value, filter_value)
            elif operator == "less_than":
                return self._less_than_comparison(field_value, filter_value)
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error applying operator {operator}: {e}")
            return False
    
    def _equals_comparison(self, field_value: Any, filter_value: Any) -> bool:
        """Handle equals comparison with case-insensitive string matching."""
        if isinstance(field_value, str) and isinstance(filter_value, str):
            return field_value.lower() == filter_value.lower()
        return field_value == filter_value
    
    def _contains_comparison(self, field_value: Any, filter_value: Any) -> bool:
        """Handle contains comparison for strings and arrays."""
        if isinstance(field_value, list):
            # For arrays, check if filter_value is in the array
            return filter_value in field_value
        elif isinstance(field_value, str) and isinstance(filter_value, str):
            # For strings, check substring match (case-insensitive)
            return filter_value.lower() in field_value.lower()
        return False
    
    def _greater_than_comparison(self, field_value: Any, filter_value: Any) -> bool:
        """Handle greater than comparison for numeric and date values."""
        try:
            # Handle date comparisons
            if isinstance(field_value, str) and isinstance(filter_value, str):
                # Try to parse as dates
                try:
                    from datetime import datetime
                    field_date = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                    filter_date = datetime.fromisoformat(filter_value.replace('Z', '+00:00'))
                    return field_date > filter_date
                except ValueError:
                    # Not valid dates, fallback to string comparison
                    return field_value > filter_value
            
            # Handle numeric comparisons
            return float(field_value) > float(filter_value)
            
        except (ValueError, TypeError):
            return False
    
    def _less_than_comparison(self, field_value: Any, filter_value: Any) -> bool:
        """Handle less than comparison for numeric and date values."""
        try:
            # Handle date comparisons
            if isinstance(field_value, str) and isinstance(filter_value, str):
                # Try to parse as dates
                try:
                    from datetime import datetime
                    field_date = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                    filter_date = datetime.fromisoformat(filter_value.replace('Z', '+00:00'))
                    return field_date < filter_date
                except ValueError:
                    # Not valid dates, fallback to string comparison
                    return field_value < filter_value
            
            # Handle numeric comparisons
            return float(field_value) < float(filter_value)
            
        except (ValueError, TypeError):
            return False
    
    # Re-ranking helper methods
    
    def _convert_to_search_results(self, candidates: List[Dict[str, Any]]) -> List:
        """Convert dictionary candidates to SearchResult objects for reranker compatibility."""
        from research_agent_backend.core.integration_pipeline.models import SearchResult
        
        search_results = []
        
        for candidate in candidates:
            # Calculate relevance score from distance (inverse relationship)
            distance = candidate.get("distance", 0.5)
            relevance_score = max(0.0, min(1.0, 1.0 - distance))
            
            search_result = SearchResult(
                content=candidate.get("content", ""),
                metadata=candidate.get("metadata", {}),
                relevance_score=relevance_score,
                document_id=candidate.get("id"),
                chunk_id=candidate.get("chunk_id", f"{candidate.get('id', 'unknown')}_chunk")
            )
            
            search_results.append(search_result)
        
        return search_results
    
    def _convert_ranked_results_to_dict(
        self, 
        ranked_results: List, 
        original_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert RankedResult objects back to dictionary format for pipeline compatibility."""
        dict_results = []
        
        # Create mapping for quick lookup of original data
        original_by_id = {
            candidate.get("id"): candidate 
            for candidate in original_candidates
        }
        
        for ranked_result in ranked_results:
            # Get original result data
            original_result = ranked_result.original_result
            document_id = original_result.document_id
            original_candidate = original_by_id.get(document_id, {})
            
            # Calculate score improvement
            score_improvement = ranked_result.rerank_score - ranked_result.original_score
            
            # Build enhanced result dictionary
            enhanced_result = {
                # Preserve original fields
                "id": document_id,
                "content": original_result.content,
                "metadata": original_result.metadata.copy(),
                
                # Add re-ranking information
                "rerank_score": ranked_result.rerank_score,
                "original_score": ranked_result.original_score,
                "rank": ranked_result.rank,
                "score_improvement": score_improvement,
                
                # Preserve original distance information if available
                "distance": original_candidate.get("distance"),
                "original_distance": original_candidate.get("distance"),
                
                # Add re-ranking metadata
                "reranking_metadata": ranked_result.metadata or {}
            }
            
            # Remove None values for cleaner output
            enhanced_result = {k: v for k, v in enhanced_result.items() if v is not None}
            
            dict_results.append(enhanced_result)
        
        return dict_results
    
    def _create_fallback_reranked_results(
        self, 
        candidates: List[Dict[str, Any]], 
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create fallback re-ranked results when re-ranking fails."""
        fallback_results = []
        
        # Sort by original relevance (inverse of distance)
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("distance", 0.5)
        )
        
        # Apply top_n limit if specified
        if top_n is not None:
            sorted_candidates = sorted_candidates[:top_n]
        
        # Add neutral re-ranking information
        for i, candidate in enumerate(sorted_candidates):
            distance = candidate.get("distance", 0.5)
            original_score = max(0.0, min(1.0, 1.0 - distance))
            
            fallback_result = candidate.copy()
            fallback_result.update({
                "rerank_score": 0.5,  # Neutral score
                "original_score": original_score,
                "rank": i + 1,
                "score_improvement": 0.5 - original_score,
                "original_distance": distance,
                "reranking_metadata": {"fallback": True, "error": "Re-ranking service unavailable"}
            })
            
            fallback_results.append(fallback_result)
        
        return fallback_results 