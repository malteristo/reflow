"""
RAG Query Engine - Core query processing and orchestration.

This module implements the main RAG (Retrieval-Augmented Generation) query engine
that orchestrates the complete query processing pipeline including context parsing,
embedding generation, vector search, metadata filtering, re-ranking, and result formatting.

Implements FR-RQ-005, FR-RQ-008: Core query processing and re-ranking pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
import time

from .constants import DEFAULT_TOP_K, DEFAULT_DISTANCE_THRESHOLD
from .query_context import QueryContext, QueryIntent, ContextualFilter
from .query_parsing import QueryParser, QueryEnhancer
from .feedback_generation import FeedbackGenerator

logger = logging.getLogger(__name__)


class ContentHighlighter:
    """Utility class for highlighting query terms in content."""
    
    @staticmethod
    def highlight_terms(
        content: str,
        key_terms: List[str],
        style: str = "html"
    ) -> str:
        """Highlight query terms in content based on style."""
        if not content or not key_terms:
            return content
        
        import re
        
        # Sort terms by length (longest first) to prioritize phrases over individual words
        sorted_terms = sorted(key_terms, key=len, reverse=True)
        
        # Create a single regex pattern that matches all terms
        # Escape each term and join with | (OR)
        escaped_terms = [re.escape(term) for term in sorted_terms if term.strip()]
        if not escaped_terms:
            return content
        
        pattern = r'\b(?:' + '|'.join(escaped_terms) + r')\b'
        
        def highlight_replacement(match):
            term = match.group(0)
            if style == "html":
                return f"<mark>{term}</mark>"
            elif style == "markdown":
                return f"**{term}**"
            elif style == "cli":
                return f"\033[1;33m{term}\033[0m"  # Yellow text
            else:
                return f"**{term}**"  # Default to markdown
        
        # Apply highlighting using case-insensitive matching
        highlighted = re.sub(pattern, highlight_replacement, content, flags=re.IGNORECASE)
        return highlighted


class SnippetGenerator:
    """Utility class for generating intelligent content snippets."""
    
    @staticmethod
    def generate_snippet(
        content: str,
        key_terms: List[str],
        max_length: int = 200,
        context_window: int = 50
    ) -> str:
        """Generate an intelligent snippet from content focusing on query terms."""
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        import re
        
        # Find positions of all key terms (case-insensitive)
        term_positions = []
        for term in key_terms:
            if not term.strip():
                continue
            pattern = re.escape(term)
            for match in re.finditer(pattern, content, re.IGNORECASE):
                term_positions.append((match.start(), match.end()))
        
        if not term_positions:
            # No terms found, return beginning of content
            return content[:max_length].rsplit(' ', 1)[0] + "..."
        
        # Find the best position to center the snippet for maximum term coverage
        term_positions.sort()
        
        # Calculate snippet start position
        first_term_start = term_positions[0][0]
        last_term_end = term_positions[-1][1]
        
        # Try to center around the term cluster
        cluster_center = (first_term_start + last_term_end) // 2
        snippet_start = max(0, cluster_center - max_length // 2)
        
        # Adjust to word boundaries
        if snippet_start > 0:
            # Find the next word boundary
            next_space = content.find(' ', snippet_start)
            if next_space != -1 and next_space < snippet_start + context_window:
                snippet_start = next_space + 1
        
        snippet_end = min(len(content), snippet_start + max_length)
        
        # Adjust end to word boundary
        if snippet_end < len(content):
            last_space = content.rfind(' ', snippet_start, snippet_end)
            if last_space > snippet_start + max_length // 2:
                snippet_end = last_space
        
        snippet = content[snippet_start:snippet_end]
        
        # Add ellipsis if needed
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."
        
        return snippet


class ResultFormatter:
    """Utility class for formatting query results in different output formats."""
    
    def __init__(self):
        self.highlighter = ContentHighlighter()
        self.snippet_generator = SnippetGenerator()
    
    def format_for_output(
        self,
        formatted_results: Dict[str, Any],
        output_format: str
    ) -> Dict[str, Any]:
        """Format results for specific output type."""
        if output_format == "cli":
            return self._format_for_cli(formatted_results)
        elif output_format == "api":
            return self._format_for_api(formatted_results)
        else:
            return formatted_results  # Default structured format
    
    def _format_for_cli(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for CLI display."""
        cli_results = results.copy()
        
        # Simplify metadata for CLI readability
        if "results" in cli_results:
            for result in cli_results["results"]:
                if "metadata" in result:
                    # Keep only essential metadata for CLI
                    essential_meta = {
                        "source": result["metadata"].get("source", "Unknown"),
                        "type": result["metadata"].get("type", "document"),
                        "collection": result["metadata"].get("collection", "default")
                    }
                    result["metadata"] = essential_meta
        
        # Remove performance metrics for cleaner CLI output
        cli_results.pop("performance_metrics", None)
        
        return cli_results
    
    def _format_for_api(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for API response."""
        api_results = results.copy()
        
        # Ensure all fields are properly serializable
        if "query_info" in api_results:
            query_info = api_results["query_info"]
            if "timestamp" in query_info and hasattr(query_info["timestamp"], "isoformat"):
                query_info["timestamp"] = query_info["timestamp"].isoformat()
        
        # Add API-specific metadata
        api_results["api_version"] = "1.0"
        api_results["response_type"] = "search_results"
        
        return api_results


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
        self.query_parser = QueryParser()
        self.feedback_generator = FeedbackGenerator()
        self.result_formatter = ResultFormatter()
    
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
    
    def format_results(
        self,
        query_context: QueryContext,
        reranked_results: List[Dict[str, Any]],
        feedback: Dict[str, Any],
        output_format: str = "structured",
        format_options: Optional[Dict[str, Any]] = None,
        formatting_options: Optional[Dict[str, Any]] = None,  # Backward compatibility
        include_performance: bool = False  # Backward compatibility
    ) -> Dict[str, Any]:
        """
        Format final ranked results into user-friendly structure.
        
        Args:
            query_context: Original query context
            reranked_results: Results after re-ranking
            feedback: Query feedback and suggestions
            output_format: Output format ("structured", "cli", "api")
            format_options: Additional formatting options
            formatting_options: Legacy parameter name (backward compatibility)
            include_performance: Whether to include performance metrics
            
        Returns:
            Dictionary with formatted results and metadata
        """
        # Handle backward compatibility
        options = format_options or formatting_options or {}
        if include_performance:
            options["include_performance"] = True
        
        try:
            # Process each result with highlighting and snippets
            formatted_results = []
            for result in reranked_results:
                formatted_result = self._format_single_result(
                    result, query_context, options
                )
                formatted_results.append(formatted_result)
            
            # Build comprehensive response structure
            response = {
                "query_info": {
                    "original_query": query_context.original_query,
                    "intent": query_context.intent.value if query_context.intent else "unknown",
                    "key_terms": query_context.key_terms,
                    "timestamp": time.time()
                },
                "results": formatted_results,
                "result_summary": {
                    "total_results": len(formatted_results),
                    "confidence_levels": self._calculate_confidence_distribution(formatted_results),
                    "collections_covered": list(set(
                        r.get("metadata", {}).get("collection", "unknown") 
                        for r in formatted_results
                    ))
                },
                "feedback": feedback,
                "improvement_suggestions": self._generate_improvement_suggestions(
                    query_context, formatted_results
                )
            }
            
            # Add legacy field mappings for backward compatibility
            if output_format == "structured":
                response["summary"] = response["result_summary"]
                response["suggestions"] = response["improvement_suggestions"]
                response["metadata"] = {
                    "format": output_format,
                    "timestamp": response["query_info"]["timestamp"],
                    "processing_info": "Results formatted for structured output"
                }
            elif output_format == "cli":
                response["header"] = f"Search Results for: {query_context.original_query}"
                response["summary"] = response["result_summary"]
            elif output_format == "api":
                response["query"] = response["query_info"]
                response["metadata"] = {"processing_info": "Results formatted for API"}
            
            # Add performance metrics if requested
            if options.get("include_performance", False):
                response["performance_metrics"] = {
                    "processing_time": options.get("processing_time", 0),
                    "total_candidates_evaluated": options.get("total_candidates", 0),
                    "reranking_time": options.get("reranking_time", 0)
                }
            
            # Format for specific output type
            return self.result_formatter.format_for_output(response, output_format)
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            # Return basic fallback structure with backward compatibility
            fallback = {
                "query_info": {"original_query": query_context.original_query},
                "results": reranked_results,
                "error": "Result formatting failed",
                "feedback": feedback
            }
            
            # Add legacy fields for error handling tests
            if output_format == "structured":
                fallback["metadata"] = {"errors": [str(e)]}
                fallback["suggestions"] = []
            
            return fallback
    
    def _format_single_result(
        self,
        result: Dict[str, Any],
        query_context: QueryContext,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format a single result with highlighting and snippets."""
        content = result.get("content", "")
        
        # Generate highlighted snippet
        snippet_length = options.get("snippet_length", 200)
        snippet = self.result_formatter.snippet_generator.generate_snippet(
            content, query_context.key_terms, snippet_length
        )
        
        # Apply highlighting
        highlight_style = options.get("highlight_style", "html")
        highlighted_content = self.result_formatter.highlighter.highlight_terms(
            content, query_context.key_terms, highlight_style
        )
        highlighted_snippet = self.result_formatter.highlighter.highlight_terms(
            snippet, query_context.key_terms, highlight_style
        )
        
        # Enrich metadata
        metadata = result.get("metadata", {})
        enriched_metadata = self._enrich_metadata(metadata, options)
        
        formatted_result = {
            "content": highlighted_content,
            "snippet": highlighted_snippet,
            "metadata": enriched_metadata,
            "relevance_score": result.get("relevance_score", 0.0),
            "confidence": self._calculate_result_confidence(result),
            "match_info": {
                "matched_terms": self._find_matched_terms(content, query_context.key_terms),
                "term_frequency": self._calculate_term_frequency(content, query_context.key_terms)
            }
        }
        
        # Add legacy fields that tests expect
        formatted_result.update({
            "id": result.get("id", "unknown"),
            "rerank_score": result.get("rerank_score", result.get("relevance_score", 0.0)),
            "original_score": result.get("original_score", 0.0),
            "rank": result.get("rank", 0),
            "score_improvement": result.get("score_improvement", 0.0),
            "distance": result.get("distance", 0.0),
            "relevance": {
                "rank": result.get("rank", 0),
                "confidence": self._calculate_result_confidence(result),
                "rerank_score": result.get("rerank_score", result.get("relevance_score", 0.0)),
                "original_score": result.get("original_score", 0.0),
                "score_improvement": result.get("score_improvement", 0.0)
            }
        })
        
        # Add source_info for metadata enrichment tests
        if "source" in metadata:
            formatted_result["source_info"] = {
                "file": metadata["source"],
                "type": metadata.get("type", "document"),
                "author": metadata.get("author", "Unknown")
            }
        
        return formatted_result
    
    # Helper methods that were missing
    def _calculate_confidence_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence level distribution across results."""
        if not results:
            return {"high": 0.0, "medium": 0.0, "low": 0.0}
        
        high_count = sum(1 for r in results if r.get("confidence", 0) > 0.8)
        medium_count = sum(1 for r in results if 0.5 <= r.get("confidence", 0) <= 0.8)
        low_count = len(results) - high_count - medium_count
        
        total = len(results)
        return {
            "high": high_count / total,
            "medium": medium_count / total,
            "low": low_count / total
        }
    
    def _generate_improvement_suggestions(
        self, 
        query_context: QueryContext, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate suggestions for improving search results."""
        suggestions = []
        
        if not results:
            suggestions.append({
                "type": "expand_scope",
                "suggestion": "Try broader terms or check spelling",
                "reason": "No results found"
            })
        elif len(results) < 3:
            suggestions.append({
                "type": "expand_query",
                "suggestion": "Add related terms or synonyms",
                "reason": "Limited results found"
            })
        
        # Check collection diversity
        collections = set(r.get("metadata", {}).get("collection", "unknown") for r in results)
        if len(collections) > 2:
            suggestions.append({
                "type": "filter_collections",
                "suggestion": "Add collection filter to focus results",
                "reason": "Results span multiple collections"
            })
        
        return suggestions
    
    def _enrich_metadata(
        self, 
        metadata: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich metadata with user-friendly formatting."""
        enriched = metadata.copy()
        
        # Format type information
        doc_type = metadata.get("type", "document")
        enriched["type_display"] = doc_type.replace("_", " ").title()
        
        # Format tags for display
        if "tags" in metadata:
            tags = metadata["tags"]
            if isinstance(tags, list):
                enriched["tags_display"] = ", ".join(tags)
            else:
                enriched["tags_display"] = str(tags)
        
        # Add creation info if available
        if "created" in metadata:
            enriched["created_display"] = metadata["created"]
        
        return enriched
    
    def _calculate_result_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for a result."""
        relevance_score = result.get("relevance_score", result.get("rerank_score", 0.0))
        
        # Base confidence on relevance score
        if relevance_score > 0.9:
            return 0.95
        elif relevance_score > 0.8:
            return 0.85
        elif relevance_score > 0.6:
            return 0.7
        elif relevance_score > 0.4:
            return 0.5
        else:
            return 0.3
    
    def _find_matched_terms(self, content: str, key_terms: List[str]) -> List[str]:
        """Find which query terms are present in the content."""
        import re
        matched = []
        content_lower = content.lower()
        
        for term in key_terms:
            if term.lower() in content_lower:
                matched.append(term)
        
        return matched
    
    def _calculate_term_frequency(self, content: str, key_terms: List[str]) -> Dict[str, int]:
        """Calculate frequency of query terms in content."""
        import re
        frequencies = {}
        content_lower = content.lower()
        
        for term in key_terms:
            pattern = re.escape(term.lower())
            matches = re.findall(pattern, content_lower)
            frequencies[term] = len(matches)
        
        return frequencies

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