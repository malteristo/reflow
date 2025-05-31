"""
RAG Feedback Generation - Result feedback and refinement suggestions.

This module contains utilities for generating feedback on search results,
explanations of ranking decisions, and suggestions for query refinement.
"""

import logging
from typing import Dict, List, Any, Optional

from .constants import (
    RELEVANCE_THRESHOLDS, RELEVANCE_DESCRIPTIONS, PREFERENCE_KEYWORDS,
    TECH_TERMS, MISMATCH_COLLECTIONS
)
from .query_context import QueryContext, QueryIntent

logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Utility class for generating result feedback and suggestions."""
    
    @staticmethod
    def generate_result_feedback(
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
        try:
            feedback = {
                "search_summary": FeedbackGenerator._generate_search_summary(
                    query_context, search_results, top_k
                ),
                "result_explanations": FeedbackGenerator._generate_result_explanations(
                    query_context, search_results
                ),
                "refinement_suggestions": FeedbackGenerator._generate_refinement_suggestions(
                    query_context, search_results
                ),
                "relevance_metrics": FeedbackGenerator._calculate_relevance_metrics(
                    query_context, search_results
                )
            }
            
            logger.info(f"Generated feedback for {len(search_results)} results")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating result feedback: {e}")
            raise
    
    @staticmethod
    def _generate_search_summary(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]], 
        top_k: int
    ) -> Dict[str, Any]:
        """Generate overall search summary."""
        summary = {
            "total_results": len(search_results),
            "collections_searched": FeedbackGenerator._extract_collections_from_results(search_results),
            "best_match_score": FeedbackGenerator._calculate_best_match_score(search_results),
            "query_coverage": FeedbackGenerator._calculate_query_coverage(query_context, search_results)
        }
        
        # Add filters applied if any
        if query_context.filters:
            summary["filters_applied"] = [
                {
                    "field": f.field,
                    "value": f.value,
                    "operator": f.operator,
                    "confidence": f.confidence
                }
                for f in query_context.filters
            ]
        
        return summary
    
    @staticmethod
    def _generate_result_explanations(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate explanations for individual results."""
        explanations = []
        
        for result in search_results:
            relevance_score = FeedbackGenerator._calculate_relevance_score(result)
            ranking_reason = FeedbackGenerator._generate_ranking_reason(
                query_context, result, relevance_score
            )
            
            explanations.append({
                "result_id": result["id"],
                "relevance_score": relevance_score,
                "ranking_reason": ranking_reason
            })
        
        return explanations
    
    @staticmethod
    def _generate_refinement_suggestions(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate contextual refinement suggestions."""
        suggestions = []
        
        # Handle empty results
        if not search_results:
            return FeedbackGenerator._get_empty_results_suggestions()
        
        # Analyze result quality and add quality-based suggestions
        quality_suggestions = FeedbackGenerator._generate_quality_based_suggestions(search_results)
        suggestions.extend(quality_suggestions)
        
        # Add collection mismatch suggestions
        collection_suggestions = FeedbackGenerator._generate_collection_suggestions(
            query_context, search_results
        )
        suggestions.extend(collection_suggestions)
        
        # Add intent-specific suggestions
        intent_suggestions = FeedbackGenerator._generate_intent_based_suggestions(query_context)
        suggestions.extend(intent_suggestions)
        
        # Add fallback suggestions if none generated
        if not suggestions:
            suggestions.extend(FeedbackGenerator._get_fallback_suggestions(query_context))
        
        return suggestions
    
    @staticmethod
    def _get_empty_results_suggestions() -> List[Dict[str, Any]]:
        """Generate suggestions for empty search results."""
        return [
            {
                "type": "expand_scope",
                "suggestion": "Try broadening your search terms or removing filters",
                "reason": "No results found for current query"
            },
            {
                "type": "refine_terms",
                "suggestion": "Consider using alternative keywords or synonyms",
                "reason": "Current search terms may be too specific"
            }
        ]
    
    @staticmethod
    def _generate_quality_based_suggestions(
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions based on result quality analysis."""
        suggestions = []
        avg_distance = sum(r.get("distance", 0.5) for r in search_results) / len(search_results)
        
        # Poor relevance suggestions
        if avg_distance > 0.7:
            suggestions.append({
                "type": "refine_terms",
                "suggestion": "Consider using more specific or alternative keywords",
                "reason": "Current results have low relevance scores"
            })
        
        return suggestions
    
    @staticmethod
    def _generate_collection_suggestions(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate collection-related suggestions."""
        suggestions = []
        result_collections = FeedbackGenerator._extract_collections_from_results(search_results)
        query_collections = [f.value for f in query_context.filters if f.field == "collection"]
        
        # Handle explicit collection filter mismatches
        if query_collections and not any(col in result_collections for col in query_collections):
            suggestions.append({
                "type": "change_collection",
                "suggestion": f"Try searching in collections: {', '.join(result_collections)}",
                "reason": "Results found in different collections than specified"
            })
        
        # Handle semantic mismatches for tech queries
        if not query_collections and result_collections:
            semantic_suggestions = FeedbackGenerator._check_semantic_collection_mismatch(
                query_context, result_collections
            )
            suggestions.extend(semantic_suggestions)
        
        return suggestions
    
    @staticmethod
    def _check_semantic_collection_mismatch(
        query_context: QueryContext, 
        result_collections: List[str]
    ) -> List[Dict[str, Any]]:
        """Check for semantic mismatches between query and result collections."""
        suggestions = []
        query_terms_lower = [term.lower() for term in query_context.key_terms]
        
        # Check if tech query returned non-tech results
        if any(tech_term in query_terms_lower for tech_term in TECH_TERMS):
            if any(bad_collection in result_collections for bad_collection in MISMATCH_COLLECTIONS):
                suggestions.append({
                    "type": "change_collection",
                    "suggestion": "Try searching in collections: programming, development",
                    "reason": "Results from unrelated collections detected"
                })
            elif not any(good_collection in result_collections 
                        for good_collection in ["programming", "development"]):
                suggestions.append({
                    "type": "change_collection", 
                    "suggestion": f"Try searching in collections: {', '.join(result_collections)}",
                    "reason": "Consider more specific collection targeting"
                })
        
        return suggestions
    
    @staticmethod
    def _generate_intent_based_suggestions(
        query_context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Generate suggestions based on query intent."""
        suggestions = []
        
        # Tutorial-specific suggestions
        if (query_context.intent == QueryIntent.TUTORIAL_SEEKING and 
            not query_context.preferences.get("complexity_level")):
            suggestions.append({
                "type": "add_context",
                "suggestion": "Specify difficulty level (beginner, intermediate, advanced)",
                "reason": "Adding complexity preference can improve tutorial recommendations"
            })
        
        return suggestions
    
    @staticmethod
    def _get_fallback_suggestions(
        query_context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Generate fallback suggestions when no specific issues detected."""
        suggestions = []
        
        # General filtering suggestion
        if query_context.intent == QueryIntent.INFORMATION_SEEKING:
            suggestions.append({
                "type": "add_filter",
                "suggestion": "Try adding collection filters to narrow your search",
                "reason": "Filtering by collection can improve result relevance"
            })
        
        # Query expansion suggestion
        if len(query_context.key_terms) < 2:
            suggestions.append({
                "type": "refine_terms",
                "suggestion": "Add more specific keywords to your query",
                "reason": "More detailed queries often yield better results"
            })
        
        # Intent-specific fallback suggestions
        if query_context.intent == QueryIntent.CODE_SEARCH:
            suggestions.append({
                "type": "add_context",
                "suggestion": "Specify programming language or framework",
                "reason": "Technology-specific searches yield more relevant code examples"
            })
        
        return suggestions
    
    @staticmethod
    def _calculate_relevance_metrics(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall relevance quality metrics."""
        if not search_results:
            return {
                "average_relevance": 0.0,
                "result_diversity": 0.0,
                "query_term_coverage": 0.0
            }
        
        # Calculate average relevance (inverse of distance)
        relevance_scores = [FeedbackGenerator._calculate_relevance_score(result) for result in search_results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Calculate result diversity (collection and content diversity)
        collections = FeedbackGenerator._extract_collections_from_results(search_results)
        diversity = min(len(collections) / max(len(search_results), 1), 1.0)
        
        # Calculate query term coverage
        coverage = FeedbackGenerator._calculate_query_coverage(query_context, search_results)
        
        return {
            "average_relevance": round(avg_relevance, 3),
            "result_diversity": round(diversity, 3),
            "query_term_coverage": round(coverage, 3)
        }
    
    @staticmethod
    def _calculate_relevance_score(result: Dict[str, Any]) -> float:
        """Calculate normalized relevance score from distance."""
        distance = result.get("distance", 0.5)
        # Convert distance to relevance (inverse relationship)
        relevance = max(0.0, 1.0 - distance)
        return round(relevance, 3)
    
    @staticmethod
    def _generate_ranking_reason(
        query_context: QueryContext, 
        result: Dict[str, Any], 
        relevance_score: float
    ) -> str:
        """Generate human-readable ranking explanation."""
        reasons = []
        
        # Add relevance reason using constants
        reasons.append(FeedbackGenerator._get_relevance_description(relevance_score))
        
        # Add content match reasons
        content_reasons = FeedbackGenerator._extract_content_match_reasons(query_context, result)
        reasons.extend(content_reasons)
        
        # Add collection context
        collection_reason = FeedbackGenerator._get_collection_reason(result)
        if collection_reason:
            reasons.append(collection_reason)
        
        # Add preference alignment reasons
        preference_reasons = FeedbackGenerator._extract_preference_reasons(query_context, result)
        reasons.extend(preference_reasons)
        
        # Format final reason with relevance score
        base_reason = f"Ranked due to {', '.join(reasons[:3])}"  # Limit to top 3 reasons
        return f"{base_reason}. Relevance score: {relevance_score}"
    
    @staticmethod
    def _get_relevance_description(relevance_score: float) -> str:
        """Get human-readable relevance description."""
        if relevance_score >= RELEVANCE_THRESHOLDS["HIGH"]:
            return RELEVANCE_DESCRIPTIONS["high"]
        elif relevance_score >= RELEVANCE_THRESHOLDS["MODERATE"]:
            return RELEVANCE_DESCRIPTIONS["moderate"]
        else:
            return RELEVANCE_DESCRIPTIONS["low"]
    
    @staticmethod
    def _extract_content_match_reasons(
        query_context: QueryContext, 
        result: Dict[str, Any]
    ) -> List[str]:
        """Extract reasons based on content matches."""
        reasons = []
        content = result.get("content", "").lower()
        
        for term in query_context.key_terms:
            if term.lower() in content:
                reasons.append(f"contains '{term}'")
        
        return reasons
    
    @staticmethod
    def _get_collection_reason(result: Dict[str, Any]) -> Optional[str]:
        """Get collection-based reason if available."""
        collection = result.get("metadata", {}).get("collection", "")
        return f"from {collection} collection" if collection else None
    
    @staticmethod
    def _extract_preference_reasons(
        query_context: QueryContext, 
        result: Dict[str, Any]
    ) -> List[str]:
        """Extract reasons based on user preference alignment."""
        reasons = []
        content = result.get("content", "").lower()
        
        # Check complexity level preferences
        if query_context.preferences.get("complexity_level") == "beginner":
            if any(word in content for word in PREFERENCE_KEYWORDS["beginner"]):
                reasons.append("matches beginner-level preference")
        
        # Check example preferences
        if query_context.preferences.get("include_examples"):
            if any(word in content for word in PREFERENCE_KEYWORDS["examples"]):
                reasons.append("includes examples as requested")
        
        return reasons
    
    @staticmethod
    def _extract_collections_from_results(search_results: List[Dict[str, Any]]) -> List[str]:
        """Extract unique collections from search results."""
        collections = set()
        for result in search_results:
            collection = result.get("metadata", {}).get("collection")
            if collection:
                collections.add(collection)
        return list(collections)
    
    @staticmethod
    def _calculate_best_match_score(search_results: List[Dict[str, Any]]) -> float:
        """Calculate the best match score from results."""
        if not search_results:
            return 0.0
        
        best_distance = min(r.get("distance", 1.0) for r in search_results)
        return round(max(0.0, 1.0 - best_distance), 3)
    
    @staticmethod
    def _calculate_query_coverage(
        query_context: QueryContext, 
        search_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well results cover query terms."""
        if not query_context.key_terms or not search_results:
            return 0.0
        
        covered_terms = set()
        for result in search_results:
            content = result.get("content", "").lower()
            for term in query_context.key_terms:
                if term.lower() in content:
                    covered_terms.add(term.lower())
        
        coverage = len(covered_terms) / len(query_context.key_terms)
        return min(coverage, 1.0) 