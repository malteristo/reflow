"""
Contextual Feedback System for Structured Feedback and Progress Reporting.

Implements intelligent query analysis, refinement suggestions, and contextual
feedback generation for improved user experience and query optimization.
Part of subtask 15.7: Implement Structured Feedback and Progress Reporting.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of contextual feedback."""
    QUERY_REFINEMENT = "query_refinement"
    INSUFFICIENT_RESULTS = "insufficient_results"
    AMBIGUOUS_QUERY = "ambiguous_query"
    COLLECTION_RECOMMENDATION = "collection_recommendation"
    SEARCH_STRATEGY = "search_strategy"
    KNOWLEDGE_GAP = "knowledge_gap"


class SuggestionConfidence(Enum):
    """Confidence levels for feedback suggestions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QueryRefinementSuggestion:
    """Query refinement suggestion data structure."""
    suggestion_type: FeedbackType = FeedbackType.QUERY_REFINEMENT
    suggestion_text: str = ""
    confidence: SuggestionConfidence = SuggestionConfidence.MEDIUM
    rationale: str = ""
    example_query: str = ""
    expected_improvement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary for transmission."""
        return {
            "suggestion_type": self.suggestion_type.value,
            "suggestion_text": self.suggestion_text,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "example_query": self.example_query,
            "expected_improvement": self.expected_improvement,
            "metadata": self.metadata
        }


@dataclass
class ContextualFeedback:
    """Contextual feedback package."""
    feedback_type: FeedbackType = FeedbackType.QUERY_REFINEMENT
    message: str = ""
    suggestions: List[QueryRefinementSuggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary for transmission."""
        return {
            "feedback_type": self.feedback_type.value,
            "message": self.message,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class FeedbackAnalyzer:
    """Analyzes queries and provides contextual feedback."""
    
    def __init__(self):
        """Initialize the feedback analyzer."""
        self._common_terms = {
            "machine learning", "artificial intelligence", "deep learning",
            "neural networks", "nlp", "computer vision", "data science",
            "algorithm", "model", "training", "inference", "classification"
        }
        
        self._collection_keywords = {
            "research_papers": ["paper", "study", "research", "academic", "journal"],
            "documentation": ["docs", "documentation", "manual", "guide", "tutorial"],
            "code_examples": ["code", "example", "implementation", "function", "class"],
            "fundamentals": ["basic", "introduction", "fundamental", "overview", "concept"]
        }
        
        logger.info("FeedbackAnalyzer initialized with contextual analysis capabilities")
    
    def analyze_query(self, query: str) -> List[QueryRefinementSuggestion]:
        """Analyze query and provide refinement suggestions."""
        suggestions = []
        
        # Analyze query length
        if len(query.split()) < 3:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.QUERY_REFINEMENT,
                suggestion_text="Consider adding more specific terms to your query",
                confidence=SuggestionConfidence.HIGH,
                rationale="Short queries often return too many broad results",
                example_query=f"{query} methodology implementation",
                expected_improvement="More precise and relevant results"
            ))
        
        # Analyze for ambiguous terms
        ambiguous_terms = ["it", "this", "that", "thing", "stuff"]
        if any(term in query.lower() for term in ambiguous_terms):
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.AMBIGUOUS_QUERY,
                suggestion_text="Replace vague terms with specific technical terms",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Ambiguous terms reduce search precision",
                example_query=query.replace("it", "[specific technology]"),
                expected_improvement="More targeted and accurate results"
            ))
        
        # Analyze for technical depth
        if not any(term in query.lower() for term in self._common_terms):
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.SEARCH_STRATEGY,
                suggestion_text="Add domain-specific terminology for better results",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Technical terms improve matching with specialized content",
                example_query=f"{query} machine learning algorithm",
                expected_improvement="Access to more technical and detailed content"
            ))
        
        return suggestions
    
    def analyze_insufficient_results(
        self, 
        query: str, 
        result_count: int, 
        collections_searched: List[str]
    ) -> ContextualFeedback:
        """Analyze insufficient results and provide improvement suggestions."""
        suggestions = []
        
        # Suggest query broadening
        if result_count == 0:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.INSUFFICIENT_RESULTS,
                suggestion_text="Try broader terms or synonyms",
                confidence=SuggestionConfidence.HIGH,
                rationale="No results found with current query terms",
                example_query=" ".join(query.split()[:-1]) if len(query.split()) > 1 else f"{query} overview",
                expected_improvement="Increased likelihood of finding relevant content"
            ))
        
        # Suggest collection expansion
        if len(collections_searched) < 3:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.COLLECTION_RECOMMENDATION,
                suggestion_text="Search additional collections for broader coverage",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Limited collection search may miss relevant content",
                example_query=query,
                expected_improvement="Access to more diverse content sources",
                metadata={"suggested_collections": ["research_papers", "documentation", "fundamentals"]}
            ))
        
        return ContextualFeedback(
            feedback_type=FeedbackType.INSUFFICIENT_RESULTS,
            message=f"Found only {result_count} results. Here are some suggestions to improve your search:",
            suggestions=suggestions,
            metadata={
                "original_query": query,
                "result_count": result_count,
                "collections_searched": collections_searched
            }
        )
    
    def analyze_ambiguous_query(self, query: str) -> ContextualFeedback:
        """Analyze query for ambiguity and provide clarification suggestions."""
        suggestions = []
        
        # Check for pronouns and vague terms
        vague_patterns = [
            (r'\bit\b', "the specific technology or concept"),
            (r'\bthis\b', "the particular method or approach"),
            (r'\bthat\b', "the specific technique or tool"),
            (r'\bthey\b', "the specific algorithms or methods"),
            (r'\bthing\b', "the specific component or feature")
        ]
        
        clarified_query = query
        for pattern, replacement in vague_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                clarified_query = re.sub(pattern, replacement, clarified_query, flags=re.IGNORECASE)
                pattern_clean = pattern.strip('\\b')
                suggestions.append(QueryRefinementSuggestion(
                    suggestion_type=FeedbackType.AMBIGUOUS_QUERY,
                    suggestion_text=f"Replace '{pattern_clean}' with specific terms",
                    confidence=SuggestionConfidence.HIGH,
                    rationale="Vague pronouns reduce search accuracy",
                    example_query=clarified_query,
                    expected_improvement="More precise results matching your intent"
                ))
        
        return ContextualFeedback(
            feedback_type=FeedbackType.AMBIGUOUS_QUERY,
            message="Your query contains ambiguous terms that could be clarified:",
            suggestions=suggestions,
            metadata={"original_query": query, "clarified_query": clarified_query}
        )
    
    def recommend_collections(self, query: str) -> ContextualFeedback:
        """Recommend collections based on query content."""
        suggestions = []
        recommended_collections = []
        
        query_lower = query.lower()
        
        # Analyze query for collection-specific keywords
        for collection, keywords in self._collection_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                recommended_collections.append(collection)
        
        # Default recommendations if no specific matches
        if not recommended_collections:
            recommended_collections = ["research_papers", "fundamentals"]
        
        for collection in recommended_collections:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.COLLECTION_RECOMMENDATION,
                suggestion_text=f"Search '{collection}' collection for specialized content",
                confidence=SuggestionConfidence.MEDIUM,
                rationale=f"Query content suggests relevance to {collection} collection",
                example_query=query,
                expected_improvement=f"Access to curated {collection} content",
                metadata={"recommended_collection": collection}
            ))
        
        return ContextualFeedback(
            feedback_type=FeedbackType.COLLECTION_RECOMMENDATION,
            message="Based on your query, these collections may contain relevant content:",
            suggestions=suggestions,
            metadata={
                "query": query,
                "recommended_collections": recommended_collections,
                "analysis_method": "keyword_matching"
            }
        )
    
    def suggest_search_strategy(self, query: str, previous_results: int = 0) -> ContextualFeedback:
        """Suggest search strategy optimizations."""
        suggestions = []
        
        # Analyze query structure
        words = query.split()
        
        if len(words) > 8:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.SEARCH_STRATEGY,
                suggestion_text="Consider breaking long queries into shorter, focused searches",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Shorter queries often yield more precise results",
                example_query=" ".join(words[:5]),
                expected_improvement="Better ranking of most relevant results"
            ))
        
        # Suggest using quotes for exact phrases
        if len(words) >= 2 and '"' not in query:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.SEARCH_STRATEGY,
                suggestion_text="Use quotes around key phrases for exact matching",
                confidence=SuggestionConfidence.HIGH,
                rationale="Exact phrase matching improves precision",
                example_query=f'"{" ".join(words[:2])}" {" ".join(words[2:])}',
                expected_improvement="Results containing exact phrases will rank higher"
            ))
        
        # Suggest alternative search terms
        if previous_results < 5:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.SEARCH_STRATEGY,
                suggestion_text="Try alternative terminology or synonyms",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Different terms may unlock additional relevant content",
                example_query=f"{query} OR {query.replace(words[0], '[synonym]')}",
                expected_improvement="Broader coverage of relevant content"
            ))
        
        return ContextualFeedback(
            feedback_type=FeedbackType.SEARCH_STRATEGY,
            message="Here are some search strategy suggestions:",
            suggestions=suggestions,
            metadata={
                "query_analysis": {
                    "word_count": len(words),
                    "has_quotes": '"' in query,
                    "previous_results": previous_results
                }
            }
        )
    
    def detect_knowledge_gap(self, query: str, available_collections: List[str]) -> ContextualFeedback:
        """Detect potential knowledge gaps and provide guidance."""
        suggestions = []
        
        # Analyze for advanced topics in basic collections
        advanced_terms = ["optimization", "advanced", "deep", "complex", "sophisticated"]
        basic_collections = ["fundamentals", "introduction"]
        
        has_advanced_terms = any(term in query.lower() for term in advanced_terms)
        searching_basic = any(col in available_collections for col in basic_collections)
        
        if has_advanced_terms and searching_basic:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.KNOWLEDGE_GAP,
                suggestion_text="Consider prerequisite topics before advanced concepts",
                confidence=SuggestionConfidence.MEDIUM,
                rationale="Advanced topics may require foundational knowledge",
                example_query=query.replace("advanced", "introduction to"),
                expected_improvement="Better understanding progression"
            ))
        
        # Suggest learning path
        if not suggestions:
            suggestions.append(QueryRefinementSuggestion(
                suggestion_type=FeedbackType.KNOWLEDGE_GAP,
                suggestion_text="Explore related foundational concepts for deeper understanding",
                confidence=SuggestionConfidence.LOW,
                rationale="Building conceptual foundation enhances learning",
                example_query=f"{query} fundamentals basics",
                expected_improvement="More comprehensive understanding"
            ))
        
        return ContextualFeedback(
            feedback_type=FeedbackType.KNOWLEDGE_GAP,
            message="Knowledge building suggestions:",
            suggestions=suggestions,
            metadata={
                "query": query,
                "has_advanced_terms": has_advanced_terms,
                "available_collections": available_collections
            }
        )


class ContextualFeedbackSystem:
    """Comprehensive contextual feedback system."""
    
    def __init__(self, enable_analysis: bool = True, feedback_threshold: int = 3):
        """
        Initialize the contextual feedback system.
        
        Args:
            enable_analysis: Whether to enable query analysis
            feedback_threshold: Minimum result count before suggesting improvements
        """
        self.enable_analysis = enable_analysis
        self.feedback_threshold = feedback_threshold
        self.analyzer = FeedbackAnalyzer()
        
        # Feedback history for learning
        self._feedback_history: List[Dict[str, Any]] = []
        
        logger.info(f"ContextualFeedbackSystem initialized with analysis={'enabled' if enable_analysis else 'disabled'}")
    
    def generate_feedback(
        self,
        query: str,
        result_count: int,
        collections_searched: List[str],
        query_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextualFeedback]:
        """Generate comprehensive contextual feedback."""
        if not self.enable_analysis:
            return []
        
        feedback_items = []
        
        # Analyze insufficient results
        if result_count < self.feedback_threshold:
            feedback_items.append(
                self.analyzer.analyze_insufficient_results(query, result_count, collections_searched)
            )
        
        # Analyze query for ambiguity
        ambiguity_feedback = self.analyzer.analyze_ambiguous_query(query)
        if ambiguity_feedback.suggestions:
            feedback_items.append(ambiguity_feedback)
        
        # Recommend collections
        collection_feedback = self.analyzer.recommend_collections(query)
        feedback_items.append(collection_feedback)
        
        # Suggest search strategy improvements
        strategy_feedback = self.analyzer.suggest_search_strategy(query, result_count)
        if strategy_feedback.suggestions:
            feedback_items.append(strategy_feedback)
        
        # Detect knowledge gaps
        gap_feedback = self.analyzer.detect_knowledge_gap(query, collections_searched)
        if gap_feedback.suggestions:
            feedback_items.append(gap_feedback)
        
        # Store feedback history
        self._feedback_history.append({
            "query": query,
            "result_count": result_count,
            "collections_searched": collections_searched,
            "feedback_count": len(feedback_items),
            "metadata": query_metadata or {}
        })
        
        return feedback_items
    
    def get_query_refinement_suggestions(self, query: str) -> List[QueryRefinementSuggestion]:
        """Get specific query refinement suggestions."""
        return self.analyzer.analyze_query(query)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback system statistics."""
        if not self._feedback_history:
            return {"total_queries": 0, "avg_feedback_per_query": 0.0}
        
        total_feedback = sum(item["feedback_count"] for item in self._feedback_history)
        
        return {
            "total_queries": len(self._feedback_history),
            "total_feedback_generated": total_feedback,
            "avg_feedback_per_query": total_feedback / len(self._feedback_history),
            "feedback_threshold": self.feedback_threshold,
            "analysis_enabled": self.enable_analysis
        } 