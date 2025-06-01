"""
Knowledge Gap Detection Service for Research Agent.

This module identifies when search results indicate insufficient knowledge
and generates suggestions for external research strategies.

Implements FR-RQ-007: Knowledge gap identification and external research guidance.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from research_agent_backend.core.query_manager.types import QueryResult

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels for search results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class GapDetectionConfig:
    """Configuration for knowledge gap detection.
    
    Attributes:
        low_confidence_threshold: Threshold below which similarity scores are considered low (0.0-1.0)
        sparse_results_threshold: Minimum number of results required to avoid sparse results gap
        minimum_coverage_score: Minimum coverage score required for adequate query coverage (0.0-1.0)
        enable_external_suggestions: Whether to generate external research suggestions
    """
    low_confidence_threshold: float = 0.4
    sparse_results_threshold: int = 3
    minimum_coverage_score: float = 0.5
    enable_external_suggestions: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_threshold("low_confidence_threshold", self.low_confidence_threshold)
        self._validate_threshold("minimum_coverage_score", self.minimum_coverage_score)
        
        if self.sparse_results_threshold < 0:
            raise ValueError("sparse_results_threshold must be non-negative")
    
    def _validate_threshold(self, name: str, value: float) -> None:
        """Validate that a threshold value is between 0.0 and 1.0."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass
class ResearchSuggestion:
    """A suggestion for external research.
    
    Attributes:
        platform: Research platform (e.g., 'perplexity', 'google_gemini_deep_research')
        search_query: Optimized search query for the platform
        prompt_text: Detailed prompt for research guidance
        priority_score: Priority score for suggestion ranking (0.0-1.0)
        rationale: Explanation of why this suggestion is recommended
        prompt_type: Type of research prompt (e.g., 'gemini_deep_research_comprehensive')
        expected_depth: Expected depth of research (e.g., 'comprehensive', 'focused')
    """
    platform: str
    search_query: str
    prompt_text: str
    priority_score: float
    rationale: str = ""
    prompt_type: str = ""
    expected_depth: str = ""
    
    def __post_init__(self):
        """Validate research suggestion parameters."""
        if not (0.0 <= self.priority_score <= 1.0):
            raise ValueError("priority_score must be between 0.0 and 1.0")


@dataclass
class GapAnalysisResult:
    """Result of knowledge gap analysis.
    
    Attributes:
        has_knowledge_gap: Whether a knowledge gap was detected
        confidence_level: Overall confidence level of the search results
        gap_reasons: Dictionary of specific gap detection reasons
        research_suggestions: List of external research suggestions
        coverage_analysis: Analysis of query coverage and missing aspects
    """
    has_knowledge_gap: bool
    confidence_level: ConfidenceLevel
    gap_reasons: Dict[str, bool] = field(default_factory=dict)
    research_suggestions: List[ResearchSuggestion] = field(default_factory=list)
    coverage_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert GapAnalysisResult to dictionary representation.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "has_knowledge_gap": self.has_knowledge_gap,
            "confidence_level": self.confidence_level.value.upper(),
            "gap_reasons": self.gap_reasons,
            "research_suggestions": [
                {
                    "platform": s.platform,
                    "search_query": s.search_query,
                    "prompt_text": s.prompt_text,
                    "priority_score": s.priority_score,
                    "rationale": s.rationale
                }
                for s in self.research_suggestions
            ],
            "coverage_analysis": self.coverage_analysis
        }


class KnowledgeGapDetector:
    """
    Service for detecting knowledge gaps in search results and suggesting external research.
    
    Analyzes query results to identify insufficient knowledge coverage and generates
    structured suggestions for external research sources.
    
    The detector uses multiple heuristics to identify knowledge gaps:
    - No results returned from the search
    - Sparse results (below configured threshold)
    - Low similarity scores across all results
    
    High confidence results can override gap detection in certain scenarios,
    but extremely sparse results (single result) will still trigger gaps.
    """
    
    # Class constants for confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4
    MINIMUM_RESULTS_FOR_OVERRIDE = 2
    
    def __init__(self, config: GapDetectionConfig):
        """
        Initialize the knowledge gap detector.
        
        Args:
            config: Configuration for gap detection parameters
        """
        self.config = config
        logger.info("KnowledgeGapDetector initialized with config: %s", config)
    
    def analyze_knowledge_gap(self, query: str, query_result: QueryResult) -> GapAnalysisResult:
        """
        Analyze search results to detect knowledge gaps.
        
        Args:
            query: Original query string
            query_result: Results from the search operation
            
        Returns:
            GapAnalysisResult with gap detection and suggestions
        """
        # Classify confidence level based on similarity scores
        confidence_level = self._classify_confidence_level(query_result.similarity_scores)
        
        # Determine specific gap reasons
        gap_reasons = self._analyze_gap_reasons(query_result)
        
        # Determine overall gap status with confidence override logic
        has_knowledge_gap = self._determine_knowledge_gap(
            gap_reasons, confidence_level, len(query_result.results)
        )
        
        # Generate research suggestions if gap detected and enabled
        research_suggestions = self._generate_suggestions_if_needed(
            query, has_knowledge_gap, gap_reasons, confidence_level
        )
        
        return GapAnalysisResult(
            has_knowledge_gap=has_knowledge_gap,
            confidence_level=confidence_level,
            gap_reasons=gap_reasons,
            research_suggestions=research_suggestions,
            coverage_analysis={}
        )
    
    def _analyze_gap_reasons(self, query_result: QueryResult) -> Dict[str, bool]:
        """Analyze specific reasons for knowledge gaps."""
        return {
            "no_results": len(query_result.results) == 0,
            "sparse_results": len(query_result.results) < self.config.sparse_results_threshold,
            "low_similarity_scores": self._has_low_similarity_scores(query_result.similarity_scores)
        }
    
    def _determine_knowledge_gap(
        self, 
        gap_reasons: Dict[str, bool], 
        confidence_level: ConfidenceLevel, 
        result_count: int
    ) -> bool:
        """
        Determine if there's a knowledge gap using confidence override logic.
        
        High confidence can override gap detection with nuanced rules:
        - Can override if we have some results (not empty)
        - Can override sparse results only if we have at least 2 results
        - Single result scenarios still trigger gaps even with high confidence
        """
        has_gap = any(gap_reasons.values())
        
        if confidence_level == ConfidenceLevel.HIGH and not gap_reasons["no_results"]:
            # High confidence with sparse results: override only if we have enough results
            if gap_reasons["sparse_results"] and result_count >= self.MINIMUM_RESULTS_FOR_OVERRIDE:
                return False
            # High confidence without sparse results: always override
            elif not gap_reasons["sparse_results"]:
                return False
        
        return has_gap
    
    def _generate_suggestions_if_needed(
        self, 
        query: str, 
        has_gap: bool, 
        gap_reasons: Dict[str, bool], 
        confidence_level: ConfidenceLevel
    ) -> List[ResearchSuggestion]:
        """Generate research suggestions if gap detected and external suggestions enabled."""
        if not (has_gap and self.config.enable_external_suggestions):
            return []
        
        gap_analysis = GapAnalysisResult(
            has_knowledge_gap=has_gap,
            confidence_level=confidence_level,
            gap_reasons=gap_reasons,
            research_suggestions=[],
            coverage_analysis={}
        )
        return self.generate_external_search_suggestions(query, gap_analysis)
    
    def _classify_confidence_level(self, similarity_scores: List[float]) -> ConfidenceLevel:
        """
        Classify confidence level based on similarity scores.
        
        Uses the highest similarity score to determine overall confidence level.
        
        Args:
            similarity_scores: List of similarity scores from search results
            
        Returns:
            ConfidenceLevel indicating overall confidence
        """
        if not similarity_scores:
            return ConfidenceLevel.NONE
        
        max_score = max(similarity_scores)
        
        if max_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif max_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _has_low_similarity_scores(self, similarity_scores: List[float]) -> bool:
        """
        Check if similarity scores indicate low confidence.
        
        Args:
            similarity_scores: List of similarity scores
            
        Returns:
            True if all scores are below the configured threshold
        """
        if not similarity_scores:
            return True
        
        return all(score < self.config.low_confidence_threshold for score in similarity_scores)
    
    def generate_external_search_suggestions(
        self, 
        query: str, 
        gap_analysis: GapAnalysisResult
    ) -> List[ResearchSuggestion]:
        """
        Generate suggestions for external search platforms.
        
        Creates prioritized suggestions for different research platforms
        based on their strengths and the nature of the knowledge gap.
        
        Args:
            query: Original query string
            gap_analysis: Results of gap analysis
            
        Returns:
            List of prioritized research suggestions
        """
        suggestions = [
            self._create_perplexity_suggestion(query),
            self._create_gemini_deep_research_suggestion(query)
        ]
        
        # Sort by priority score (descending)
        suggestions.sort(key=lambda x: x.priority_score, reverse=True)
        
        return suggestions
    
    def _create_perplexity_suggestion(self, query: str) -> ResearchSuggestion:
        """Create a Perplexity AI research suggestion."""
        return ResearchSuggestion(
            platform="perplexity",
            search_query=f"{query} recent developments comprehensive analysis",
            prompt_text=f"Research comprehensive information about {query} including recent developments and detailed analysis",
            priority_score=0.9,
            rationale="Perplexity AI provides up-to-date information with citations"
        )
    
    def _create_gemini_deep_research_suggestion(self, query: str) -> ResearchSuggestion:
        """Create a Google Gemini Deep Research suggestion."""
        return ResearchSuggestion(
            platform="google_gemini_deep_research",
            search_query=query,
            prompt_text=(
                f"Conduct comprehensive deep research on {query}. "
                f"Analyze current developments, key challenges, practical applications, "
                f"emerging trends, and provide actionable insights with credible citations. "
                f"Include both technical depth and practical implications."
            ),
            priority_score=0.95,
            rationale="Google Gemini Deep Research provides AI-powered comprehensive analysis with synthesis of multiple sources",
            prompt_type="gemini_deep_research_comprehensive",
            expected_depth="comprehensive"
        )
    
    def generate_research_prompts(self, topic: str) -> List[ResearchSuggestion]:
        """
        Generate structured research prompts for knowledge augmentation.
        
        Creates comprehensive research prompts with different depths and approaches
        to guide thorough investigation of the topic.
        
        Args:
            topic: Topic to generate research prompts for
            
        Returns:
            List of research prompts with different depths and approaches
        """
        return [
            ResearchSuggestion(
                platform="deep research",
                search_query=topic,
                prompt_text=(
                    f"Conduct comprehensive research on {topic} including "
                    "historical context, current state, future trends, "
                    "key challenges, and practical applications"
                ),
                priority_score=0.9,
                rationale="Comprehensive analysis for thorough understanding",
                prompt_type="deep research comprehensive",
                expected_depth="comprehensive"
            )
        ]
    
    def calculate_coverage_score(self, query: str, results: List[Dict[str, Any]]) -> float:
        """
        Calculate how well the results cover the query aspects.
        
        Uses keyword overlap analysis to determine coverage quality.
        
        Args:
            query: Original query string
            results: List of search result documents
            
        Returns:
            Coverage score between 0.0 and 1.0
        """
        if not results:
            return 0.0
        
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        
        total_coverage = sum(
            self._calculate_result_coverage(result, query_words)
            for result in results
        )
        
        return total_coverage / len(results)
    
    def _calculate_result_coverage(self, result: Dict[str, Any], query_words: set) -> float:
        """Calculate coverage score for a single result."""
        content = result.get("content", "").lower()
        content_words = set(content.split())
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def detect_missing_query_aspects(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """
        Detect aspects of the query that are missing from results.
        
        Args:
            query: Original query string
            results: List of search result documents
            
        Returns:
            List of missing aspects/keywords
        """
        query_words = set(query.lower().split())
        
        # Combine all result content
        all_content = " ".join(
            result.get("content", "").lower() 
            for result in results
        )
        content_words = set(all_content.split())
        
        # Find query words not present in any result content
        return list(query_words - content_words)
    
    def suggest_knowledge_base_augmentation(
        self, 
        query: str, 
        gap_analysis: GapAnalysisResult
    ) -> List[Dict[str, Any]]:
        """
        Generate suggestions for augmenting the knowledge base.
        
        Analyzes the gap analysis results to provide specific recommendations
        for improving knowledge base coverage.
        
        Args:
            query: Original query string
            gap_analysis: Results of gap analysis
            
        Returns:
            List of augmentation suggestions with priorities and rationales
        """
        suggestions = []
        
        # Add suggestions for missing aspects
        missing_aspects = gap_analysis.coverage_analysis.get("missing_aspects", [])
        suggestions.extend(
            self._create_aspect_suggestion(aspect, query)
            for aspect in missing_aspects
        )
        
        # Add general suggestions for sparse results
        if gap_analysis.gap_reasons.get("sparse_results", False):
            suggestions.append({
                "type": "content_expansion",
                "suggestion": f"Add more comprehensive documents about {query}",
                "priority": "high",
                "rationale": "Insufficient documents found for the topic"
            })
        
        return suggestions
    
    def _create_aspect_suggestion(self, aspect: str, query: str) -> Dict[str, Any]:
        """Create a suggestion for a missing aspect."""
        priority = "high" if "recent developments" in aspect else "medium"
        return {
            "type": "content_addition",
            "suggestion": f"Add documents covering {aspect} related to {query}",
            "priority": priority,
            "rationale": f"Knowledge base lacks coverage of {aspect}"
        } 