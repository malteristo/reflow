"""
Test suite for Knowledge Gap Detection System.

This module contains comprehensive unit tests for the knowledge gap detection system
that identifies when search results indicate insufficient knowledge and suggests
external research strategies.

Following TDD methodology: RED phase - all tests should initially fail.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import the classes we'll be implementing
from research_agent_backend.services.knowledge_gap_detector import (
    KnowledgeGapDetector,
    GapDetectionConfig,
    GapAnalysisResult,
    ResearchSuggestion,
    ConfidenceLevel
)
from research_agent_backend.core.query_manager.types import QueryResult, PerformanceMetrics


class TestKnowledgeGapDetector:
    """Test suite for KnowledgeGapDetector class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = GapDetectionConfig(
            low_confidence_threshold=0.3,
            sparse_results_threshold=5,
            minimum_coverage_score=0.4,
            enable_external_suggestions=True
        )
        self.detector = KnowledgeGapDetector(self.config)

    def test_detector_initialization(self):
        """Test that KnowledgeGapDetector initializes with correct configuration."""
        assert self.detector.config == self.config
        assert self.detector.config.low_confidence_threshold == 0.3
        assert self.detector.config.sparse_results_threshold == 5
        assert self.detector.config.minimum_coverage_score == 0.4
        assert self.detector.config.enable_external_suggestions is True

    def test_analyze_high_confidence_results_no_gap(self):
        """Test that high confidence results don't trigger knowledge gap detection."""
        # Create high-confidence query results
        query_result = QueryResult(
            results=[
                {"content": "Detailed information about topic A", "metadata": {"source": "doc1"}},
                {"content": "Comprehensive coverage of topic B", "metadata": {"source": "doc2"}},
                {"content": "Thorough explanation of topic C", "metadata": {"source": "doc3"}}
            ],
            similarity_scores=[0.9, 0.85, 0.8],
            total_results=3
        )
        
        analysis = self.detector.analyze_knowledge_gap("machine learning basics", query_result)
        
        assert isinstance(analysis, GapAnalysisResult)
        assert analysis.has_knowledge_gap is False
        assert analysis.confidence_level == ConfidenceLevel.HIGH
        assert len(analysis.research_suggestions) == 0

    def test_analyze_low_confidence_results_triggers_gap(self):
        """Test that low confidence similarity scores trigger knowledge gap detection."""
        # Create low-confidence query results
        query_result = QueryResult(
            results=[
                {"content": "Vague mention of topic", "metadata": {"source": "doc1"}},
                {"content": "Tangential reference", "metadata": {"source": "doc2"}}
            ],
            similarity_scores=[0.2, 0.15],
            total_results=2
        )
        
        analysis = self.detector.analyze_knowledge_gap("advanced quantum computing", query_result)
        
        assert analysis.has_knowledge_gap is True
        assert analysis.confidence_level == ConfidenceLevel.LOW
        assert len(analysis.research_suggestions) > 0
        assert analysis.gap_reasons["low_similarity_scores"] is True

    def test_analyze_sparse_results_triggers_gap(self):
        """Test that insufficient number of results triggers knowledge gap detection."""
        # Create sparse query results (below threshold)
        query_result = QueryResult(
            results=[
                {"content": "Single relevant document", "metadata": {"source": "doc1"}}
            ],
            similarity_scores=[0.7],
            total_results=1
        )
        
        analysis = self.detector.analyze_knowledge_gap("specific research topic", query_result)
        
        assert analysis.has_knowledge_gap is True
        assert analysis.gap_reasons["sparse_results"] is True
        assert len(analysis.research_suggestions) > 0

    def test_analyze_empty_results_triggers_gap(self):
        """Test that empty results trigger knowledge gap detection."""
        query_result = QueryResult(
            results=[],
            similarity_scores=[],
            total_results=0
        )
        
        analysis = self.detector.analyze_knowledge_gap("completely unknown topic", query_result)
        
        assert analysis.has_knowledge_gap is True
        assert analysis.confidence_level == ConfidenceLevel.NONE
        assert analysis.gap_reasons["no_results"] is True
        assert len(analysis.research_suggestions) > 0

    def test_generate_external_search_suggestions(self):
        """Test external search suggestion generation."""
        gap_analysis = GapAnalysisResult(
            has_knowledge_gap=True,
            confidence_level=ConfidenceLevel.LOW,
            gap_reasons={"sparse_results": True, "low_similarity_scores": True},
            research_suggestions=[],
            coverage_analysis={}
        )
        
        suggestions = self.detector.generate_external_search_suggestions(
            "machine learning algorithms", gap_analysis
        )
        
        assert len(suggestions) == 2
        
        # Check that we have both Perplexity and Google Gemini Deep Research suggestions
        perplexity_suggestion = next(
            (s for s in suggestions if s.platform == "perplexity"), None
        )
        gemini_suggestion = next(
            (s for s in suggestions if s.platform == "google_gemini_deep_research"), None
        )
        
        assert perplexity_suggestion is not None
        assert gemini_suggestion is not None
        
        # Verify suggestion content quality
        assert "machine learning algorithms" in perplexity_suggestion.search_query
        assert "machine learning algorithms" in gemini_suggestion.prompt_text
        assert perplexity_suggestion.priority_score > 0
        assert gemini_suggestion.priority_score > 0

    def test_generate_research_prompts(self):
        """Test generation of structured research prompts."""
        suggestions = self.detector.generate_research_prompts("blockchain technology")
        
        assert len(suggestions) > 0
        
        deep_research_prompt = next(
            (s for s in suggestions if "deep research" in s.prompt_type), None
        )
        assert deep_research_prompt is not None
        assert "blockchain technology" in deep_research_prompt.prompt_text
        assert deep_research_prompt.expected_depth == "comprehensive"

    def test_calculate_coverage_score(self):
        """Test calculation of query coverage score based on results."""
        query = "python machine learning libraries"
        results = [
            {"content": "Python is a programming language used for machine learning"},
            {"content": "Popular machine learning libraries include scikit-learn"},
            {"content": "Libraries like TensorFlow are used for deep learning"}
        ]
        
        coverage_score = self.detector.calculate_coverage_score(query, results)
        
        assert isinstance(coverage_score, float)
        assert 0.0 <= coverage_score <= 1.0

    def test_detect_missing_query_aspects(self):
        """Test detection of missing aspects from the original query."""
        query = "python deep learning frameworks performance comparison"
        results = [
            {"content": "Python is great for deep learning"},
            {"content": "Popular frameworks include TensorFlow and PyTorch"}
            # Note: Missing performance comparison aspect
        ]
        
        missing_aspects = self.detector.detect_missing_query_aspects(query, results)
        
        assert "performance" in missing_aspects or "comparison" in missing_aspects
        assert len(missing_aspects) > 0

    def test_suggest_knowledge_base_augmentation(self):
        """Test suggestions for augmenting the knowledge base."""
        gap_analysis = GapAnalysisResult(
            has_knowledge_gap=True,
            confidence_level=ConfidenceLevel.LOW,
            gap_reasons={"sparse_results": True},
            research_suggestions=[],
            coverage_analysis={"missing_aspects": ["recent developments", "case studies"]}
        )
        
        augmentation_suggestions = self.detector.suggest_knowledge_base_augmentation(
            "artificial intelligence ethics", gap_analysis
        )
        
        assert len(augmentation_suggestions) > 0
        assert any("recent developments" in s["suggestion"] for s in augmentation_suggestions)
        assert any("case studies" in s["suggestion"] for s in augmentation_suggestions)

    def test_confidence_level_classification(self):
        """Test proper classification of confidence levels."""
        # High confidence scenario
        high_scores = [0.9, 0.85, 0.8, 0.75]
        assert self.detector._classify_confidence_level(high_scores) == ConfidenceLevel.HIGH
        
        # Medium confidence scenario
        medium_scores = [0.6, 0.55, 0.5]
        assert self.detector._classify_confidence_level(medium_scores) == ConfidenceLevel.MEDIUM
        
        # Low confidence scenario
        low_scores = [0.3, 0.25, 0.2]
        assert self.detector._classify_confidence_level(low_scores) == ConfidenceLevel.LOW
        
        # No results scenario
        assert self.detector._classify_confidence_level([]) == ConfidenceLevel.NONE

    def test_gap_analysis_with_mixed_confidence(self):
        """Test gap analysis with mixed confidence scores."""
        query_result = QueryResult(
            results=[
                {"content": "High quality match", "metadata": {"source": "doc1"}},
                {"content": "Medium quality match", "metadata": {"source": "doc2"}},
                {"content": "Low quality match", "metadata": {"source": "doc3"}},
                {"content": "Very poor match", "metadata": {"source": "doc4"}}
            ],
            similarity_scores=[0.9, 0.6, 0.3, 0.1],
            total_results=4
        )
        
        analysis = self.detector.analyze_knowledge_gap("mixed confidence query", query_result)
        
        # Should not trigger gap if we have some high confidence results
        assert analysis.has_knowledge_gap is False
        assert analysis.confidence_level == ConfidenceLevel.HIGH

    def test_research_suggestion_prioritization(self):
        """Test that research suggestions are properly prioritized."""
        suggestions = self.detector.generate_external_search_suggestions(
            "quantum computing applications", 
            GapAnalysisResult(
                has_knowledge_gap=True,
                confidence_level=ConfidenceLevel.LOW,
                gap_reasons={"low_similarity_scores": True},
                research_suggestions=[],
                coverage_analysis={}
            )
        )
        
        # Verify suggestions are sorted by priority score
        priority_scores = [s.priority_score for s in suggestions]
        assert priority_scores == sorted(priority_scores, reverse=True)

    def test_configuration_validation(self):
        """Test that configuration validation works properly."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            invalid_config = GapDetectionConfig(
                low_confidence_threshold=1.5,  # Invalid: > 1.0
                sparse_results_threshold=-1,   # Invalid: negative
                minimum_coverage_score=2.0     # Invalid: > 1.0
            )

    def test_analyze_with_disabled_external_suggestions(self):
        """Test gap analysis with external suggestions disabled."""
        config_no_external = GapDetectionConfig(
            low_confidence_threshold=0.3,
            sparse_results_threshold=5,
            minimum_coverage_score=0.4,
            enable_external_suggestions=False
        )
        detector_no_external = KnowledgeGapDetector(config_no_external)
        
        query_result = QueryResult(
            results=[],
            similarity_scores=[],
            total_results=0
        )
        
        analysis = detector_no_external.analyze_knowledge_gap("test query", query_result)
        
        assert analysis.has_knowledge_gap is True
        # Should have no external research suggestions when disabled
        external_suggestions = [
            s for s in analysis.research_suggestions 
            if s.platform in ["perplexity", "google_gemini_deep_research"]
        ]
        assert len(external_suggestions) == 0


class TestGapDetectionConfig:
    """Test suite for GapDetectionConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = GapDetectionConfig()
        
        assert config.low_confidence_threshold == 0.4
        assert config.sparse_results_threshold == 3
        assert config.minimum_coverage_score == 0.5
        assert config.enable_external_suggestions is True

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration should not raise
        valid_config = GapDetectionConfig(
            low_confidence_threshold=0.5,
            sparse_results_threshold=10,
            minimum_coverage_score=0.6
        )
        assert valid_config.low_confidence_threshold == 0.5

        # Invalid threshold values should raise ValueError
        with pytest.raises(ValueError):
            GapDetectionConfig(low_confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            GapDetectionConfig(low_confidence_threshold=1.1)
            
        with pytest.raises(ValueError):
            GapDetectionConfig(sparse_results_threshold=-1)
            
        with pytest.raises(ValueError):
            GapDetectionConfig(minimum_coverage_score=1.5)


class TestGapAnalysisResult:
    """Test suite for GapAnalysisResult class."""

    def test_gap_analysis_result_creation(self):
        """Test creation and attributes of GapAnalysisResult."""
        analysis = GapAnalysisResult(
            has_knowledge_gap=True,
            confidence_level=ConfidenceLevel.LOW,
            gap_reasons={"sparse_results": True, "low_similarity_scores": False},
            research_suggestions=[],
            coverage_analysis={"missing_aspects": ["performance", "scalability"]}
        )
        
        assert analysis.has_knowledge_gap is True
        assert analysis.confidence_level == ConfidenceLevel.LOW
        assert analysis.gap_reasons["sparse_results"] is True
        assert analysis.gap_reasons["low_similarity_scores"] is False
        assert "performance" in analysis.coverage_analysis["missing_aspects"]

    def test_gap_analysis_result_serialization(self):
        """Test that GapAnalysisResult can be converted to dict."""
        analysis = GapAnalysisResult(
            has_knowledge_gap=True,
            confidence_level=ConfidenceLevel.MEDIUM,
            gap_reasons={"no_results": False, "sparse_results": True},
            research_suggestions=[],
            coverage_analysis={}
        )
        
        result_dict = analysis.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["has_knowledge_gap"] is True
        assert result_dict["confidence_level"] == "MEDIUM"
        assert result_dict["gap_reasons"]["sparse_results"] is True


class TestResearchSuggestion:
    """Test suite for ResearchSuggestion class."""

    def test_research_suggestion_creation(self):
        """Test creation of ResearchSuggestion objects."""
        suggestion = ResearchSuggestion(
            platform="perplexity",
            search_query="advanced machine learning techniques",
            prompt_text="Research the latest developments in machine learning",
            priority_score=0.8,
            rationale="Current knowledge base lacks recent developments"
        )
        
        assert suggestion.platform == "perplexity"
        assert suggestion.search_query == "advanced machine learning techniques"
        assert suggestion.priority_score == 0.8
        assert "recent developments" in suggestion.rationale

    def test_research_suggestion_validation(self):
        """Test validation of ResearchSuggestion parameters."""
        # Valid priority score
        suggestion = ResearchSuggestion(
            platform="google_gemini_deep_research",
            search_query="test query",
            prompt_text="test prompt",
            priority_score=0.5
        )
        assert suggestion.priority_score == 0.5

        # Invalid priority score should raise ValueError
        with pytest.raises(ValueError):
            ResearchSuggestion(
                platform="test",
                search_query="test",
                prompt_text="test",
                priority_score=1.5  # Invalid: > 1.0
            )


class TestConfidenceLevel:
    """Test suite for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """Test that ConfidenceLevel enum has correct values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.NONE.value == "none"

    def test_confidence_level_comparison(self):
        """Test that ConfidenceLevel values can be compared."""
        assert ConfidenceLevel.HIGH != ConfidenceLevel.LOW
        assert ConfidenceLevel.MEDIUM == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.NONE != ConfidenceLevel.HIGH 