"""
Tests for Result Formatting Service.

This module contains comprehensive tests for the ResultFormatter service,
covering keyword highlighting, metadata display, relevance scoring,
and various output format options.
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from research_agent_backend.services.result_formatter import (
    ResultFormatter,
    FormattingOptions,
    DisplayFormat,
    RelevanceLevel,
    FormattedResult,
    HighlightedText,
    format_results_for_cursor,
    format_results_for_cli,
    create_result_markdown
)


class TestResultFormatter:
    """Test cases for the ResultFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter()
        
        # Sample result data
        self.sample_result = {
            "content": "This is a test document about machine learning algorithms and neural networks.",
            "score": 0.85,
            "document_id": "test_doc_001",
            "collection": "research",
            "metadata": {
                "document_title": "Machine Learning Basics",
                "content_type": "prose",
                "chunk_sequence_id": 3
            },
            "header_path": "Introduction > Machine Learning > Algorithms"
        }
        
        self.sample_results = [
            {
                "content": "Deep learning is a subset of machine learning that uses neural networks.",
                "score": 0.92,
                "document_id": "deep_learning_guide",
                "collection": "research",
                "metadata": {
                    "document_title": "Deep Learning Guide",
                    "content_type": "prose",
                    "chunk_sequence_id": 1
                },
                "header_path": "Chapter 1 > Introduction"
            },
            {
                "content": "Support vector machines are powerful classification algorithms.",
                "score": 0.67,
                "document_id": "svm_tutorial", 
                "collection": "documentation",
                "metadata": {
                    "document_title": "SVM Tutorial",
                    "content_type": "tutorial",
                    "chunk_sequence_id": 2
                },
                "header_path": "Algorithms > Classification"
            },
            {
                "content": "Data preprocessing is crucial for machine learning success.",
                "score": 0.45,
                "document_id": "preprocessing_tips",
                "collection": "notes",
                "metadata": {
                    "document_title": "Data Preprocessing Tips",
                    "content_type": "notes",
                    "chunk_sequence_id": 1
                },
                "header_path": ""
            }
        ]
    
    def test_formatter_initialization(self):
        """Test formatter initialization with default and custom options."""
        # Test default initialization
        default_formatter = ResultFormatter()
        assert default_formatter.options.format_type == DisplayFormat.MARKDOWN
        assert default_formatter.options.highlight_keywords is True
        
        # Test custom options
        custom_options = FormattingOptions(
            format_type=DisplayFormat.HTML,
            highlight_keywords=False,
            max_content_length=200
        )
        custom_formatter = ResultFormatter(custom_options)
        assert custom_formatter.options.format_type == DisplayFormat.HTML
        assert custom_formatter.options.highlight_keywords is False
        assert custom_formatter.options.max_content_length == 200
    
    def test_keyword_extraction(self):
        """Test keyword extraction from queries."""
        # Test basic keyword extraction
        keywords = self.formatter._extract_keywords("machine learning algorithms")
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        assert "the" not in keywords  # Stop word should be removed
        
        # Test quoted phrases
        keywords = self.formatter._extract_keywords('"neural networks" and deep learning')
        assert "neural networks" in keywords
        assert "deep" in keywords
        assert "learning" in keywords
        
        # Test stop word filtering
        keywords = self.formatter._extract_keywords("what is the best algorithm")
        assert "best" in keywords
        assert "algorithm" in keywords
        assert "what" not in keywords
        assert "the" not in keywords
        
        # Test minimum length filtering
        keywords = self.formatter._extract_keywords("ai ml algorithms")
        assert "algorithms" in keywords
        assert "ai" not in keywords  # Too short
        assert "ml" not in keywords  # Too short
    
    def test_keyword_highlighting_markdown(self):
        """Test keyword highlighting in markdown format."""
        options = FormattingOptions(format_type=DisplayFormat.MARKDOWN)
        text = "Machine learning algorithms are powerful tools for data analysis."
        keywords = ["machine", "learning", "algorithms"]
        
        result = self.formatter._highlight_keywords(text, keywords, options)
        
        assert "**machine**" in result.highlighted_text.lower()
        assert "**learning**" in result.highlighted_text.lower()
        assert "**algorithms**" in result.highlighted_text.lower()
        assert len(result.keywords_found) == 3
        assert "machine" in result.keywords_found
    
    def test_keyword_highlighting_html(self):
        """Test keyword highlighting in HTML format."""
        options = FormattingOptions(format_type=DisplayFormat.HTML)
        text = "Machine learning is important"
        keywords = ["machine", "learning"]
        
        result = self.formatter._highlight_keywords(text, keywords, options)
        
        assert "<mark>machine</mark>" in result.highlighted_text.lower()
        assert "<mark>learning</mark>" in result.highlighted_text.lower()
    
    def test_keyword_highlighting_case_insensitive(self):
        """Test that keyword highlighting is case insensitive."""
        options = FormattingOptions(format_type=DisplayFormat.MARKDOWN)
        text = "Machine Learning and ALGORITHMS are important"
        keywords = ["machine", "learning", "algorithms"]
        
        result = self.formatter._highlight_keywords(text, keywords, options)
        
        # Should highlight regardless of case
        assert "**Machine**" in result.highlighted_text
        assert "**Learning**" in result.highlighted_text
        assert "**ALGORITHMS**" in result.highlighted_text
    
    def test_relevance_level_classification(self):
        """Test relevance level classification from scores."""
        # Test very high relevance
        level = self.formatter._get_relevance_level(0.95)
        assert level == RelevanceLevel.VERY_HIGH
        
        # Test high relevance
        level = self.formatter._get_relevance_level(0.8)
        assert level == RelevanceLevel.HIGH
        
        # Test medium relevance
        level = self.formatter._get_relevance_level(0.6)
        assert level == RelevanceLevel.MEDIUM
        
        # Test low relevance
        level = self.formatter._get_relevance_level(0.4)
        assert level == RelevanceLevel.LOW
        
        # Test very low relevance
        level = self.formatter._get_relevance_level(0.1)
        assert level == RelevanceLevel.VERY_LOW
    
    def test_relevance_info_creation(self):
        """Test creation of relevance information."""
        options = FormattingOptions(use_icons=True)
        
        relevance_info = self.formatter._create_relevance_info(0.85, options)
        
        assert relevance_info["score"] == 0.85
        assert relevance_info["level"] == RelevanceLevel.HIGH
        assert relevance_info["label"] == "Very Relevant"
        assert relevance_info["description"] == "Strong match with good context"
        assert "icon" in relevance_info
        assert relevance_info["icon"] == "🔥"
    
    def test_metadata_formatting_markdown(self):
        """Test metadata formatting in markdown."""
        options = FormattingOptions(
            format_type=DisplayFormat.MARKDOWN,
            show_metadata=True
        )
        
        metadata_display = self.formatter._format_metadata(self.sample_result, options)
        
        assert "📄 **Machine Learning Basics**" in metadata_display
        assert "Section: 3" in metadata_display
    
    def test_metadata_formatting_disabled(self):
        """Test metadata formatting when disabled."""
        options = FormattingOptions(show_metadata=False)
        
        metadata_display = self.formatter._format_metadata(self.sample_result, options)
        
        assert metadata_display == ""
    
    def test_source_info_formatting(self):
        """Test source information formatting."""
        options = FormattingOptions(
            format_type=DisplayFormat.MARKDOWN,
            include_source_links=True,
            use_icons=True
        )
        
        source_info = self.formatter._format_source_info(self.sample_result, options)
        
        assert "📚" in source_info  # Research collection icon
        assert "**research**" in source_info
        assert "`test_doc_001`" in source_info
    
    def test_header_path_formatting(self):
        """Test header path formatting."""
        options = FormattingOptions(
            format_type=DisplayFormat.MARKDOWN,
            show_header_path=True
        )
        
        header_path = self.formatter._format_header_path(self.sample_result, options)
        
        assert "📍" in header_path
        assert "Introduction > Machine Learning > Algorithms" in header_path
    
    def test_header_path_empty(self):
        """Test header path formatting when empty."""
        result_no_path = {**self.sample_result, "header_path": ""}
        options = FormattingOptions(show_header_path=True)
        
        header_path = self.formatter._format_header_path(result_no_path, options)
        
        assert header_path == ""
    
    def test_feedback_ui_creation(self):
        """Test user feedback UI creation."""
        options = FormattingOptions(
            format_type=DisplayFormat.MARKDOWN,
            show_feedback_ui=True
        )
        
        feedback_ui = self.formatter._create_feedback_ui(self.sample_result, options)
        
        assert "👍" in feedback_ui
        assert "👎" in feedback_ui
        assert "💬" in feedback_ui
        assert "Feedback" in feedback_ui
    
    def test_feedback_ui_disabled(self):
        """Test feedback UI when disabled."""
        options = FormattingOptions(show_feedback_ui=False)
        
        feedback_ui = self.formatter._create_feedback_ui(self.sample_result, options)
        
        assert feedback_ui == ""
    
    def test_single_result_formatting(self):
        """Test formatting of a single result."""
        keywords = ["machine", "learning"]
        options = FormattingOptions()
        
        formatted_result = self.formatter._format_single_result(
            self.sample_result, keywords, 1, options
        )
        
        assert isinstance(formatted_result, FormattedResult)
        assert "**machine**" in formatted_result.content.lower()
        assert "**learning**" in formatted_result.content.lower()
        assert formatted_result.relevance_info["score"] == 0.85
        assert formatted_result.highlights_count == 2
        assert formatted_result.raw_result == self.sample_result
    
    def test_content_truncation(self):
        """Test content truncation when exceeding max length."""
        long_content_result = {
            **self.sample_result,
            "content": "This is a very long content " * 50  # Make it very long
        }
        
        options = FormattingOptions(max_content_length=100)
        keywords = ["content"]
        
        formatted_result = self.formatter._format_single_result(
            long_content_result, keywords, 1, options
        )
        
        assert formatted_result.content_truncated is True
        assert len(formatted_result.content) <= 103  # 100 + "..."
        assert formatted_result.content.endswith("...")
    
    def test_format_results_list(self):
        """Test formatting a list of results."""
        query = "machine learning algorithms"
        
        formatted_results = self.formatter.format_results(self.sample_results, query)
        
        assert len(formatted_results) == 3
        assert all(isinstance(result, FormattedResult) for result in formatted_results)
        
        # Check that highest scoring result comes first (if not reordered)
        assert formatted_results[0].relevance_info["score"] == 0.92
    
    def test_format_results_with_filtering(self):
        """Test result formatting with minimum relevance filtering."""
        options = FormattingOptions(minimum_relevance=0.5)
        query = "machine learning"
        
        formatted_results = self.formatter.format_results(
            self.sample_results, query, options
        )
        
        # Should filter out the 0.45 score result
        assert len(formatted_results) == 2
        assert all(result.relevance_info["score"] >= 0.5 for result in formatted_results)
    
    def test_format_results_exclude_empty(self):
        """Test excluding empty or invalid results."""
        results_with_empty = [
            self.sample_results[0],
            {"content": "", "score": 0.8, "document_id": "empty"},  # Empty content
            self.sample_results[1]
        ]
        
        options = FormattingOptions(exclude_empty_results=True)
        query = "test"
        
        formatted_results = self.formatter.format_results(
            results_with_empty, query, options
        )
        
        assert len(formatted_results) == 2  # Empty result should be excluded
    
    def test_is_valid_result(self):
        """Test result validation."""
        # Valid result
        valid_result = {"content": "Some content here"}
        assert self.formatter._is_valid_result(valid_result) is True
        
        # Invalid results
        empty_result = {"content": ""}
        assert self.formatter._is_valid_result(empty_result) is False
        
        whitespace_result = {"content": "   \n\t  "}
        assert self.formatter._is_valid_result(whitespace_result) is False
        
        no_content_result = {"score": 0.8}
        assert self.formatter._is_valid_result(no_content_result) is False
    
    def test_fallback_result_creation(self):
        """Test creation of fallback results when formatting fails."""
        invalid_result = {"some": "invalid", "data": "structure"}
        
        fallback = self.formatter._create_fallback_result(invalid_result, 1)
        
        assert isinstance(fallback, FormattedResult)
        assert fallback.content == "Content unavailable"
        assert fallback.raw_result == invalid_result
    
    def test_query_summary_markdown(self):
        """Test query summary formatting in markdown."""
        formatted_results = self.formatter.format_results(self.sample_results, "machine learning")
        
        summary = self.formatter.format_query_summary(
            formatted_results, "machine learning", 3
        )
        
        assert "## Search Results for: *machine learning*" in summary
        assert "Found **3** relevant results" in summary
        assert "### Relevance Distribution" in summary
        assert "🎯 Highly Relevant: 1" in summary  # 0.92 score is VERY_HIGH (≥0.9)
    
    def test_query_summary_no_results(self):
        """Test query summary when no results found."""
        summary = self.formatter.format_query_summary([], "nonexistent query", 0)
        
        assert "## No Results Found" in summary
        assert "No documents matched your query: **nonexistent query**" in summary
    
    def test_query_summary_html(self):
        """Test query summary formatting in HTML."""
        options = FormattingOptions(format_type=DisplayFormat.HTML)
        formatted_results = self.formatter.format_results(self.sample_results, "test")
        
        summary = self.formatter.format_query_summary(
            formatted_results, "test", 3, options
        )
        
        assert "<h3>Search Results for: <em>test</em></h3>" in summary
        assert "<strong>3</strong>" in summary
    
    def test_collection_icons(self):
        """Test collection icon assignment."""
        research_result = {**self.sample_result, "collection": "research"}
        code_result = {**self.sample_result, "collection": "code"}
        unknown_result = {**self.sample_result, "collection": "unknown"}
        
        options = FormattingOptions(use_icons=True)
        
        # Test research icon
        source_info = self.formatter._format_source_info(research_result, options)
        assert "📚" in source_info
        
        # Test code icon
        source_info = self.formatter._format_source_info(code_result, options)
        assert "💻" in source_info
        
        # Test default icon for unknown collection
        source_info = self.formatter._format_source_info(unknown_result, options)
        assert "📁" in source_info
    
    def test_formatting_error_handling(self):
        """Test error handling during formatting."""
        # Create a result that might cause formatting errors
        problematic_result = {
            "content": None,  # None instead of string
            "score": "invalid",  # String instead of float
            "metadata": "not_a_dict"  # String instead of dict
        }
        
        # Should not raise exception, should create fallback
        keywords = ["test"]
        options = FormattingOptions()
        
        try:
            formatted_result = self.formatter._format_single_result(
                problematic_result, keywords, 1, options
            )
            # Should get a fallback result
            assert isinstance(formatted_result, FormattedResult)
        except Exception:
            # If exception occurs, the format_results method should handle it
            formatted_results = self.formatter.format_results(
                [problematic_result], "test query"
            )
            assert len(formatted_results) == 1
            assert isinstance(formatted_results[0], FormattedResult)


class TestFormattingOptions:
    """Test cases for FormattingOptions configuration."""
    
    def test_default_options(self):
        """Test default formatting options."""
        options = FormattingOptions()
        
        assert options.format_type == DisplayFormat.MARKDOWN
        assert options.highlight_keywords is True
        assert options.show_metadata is True
        assert options.show_header_path is True
        assert options.show_relevance_scores is True
        assert options.max_content_length == 500
        assert options.use_icons is True
    
    def test_custom_options(self):
        """Test custom formatting options."""
        options = FormattingOptions(
            format_type=DisplayFormat.HTML,
            highlight_keywords=False,
            max_content_length=200,
            use_icons=False,
            compact_mode=True
        )
        
        assert options.format_type == DisplayFormat.HTML
        assert options.highlight_keywords is False
        assert options.max_content_length == 200
        assert options.use_icons is False
        assert options.compact_mode is True


class TestHighlightedText:
    """Test cases for HighlightedText dataclass."""
    
    def test_highlighted_text_creation(self):
        """Test creation of HighlightedText objects."""
        original = "This is test content"
        highlighted = "This is **test** content"
        positions = [(8, 12)]
        keywords = {"test"}
        
        highlighted_text = HighlightedText(
            original_text=original,
            highlighted_text=highlighted,
            keyword_positions=positions,
            keywords_found=keywords
        )
        
        assert highlighted_text.original_text == original
        assert highlighted_text.highlighted_text == highlighted
        assert highlighted_text.keyword_positions == positions
        assert highlighted_text.keywords_found == keywords


class TestFormattedResult:
    """Test cases for FormattedResult dataclass."""
    
    def test_formatted_result_creation(self):
        """Test creation of FormattedResult objects."""
        raw_result = {"content": "test", "score": 0.8}
        relevance_info = {"score": 0.8, "level": RelevanceLevel.HIGH}
        
        formatted_result = FormattedResult(
            content="**test** content",
            relevance_info=relevance_info,
            metadata_display="Test Document",
            source_info="test_source",
            header_path="Test > Path",
            feedback_ui="👍 👎",
            raw_result=raw_result,
            content_truncated=False,
            highlights_count=1
        )
        
        assert formatted_result.content == "**test** content"
        assert formatted_result.relevance_info == relevance_info
        assert formatted_result.content_truncated is False
        assert formatted_result.highlights_count == 1
        assert formatted_result.raw_result == raw_result


class TestConvenienceFunctions:
    """Test cases for convenience formatting functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_results = [
            {
                "content": "Machine learning algorithms for data analysis",
                "score": 0.85,
                "document_id": "ml_guide",
                "collection": "research"
            }
        ]
    
    def test_format_results_for_cursor(self):
        """Test Cursor-optimized formatting."""
        formatted_results = format_results_for_cursor(
            self.sample_results, "machine learning"
        )
        
        assert len(formatted_results) == 1
        assert isinstance(formatted_results[0], FormattedResult)
        assert formatted_results[0].formatting_options.format_type == DisplayFormat.MARKDOWN
        assert formatted_results[0].formatting_options.show_feedback_ui is True
        assert formatted_results[0].formatting_options.use_icons is True
    
    def test_format_results_for_cursor_compact(self):
        """Test Cursor-optimized formatting in compact mode."""
        formatted_results = format_results_for_cursor(
            self.sample_results, "machine learning", compact=True
        )
        
        assert formatted_results[0].formatting_options.max_content_length == 400
        assert formatted_results[0].formatting_options.compact_mode is True
    
    def test_format_results_for_cli(self):
        """Test CLI-optimized formatting."""
        formatted_results = format_results_for_cli(
            self.sample_results, "machine learning"
        )
        
        assert len(formatted_results) == 1
        assert isinstance(formatted_results[0], FormattedResult)
        assert formatted_results[0].formatting_options.format_type == DisplayFormat.RICH_CONSOLE
        assert formatted_results[0].formatting_options.show_feedback_ui is False
        assert formatted_results[0].formatting_options.max_content_length == 600
    
    def test_format_results_for_cli_no_colors(self):
        """Test CLI formatting without colors."""
        formatted_results = format_results_for_cli(
            self.sample_results, "machine learning", use_colors=False
        )
        
        assert formatted_results[0].formatting_options.format_type == DisplayFormat.PLAIN_TEXT
        assert formatted_results[0].formatting_options.use_icons is False
    
    def test_create_result_markdown(self):
        """Test complete markdown result creation."""
        formatted_result = FormattedResult(
            content="This is **machine learning** content",
            relevance_info={
                "score": 0.85,
                "level": RelevanceLevel.HIGH,
                "label": "Very Relevant",
                "icon": "🔥"
            },
            metadata_display="📄 **ML Guide** | Type: `prose`",
            source_info="📚 **research** / `ml_guide`",
            header_path="📍 Chapter 1 > Introduction",
            feedback_ui="👍 👎 💬 Feedback",
            raw_result={}
        )
        
        markdown = create_result_markdown(formatted_result, 1)
        
        assert "### 1. 🔥 Very Relevant (0.85)" in markdown
        assert "📚 **research** / `ml_guide`" in markdown
        assert "📄 **ML Guide** | Type: `prose`" in markdown
        assert "📍 Chapter 1 > Introduction" in markdown
        assert "This is **machine learning** content" in markdown
        assert "👍 👎 💬 Feedback" in markdown
        assert "---" in markdown
    
    def test_create_result_markdown_minimal(self):
        """Test markdown creation with minimal data."""
        formatted_result = FormattedResult(
            content="Simple content",
            relevance_info={"score": 0.5, "label": "Relevant", "icon": "✅"},
            metadata_display="",
            source_info="",
            header_path="",
            feedback_ui="",
            raw_result={}
        )
        
        markdown = create_result_markdown(formatted_result, 2)
        
        assert "### 2. ✅ Relevant (0.50)" in markdown
        assert "Simple content" in markdown
        assert "---" in markdown


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_query_highlighting(self):
        """Test highlighting with empty query."""
        formatter = ResultFormatter()
        text = "Some text content"
        keywords = []
        options = FormattingOptions()
        
        result = formatter._highlight_keywords(text, keywords, options)
        
        assert result.highlighted_text == text
        assert len(result.keywords_found) == 0
    
    def test_empty_content_highlighting(self):
        """Test highlighting with empty content."""
        formatter = ResultFormatter()
        text = ""
        keywords = ["test"]
        options = FormattingOptions()
        
        result = formatter._highlight_keywords(text, keywords, options)
        
        assert result.highlighted_text == ""
        assert len(result.keywords_found) == 0
    
    def test_special_characters_in_keywords(self):
        """Test highlighting with special characters in keywords."""
        formatter = ResultFormatter()
        text = "The C++ programming language is powerful"
        keywords = ["C++", "programming"]
        options = FormattingOptions(format_type=DisplayFormat.MARKDOWN)
        
        result = formatter._highlight_keywords(text, keywords, options)
        
        assert "**C\\+\\+**" in result.highlighted_text or "**C++**" in result.highlighted_text
        assert "**programming**" in result.highlighted_text
    
    def test_format_results_empty_list(self):
        """Test formatting empty results list."""
        formatter = ResultFormatter()
        
        formatted_results = formatter.format_results([], "test query")
        
        assert len(formatted_results) == 0
    
    def test_very_high_and_very_low_scores(self):
        """Test edge case scores."""
        formatter = ResultFormatter()
        
        # Test score of 1.0
        level = formatter._get_relevance_level(1.0)
        assert level == RelevanceLevel.VERY_HIGH
        
        # Test score of 0.0
        level = formatter._get_relevance_level(0.0)
        assert level == RelevanceLevel.VERY_LOW
        
        # Test negative score (edge case)
        level = formatter._get_relevance_level(-0.1)
        assert level == RelevanceLevel.VERY_LOW 