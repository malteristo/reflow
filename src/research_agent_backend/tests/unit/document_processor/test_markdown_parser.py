"""
Test module for Basic Markdown Parser

This module contains comprehensive tests for the markdown parsing functionality,
including Pattern class for regex matching, Rule class for transformation rules,
and the main MarkdownParser class.

Tests are extracted from the original monolithic test file and aligned with
the modular document processor architecture.
"""

import pytest
import re
from typing import List, Dict, Any

from research_agent_backend.core.document_processor import (
    Pattern,
    Rule,
    MarkdownParser,
    MarkdownParseError
)


class TestPattern:
    """Tests for the Pattern class - regex matching functionality."""
    
    def test_pattern_creation_with_name_and_regex(self):
        """Test creating a Pattern with name and regex."""
        pattern = Pattern("header", r"^(#{1,6})\s+(.+)$")
        assert pattern.name == "header"
        assert pattern.regex_pattern == r"^(#{1,6})\s+(.+)$"
        assert isinstance(pattern.compiled_regex, re.Pattern)
    
    def test_pattern_match_finds_header(self):
        """Test Pattern can match markdown headers."""
        pattern = Pattern("header", r"^(#{1,6})\s+(.+)$")
        text = "## Hello World"
        match = pattern.match(text)
        assert match is not None
        assert match.group(1) == "##"
        assert match.group(2) == "Hello World"
    
    def test_pattern_match_returns_none_for_no_match(self):
        """Test Pattern returns None when no match found."""
        pattern = Pattern("header", r"^(#{1,6})\s+(.+)$")
        text = "This is regular text"
        match = pattern.match(text)
        assert match is None
    
    def test_pattern_findall_returns_all_matches(self):
        """Test Pattern can find all matches in text."""
        pattern = Pattern("bold", r"\*\*(.*?)\*\*")
        text = "This is **bold** and this is also **very bold**"
        matches = pattern.findall(text)
        assert len(matches) == 2
        assert matches[0] == "bold"
        assert matches[1] == "very bold"
    
    def test_pattern_with_invalid_regex_raises_error(self):
        """Test Pattern creation fails with invalid regex."""
        with pytest.raises(MarkdownParseError):
            Pattern("invalid", "[")  # Invalid regex


class TestRule:
    """Tests for the Rule class - transformation rules."""
    
    def test_rule_creation_with_pattern_and_replacement(self):
        """Test creating a Rule with pattern and replacement."""
        pattern = Pattern("bold", r"\*\*(.*?)\*\*")
        rule = Rule(pattern, r"<strong>\1</strong>")
        assert rule.pattern == pattern
        assert rule.replacement == r"<strong>\1</strong>"
    
    def test_rule_apply_transforms_bold_text(self):
        """Test Rule can transform bold markdown to HTML."""
        pattern = Pattern("bold", r"\*\*(.*?)\*\*")
        rule = Rule(pattern, r"<strong>\1</strong>")
        text = "This is **bold** text"
        result = rule.apply(text)
        assert result == "This is <strong>bold</strong> text"
    
    def test_rule_apply_transforms_multiple_occurrences(self):
        """Test Rule transforms all occurrences in text."""
        pattern = Pattern("italic", r"\*(.*?)\*")
        rule = Rule(pattern, r"<em>\1</em>")
        text = "This is *italic* and this is *also italic*"
        result = rule.apply(text)
        assert result == "This is <em>italic</em> and this is <em>also italic</em>"
    
    def test_rule_apply_with_no_matches_returns_original(self):
        """Test Rule returns original text when no matches found."""
        pattern = Pattern("bold", r"\*\*(.*?)\*\*")
        rule = Rule(pattern, r"<strong>\1</strong>")
        text = "This is regular text"
        result = rule.apply(text)
        assert result == "This is regular text"
    
    def test_rule_with_complex_header_transformation(self):
        """Test Rule can handle complex header transformations."""
        pattern = Pattern("header", r"^(#{1,6})\s+(.+)$")
        rule = Rule(pattern, lambda match: f"<h{len(match.group(1))}>{match.group(2)}</h{len(match.group(1))}>")
        text = "### My Header"
        result = rule.apply(text)
        assert result == "<h3>My Header</h3>"


class TestMarkdownParser:
    """Tests for the main MarkdownParser class."""
    
    def test_parser_creation_with_default_rules(self):
        """Test creating a MarkdownParser with default rules."""
        parser = MarkdownParser()
        assert isinstance(parser.rules, list)
        assert len(parser.rules) > 0
        # Should have default rules for headers, bold, italic, links
        rule_names = [rule.pattern.name for rule in parser.rules]
        assert "header" in rule_names
        assert "bold" in rule_names
        assert "italic" in rule_names
        assert "link" in rule_names
    
    def test_parser_creation_with_custom_rules(self):
        """Test creating a MarkdownParser with custom rules."""
        pattern = Pattern("test", r"test")
        rule = Rule(pattern, "TEST")
        parser = MarkdownParser(rules=[rule])
        assert len(parser.rules) == 1
        assert parser.rules[0] == rule
    
    def test_parser_add_rule_appends_to_rules_list(self):
        """Test adding a rule to the parser."""
        parser = MarkdownParser(rules=[])
        pattern = Pattern("test", r"test")
        rule = Rule(pattern, "TEST")
        parser.add_rule(rule)
        assert len(parser.rules) == 1
        assert parser.rules[0] == rule
    
    def test_parser_parse_transforms_headers(self):
        """Test parser transforms markdown headers to HTML."""
        parser = MarkdownParser()
        text = "# Main Header\n## Sub Header\n### Sub Sub Header"
        result = parser.parse(text)
        assert "<h1>Main Header</h1>" in result
        assert "<h2>Sub Header</h2>" in result
        assert "<h3>Sub Sub Header</h3>" in result
    
    def test_parser_parse_transforms_bold_text(self):
        """Test parser transforms bold markdown to HTML."""
        parser = MarkdownParser()
        text = "This is **bold** text"
        result = parser.parse(text)
        assert result == "This is <strong>bold</strong> text"
    
    def test_parser_parse_transforms_italic_text(self):
        """Test parser transforms italic markdown to HTML."""
        parser = MarkdownParser()
        text = "This is *italic* text"
        result = parser.parse(text)
        assert result == "This is <em>italic</em> text"
    
    def test_parser_parse_transforms_links(self):
        """Test parser transforms markdown links to HTML."""
        parser = MarkdownParser()
        text = "Check out [Google](https://google.com) for search"
        result = parser.parse(text)
        assert '<a href="https://google.com">Google</a>' in result
    
    def test_parser_parse_handles_complex_document(self):
        """Test parser handles a complex markdown document."""
        parser = MarkdownParser()
        text = """# Main Title
        
This is a paragraph with **bold** and *italic* text.

## Section Header

Another paragraph with a [link](https://example.com).

### Subsection

More content here."""
        
        result = parser.parse(text)
        assert "<h1>Main Title</h1>" in result
        assert "<h2>Section Header</h2>" in result
        assert "<h3>Subsection</h3>" in result
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert '<a href="https://example.com">link</a>' in result
    
    def test_parser_parse_preserves_non_markdown_text(self):
        """Test parser preserves text that doesn't match markdown patterns."""
        parser = MarkdownParser()
        text = "This is just regular text with no markdown."
        result = parser.parse(text)
        assert result == text
    
    def test_parser_parse_handles_empty_string(self):
        """Test parser handles empty input."""
        parser = MarkdownParser()
        result = parser.parse("")
        assert result == ""
    
    def test_parser_parse_handles_whitespace_only(self):
        """Test parser handles whitespace-only input."""
        parser = MarkdownParser()
        result = parser.parse("   \n\t  ")
        assert result == "   \n\t  "
    
    def test_parser_get_pattern_by_name_returns_correct_pattern(self):
        """Test parser can retrieve patterns by name."""
        parser = MarkdownParser()
        header_pattern = parser.get_pattern_by_name("header")
        assert header_pattern is not None
        assert header_pattern.name == "header"
    
    def test_parser_get_pattern_by_name_returns_none_for_unknown(self):
        """Test parser returns None for unknown pattern names."""
        parser = MarkdownParser()
        unknown_pattern = parser.get_pattern_by_name("unknown")
        assert unknown_pattern is None
    
    def test_parser_rule_application_order_matters(self):
        """Test that rules are applied in the order they were added."""
        # Create a parser with specific rule order
        pattern1 = Pattern("first", r"test")
        rule1 = Rule(pattern1, "FIRST")
        pattern2 = Pattern("second", r"FIRST")
        rule2 = Rule(pattern2, "SECOND")
        
        parser = MarkdownParser(rules=[rule1, rule2])
        result = parser.parse("test")
        assert result == "SECOND"  # Should apply both transformations in order


class TestMarkdownParseError:
    """Tests for MarkdownParseError exception."""
    
    def test_markdown_parse_error_is_exception(self):
        """Test MarkdownParseError is a proper exception."""
        error = MarkdownParseError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_markdown_parse_error_with_pattern_info(self):
        """Test MarkdownParseError can include pattern information."""
        error = MarkdownParseError("Invalid pattern", pattern_name="header", regex="invalid[")
        assert "Invalid pattern" in str(error)
        # Additional attributes should be accessible
        assert hasattr(error, 'pattern_name')
        assert hasattr(error, 'regex')


class TestMarkdownParserIntegration:
    """Integration tests for the complete markdown parsing system."""
    
    def test_full_markdown_document_parsing(self):
        """Test complete markdown document with all element types."""
        parser = MarkdownParser()
        
        markdown_text = """# Main Document Title

This is the introduction paragraph with **bold** text and *italic* text.

## First Section

This section contains a [link](https://example.com) and some content.

### Subsection A

More detailed content here.

### Subsection B

Even more content with **emphasis**.

## Second Section

Final section content."""
        
        result = parser.parse(markdown_text)
        
        # Verify all transformations
        assert "<h1>Main Document Title</h1>" in result
        assert "<h2>First Section</h2>" in result
        assert "<h3>Subsection A</h3>" in result
        assert "<h3>Subsection B</h3>" in result
        assert "<h2>Second Section</h2>" in result
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert '<a href="https://example.com">link</a>' in result
    
    def test_nested_formatting_combinations(self):
        """Test complex nested formatting scenarios."""
        parser = MarkdownParser()
        text = "This has **bold with *italic inside*** and standalone *italic*"
        result = parser.parse(text)
        # Should handle nested formatting gracefully
        assert "<strong>" in result and "<em>" in result
    
    def test_edge_cases_and_malformed_markdown(self):
        """Test parser handles edge cases and malformed markdown gracefully."""
        parser = MarkdownParser()
        
        # Test various edge cases
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "# ",  # Header with no content
            "**bold without closing",  # Unclosed bold
            "*italic without closing",  # Unclosed italic
            "[link without closing",  # Unclosed link
        ]
        
        for case in edge_cases:
            # Should not raise exceptions
            result = parser.parse(case)
            assert isinstance(result, str) 