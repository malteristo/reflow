"""Tests for inline metadata tag extraction functionality."""

import pytest
from core.document_processor import InlineMetadataExtractor


class TestInlineMetadataExtractor:
    """Tests for inline metadata tag extraction functionality."""
    
    def test_inline_metadata_extractor_creation(self):
        """Test creating an InlineMetadataExtractor."""
        extractor = InlineMetadataExtractor()
        assert isinstance(extractor, InlineMetadataExtractor)
    
    def test_extract_simple_tags(self):
        """Test extracting simple inline metadata tags."""
        extractor = InlineMetadataExtractor()
        
        content = """# Document Title

This document is tagged with @tag:important and @tag:urgent.

Some content here with @category:technical and @priority:high."""
        
        result = extractor.extract(content)
        assert len(result.tags) == 4
        assert "important" in [tag.value for tag in result.tags if tag.key == "tag"]
        assert "urgent" in [tag.value for tag in result.tags if tag.key == "tag"]
        assert "technical" in [tag.value for tag in result.tags if tag.key == "category"]
        assert "high" in [tag.value for tag in result.tags if tag.key == "priority"]
    
    def test_extract_key_value_metadata(self):
        """Test extracting key-value inline metadata."""
        extractor = InlineMetadataExtractor()
        
        content = """# Project Documentation

<!-- @version: 1.2.3 -->
<!-- @status: draft -->
<!-- @due_date: 2024-03-15 -->

Content with inline metadata."""
        
        result = extractor.extract(content)
        metadata_dict = {item.key: item.value for item in result.metadata}
        assert metadata_dict["version"] == "1.2.3"
        assert metadata_dict["status"] == "draft"
        assert metadata_dict["due_date"] == "2024-03-15"
    
    def test_extract_json_metadata_blocks(self):
        """Test extracting JSON metadata blocks."""
        extractor = InlineMetadataExtractor()
        
        content = """# Document

<!-- @metadata: {"type": "research", "complexity": 8, "reviewers": ["alice", "bob"]} -->

Content follows."""
        
        result = extractor.extract(content)
        json_metadata = next(item for item in result.metadata if item.key == "metadata")
        parsed_json = json_metadata.parsed_value
        assert parsed_json["type"] == "research"
        assert parsed_json["complexity"] == 8
        assert parsed_json["reviewers"] == ["alice", "bob"]
    
    def test_extract_custom_tag_patterns(self):
        """Test extracting custom tag patterns."""
        extractor = InlineMetadataExtractor()
        
        content = """# Document

This has [[priority:critical]] and [[deadline:2024-06-01]] tags.
Also includes {scope:public} and {audience:developers} metadata."""
        
        result = extractor.extract(content)
        # Should find double bracket and curly brace patterns
        priority_tags = [tag for tag in result.tags if tag.key == "priority"]
        deadline_tags = [tag for tag in result.tags if tag.key == "deadline"]
        scope_tags = [tag for tag in result.tags if tag.key == "scope"]
        
        assert len(priority_tags) > 0
        assert len(deadline_tags) > 0
        assert len(scope_tags) > 0
    
    def test_extract_with_position_tracking(self):
        """Test extracting metadata with position information."""
        extractor = InlineMetadataExtractor()
        
        content = """# Title

@tag:first appears here.

More content.

@tag:second appears later."""
        
        result = extractor.extract(content)
        positions = [(tag.key, tag.value, tag.position) for tag in result.tags]
        
        # First tag should appear before second tag
        first_pos = next(pos for key, val, pos in positions if val == "first")
        second_pos = next(pos for key, val, pos in positions if val == "second")
        assert first_pos < second_pos
    
    def test_extract_removes_metadata_from_content(self):
        """Test that extraction can remove metadata from content."""
        extractor = InlineMetadataExtractor()
        
        content = """# Document @tag:important

This is content <!-- @status: draft --> with inline metadata.

@category:technical should be removed if requested."""
        
        result = extractor.extract(content, remove_from_content=True)
        
        # Check that metadata was extracted
        assert len(result.tags) > 0
        
        # Check that metadata was removed from content
        assert "@tag:important" not in result.cleaned_content
        assert "<!-- @status: draft -->" not in result.cleaned_content
        assert "@category:technical" not in result.cleaned_content
        
        # Check that regular content remains
        assert "This is content" in result.cleaned_content
        assert "# Document" in result.cleaned_content 