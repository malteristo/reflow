"""Tests for ChunkResult class - represents chunking operation results."""

import pytest
from core.document_processor.chunking import ChunkResult


class TestChunkResult:
    """Tests for ChunkResult class - represents chunking operation results."""
    
    def test_chunk_result_creation_with_content(self):
        """Test creating ChunkResult with content and metadata."""
        result = ChunkResult(
            content="This is chunk content.",
            start_position=0,
            end_position=22,
            chunk_index=0,
            overlap_with_previous=0,
            overlap_with_next=0,
            boundary_type="sentence"
        )
        assert result.content == "This is chunk content."
        assert result.start_position == 0
        assert result.end_position == 22
        assert result.chunk_index == 0
        assert result.boundary_type == "sentence"
    
    def test_chunk_result_get_length(self):
        """Test ChunkResult can calculate its content length."""
        result = ChunkResult(
            content="Test content",
            start_position=0,
            end_position=12,
            chunk_index=0
        )
        assert result.get_length() == 12
    
    def test_chunk_result_has_overlap_detection(self):
        """Test ChunkResult can detect if it has overlap with neighbors."""
        result_with_overlap = ChunkResult(
            content="Test",
            start_position=0,
            end_position=4,
            chunk_index=0,
            overlap_with_next=10
        )
        result_without_overlap = ChunkResult(
            content="Test",
            start_position=0,
            end_position=4,
            chunk_index=0
        )
        
        assert result_with_overlap.has_overlap() == True
        assert result_without_overlap.has_overlap() == False
    
    def test_chunk_result_to_dict_serialization(self):
        """Test ChunkResult can be serialized to dictionary."""
        result = ChunkResult(
            content="Test content",
            start_position=10,
            end_position=22,
            chunk_index=1,
            boundary_type="paragraph"
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["content"] == "Test content"
        assert result_dict["start_position"] == 10
        assert result_dict["end_position"] == 22
        assert result_dict["chunk_index"] == 1
        assert result_dict["boundary_type"] == "paragraph" 