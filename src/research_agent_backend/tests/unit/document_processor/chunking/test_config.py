"""Tests for ChunkConfig class - configuration for chunking parameters."""

import pytest
from research_agent_backend.core.document_processor.chunking import ChunkConfig


class TestChunkConfig:
    """Tests for ChunkConfig class - configuration for chunking parameters."""
    
    def test_chunk_config_creation_with_defaults(self):
        """Test creating ChunkConfig with default values."""
        config = ChunkConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.preserve_sentences == True
        assert config.preserve_paragraphs == True
    
    def test_chunk_config_creation_with_custom_values(self):
        """Test creating ChunkConfig with custom parameters."""
        config = ChunkConfig(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            preserve_sentences=False,
            preserve_paragraphs=False
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.preserve_sentences == False
        assert config.preserve_paragraphs == False
    
    def test_chunk_config_validation_chunk_size_positive(self):
        """Test ChunkConfig validates chunk_size is positive."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkConfig(chunk_size=0)
    
    def test_chunk_config_validation_overlap_reasonable(self):
        """Test ChunkConfig validates overlap is less than chunk_size."""
        with pytest.raises(ValueError, match=r"chunk_overlap \(\d+\) must be less than chunk_size \(\d+\)"):
            ChunkConfig(chunk_size=100, chunk_overlap=150)
    
    def test_chunk_config_validation_min_chunk_size(self):
        """Test ChunkConfig validates min_chunk_size is reasonable."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkConfig(min_chunk_size=-1) 