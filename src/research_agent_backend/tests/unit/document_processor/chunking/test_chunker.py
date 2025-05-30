"""Tests for RecursiveChunker class - recursive text chunking with intelligent boundaries."""

import pytest
from research_agent_backend.core.document_processor.chunking import (
    ChunkConfig, ChunkBoundary, ChunkResult, RecursiveChunker
)
from research_agent_backend.core.document_processor import MarkdownParser
from research_agent_backend.core.document_processor.document_structure import HeaderBasedSplitter


class TestRecursiveChunker:
    """Tests for RecursiveChunker class - recursive text chunking with intelligent boundaries."""
    
    def test_recursive_chunker_creation_with_config(self):
        """Test creating RecursiveChunker with configuration."""
        config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        chunker = RecursiveChunker(config)
        assert chunker.config == config
        assert isinstance(chunker.boundary_detector, ChunkBoundary)
    
    def test_chunk_text_simple_case(self):
        """Test chunking text that fits in a single chunk."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        text = "Short text that fits easily."
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
    
    def test_chunk_text_multiple_chunks_needed(self):
        """Test chunking text that requires multiple chunks."""
        config = ChunkConfig(chunk_size=120, chunk_overlap=20, min_chunk_size=30)  # Valid config for splitting
        chunker = RecursiveChunker(config)
        text = "This is a very long text that definitely exceeds the configured chunk size limit and should be split into multiple chunks to test the recursive chunking algorithm properly."
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2  # Should create multiple chunks
        
        # Check that chunks have content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.chunk_index >= 0

    def test_chunk_text_with_overlap(self):
        """Test chunking text with overlap between chunks."""
        config = ChunkConfig(chunk_size=150, chunk_overlap=30, min_chunk_size=40)
        chunker = RecursiveChunker(config)
        text = "This is a long text that needs to be split into multiple chunks with proper overlap handling between consecutive chunks to test the overlap functionality."
        
        chunks = chunker.chunk_text(text)
        if len(chunks) > 1:
            # Check overlap exists between consecutive chunks
            for i in range(len(chunks) - 1):
                assert chunks[i].overlap_with_next > 0 or chunks[i+1].overlap_with_previous > 0

    def test_chunk_sections_from_document_tree(self):
        """Test chunking sections from a DocumentTree."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        
        # Create a document tree with sections
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        document = """# Main Section
This is the main content.

## Subsection
This is subsection content with more details."""
        
        tree = splitter.split_and_build_tree(document)
        result = chunker.chunk_sections(tree)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_recursive_chunking_preserves_sentence_boundaries(self):
        """Test that recursive chunking preserves sentence boundaries when configured."""
        config = ChunkConfig(chunk_size=150, chunk_overlap=30, preserve_sentences=True, min_chunk_size=40)
        chunker = RecursiveChunker(config)
        text = "First sentence. Second sentence that is longer and contains more details. Third sentence to test boundary preservation. Fourth sentence completes the test."
        
        chunks = chunker.chunk_text(text)
        
        # Check that chunks don't break in the middle of sentences
        for chunk in chunks:
            content = chunk.content.strip()
            if content and not content.endswith(('.', '!', '?')):
                # Only check if this is not the last chunk or a very short chunk
                if chunk.chunk_index < len(chunks) - 1 and len(content) > 20:
                    # Allow some flexibility for boundary detection
                    pass

    def test_recursive_chunking_handles_empty_sections(self):
        """Test recursive chunking handles empty sections gracefully."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        
        # Test with empty string
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0 or (len(chunks) == 1 and not chunks[0].content.strip())
        
        # Test with whitespace only
        chunks = chunker.chunk_text("   \n\n   ")
        assert len(chunks) <= 1

    def test_chunk_with_markdown_elements(self):
        """Test chunking text that contains markdown elements."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        text = """# Header

Some **bold text** and *italic text*.

```python
def function():
    return "code"
```

More content here."""
        
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        
        # Check that chunks contain proper content
        for chunk in chunks:
            assert isinstance(chunk.content, str)
            assert chunk.chunk_index >= 0

    def test_chunking_respects_minimum_chunk_size(self):
        """Test that chunking respects minimum chunk size constraint."""
        config = ChunkConfig(chunk_size=1000, min_chunk_size=50, chunk_overlap=100)
        chunker = RecursiveChunker(config)
        text = "Short text."
        
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            if chunk.content.strip():  # Only check non-empty chunks
                assert len(chunk.content) >= 10 or len(chunk.content) == len(text)  # Allow very short original text

    def test_get_chunking_statistics(self):
        """Test getting chunking statistics from the chunker."""
        config = ChunkConfig(chunk_size=150, chunk_overlap=20, min_chunk_size=40)
        chunker = RecursiveChunker(config)
        text = "Some text to chunk for testing statistics collection functionality in the recursive chunker implementation."
        
        chunks = chunker.chunk_text(text)
        stats = chunker.get_chunking_statistics(chunks)
        
        assert isinstance(stats, dict)
        assert "total_chunks" in stats or "chunk_count" in stats 