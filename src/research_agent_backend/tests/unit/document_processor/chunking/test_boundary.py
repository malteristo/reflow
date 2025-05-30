"""Tests for ChunkBoundary class - intelligent boundary detection."""

import pytest
from research_agent_backend.core.document_processor.chunking import ChunkConfig, ChunkBoundary


class TestChunkBoundary:
    """Tests for ChunkBoundary class - intelligent boundary detection."""
    
    def test_chunk_boundary_creation_with_config(self):
        """Test creating ChunkBoundary with configuration."""
        config = ChunkConfig()
        boundary = ChunkBoundary(config)
        assert boundary.config == config
    
    def test_find_sentence_boundary_preserves_sentences(self):
        """Test finding sentence boundaries to preserve sentence integrity."""
        config = ChunkConfig(preserve_sentences=True)
        boundary = ChunkBoundary(config)
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        # Find boundary near position 30 (middle of second sentence)
        boundary_pos = boundary.find_optimal_boundary(text, target_position=30)
        
        # Should find sentence boundary at position 22 (after "one.")
        assert boundary_pos == 22
    
    def test_find_paragraph_boundary_preserves_paragraphs(self):
        """Test finding paragraph boundaries to preserve paragraph integrity."""
        config = ChunkConfig(preserve_paragraphs=True)
        boundary = ChunkBoundary(config)
        
        text = """First paragraph content.
More content in first paragraph.

Second paragraph starts here.
More content in second paragraph."""
        
        # Find boundary near position 60 (middle of second paragraph)
        boundary_pos = boundary.find_optimal_boundary(text, target_position=60)
        
        # Should find paragraph boundary (double newline)
        assert text[boundary_pos:boundary_pos+2] == "\n\n"
    
    def test_find_word_boundary_fallback(self):
        """Test falling back to word boundaries when no sentence/paragraph boundaries."""
        config = ChunkConfig(preserve_sentences=False, preserve_paragraphs=False)
        boundary = ChunkBoundary(config)
        
        text = "This is a long text without proper sentence endings or paragraph breaks"
        boundary_pos = boundary.find_optimal_boundary(text, target_position=30)
        
        # Should find word boundary (space character)
        assert text[boundary_pos] == " " or boundary_pos == 30
    
    def test_boundary_respects_markup_elements(self):
        """Test boundary detection respects markdown markup elements."""
        config = ChunkConfig()
        boundary = ChunkBoundary(config)
        
        text = "Some text **bold text in the middle** more text here."
        # Try to split in middle of bold markup
        boundary_pos = boundary.find_optimal_boundary(text, target_position=20)
        
        # Should not split inside markup, find boundary before or after
        assert not (text[boundary_pos-2:boundary_pos+2] == "**bo" or 
                   text[boundary_pos-2:boundary_pos+2] == "dle*")
    
    def test_boundary_handles_code_blocks(self):
        """Test boundary detection preserves code block integrity."""
        config = ChunkConfig()
        boundary = ChunkBoundary(config)
        
        text = """Some text before.

```python
def function():
    return "code"
```

Some text after."""
        
        # Try to split inside code block
        boundary_pos = boundary.find_optimal_boundary(text, target_position=40)
        
        # Should not split inside code block
        code_start = text.find("```python")
        code_end = text.find("```", code_start + 3) + 3
        assert not (code_start < boundary_pos < code_end) 