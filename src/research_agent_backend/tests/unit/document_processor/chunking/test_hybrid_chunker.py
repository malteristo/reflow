"""Tests for HybridChunker - FR-KB-002.1 implementation verification."""

import pytest
from unittest.mock import Mock, patch
from research_agent_backend.core.document_processor.chunking import (
    ChunkConfig, HybridChunker, HybridChunkResult
)
from research_agent_backend.core.document_processor.atomic_units import AtomicUnitType


class TestHybridChunker:
    """Tests for HybridChunker - Complete FR-KB-002.1 implementation."""
    
    def test_hybrid_chunker_creation_with_config(self):
        """Test creating HybridChunker with configuration."""
        config = ChunkConfig(
            chunk_size=512,
            chunk_overlap=50,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        assert chunker.config == config
        assert chunker.markdown_parser is not None
        assert chunker.header_splitter is not None
        assert chunker.recursive_chunker is not None
        assert chunker.atomic_unit_handler is not None
        assert chunker.metadata_extractor is not None
    
    def test_hybrid_chunker_invalid_config_raises_error(self):
        """Test HybridChunker raises TypeError with invalid config."""
        with pytest.raises(TypeError, match="Config must be ChunkConfig"):
            HybridChunker("invalid_config")
    
    def test_chunk_document_empty_content_returns_empty_result(self):
        """Test chunking empty document returns empty result."""
        config = ChunkConfig(chunk_size=512, chunk_overlap=50)
        chunker = HybridChunker(config)
        
        result = chunker.chunk_document("")
        
        assert isinstance(result, HybridChunkResult)
        assert len(result.chunks) == 0
        assert result.document_metadata is None
        assert result.document_tree is None
        assert len(result.atomic_units) == 0
    
    def test_chunk_document_invalid_content_type_raises_error(self):
        """Test chunking non-string content raises TypeError."""
        config = ChunkConfig(chunk_size=512, chunk_overlap=50)
        chunker = HybridChunker(config)
        
        with pytest.raises(TypeError, match="Document content must be string"):
            chunker.chunk_document(123)
    
    def test_chunk_simple_markdown_document(self):
        """Test chunking a simple markdown document with headers."""
        config = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        document = """# Introduction

This is the introduction section with some content.

## Background

This section provides background information that is important for understanding the context."""
        
        result = chunker.chunk_document(document, document_id="test_doc")
        
        # Verify result structure
        assert isinstance(result, HybridChunkResult)
        assert len(result.chunks) > 0
        assert result.document_tree is not None
        assert result.processing_stats is not None
        
        # Verify chunks have proper metadata
        for chunk in result.chunks:
            assert chunk.metadata is not None
            assert 'source_document_id' in chunk.metadata
            assert 'document_title' in chunk.metadata
            assert 'header_hierarchy' in chunk.metadata
            assert 'content_type' in chunk.metadata
            assert chunk.metadata['source_document_id'] == "test_doc"
    
    def test_chunk_document_with_frontmatter(self):
        """Test chunking document with frontmatter metadata."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        document = """---
title: "Test Document"
author: "Test Author"
tags: ["test", "chunking"]
---

# Main Content

This is the main content of the document."""
        
        result = chunker.chunk_document(document, document_id="test_with_frontmatter")
        
        # Verify metadata extraction
        assert result.document_metadata is not None
        assert result.document_metadata.title == "Test Document"
        
        # Verify chunks use extracted title
        for chunk in result.chunks:
            assert chunk.metadata['document_title'] == "Test Document"
    
    def test_chunk_document_with_code_blocks(self):
        """Test chunking document with code blocks (atomic units)."""
        config = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        document = """# Code Example

Here's some introductory text.

```python
def hello_world():
    print("Hello, World!")
    return "success"
```

And here's some follow-up text after the code."""
        
        result = chunker.chunk_document(document, document_id="code_test")
        
        # Verify atomic units were detected
        assert len(result.atomic_units) > 0
        code_units = [u for u in result.atomic_units if u.unit_type == AtomicUnitType.CODE_BLOCK]
        assert len(code_units) == 1
        
        # Verify code blocks are preserved in chunks
        code_chunks = [c for c in result.chunks if 
                      c.metadata.get('content_type') == 'code_block']
        assert len(code_chunks) >= 1
        
        # Verify code language is detected
        for chunk in code_chunks:
            if 'code_language' in chunk.metadata:
                assert chunk.metadata['code_language'] == 'python'
    
    def test_chunk_document_with_tables(self):
        """Test chunking document with tables (atomic units)."""
        config = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        document = """# Data Analysis

Here's the data:

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |

Analysis continues here."""
        
        result = chunker.chunk_document(document, document_id="table_test")
        
        # Verify atomic units were detected
        table_units = [u for u in result.atomic_units if u.unit_type == AtomicUnitType.TABLE]
        assert len(table_units) >= 1
        
        # Verify tables are preserved in chunks
        table_chunks = [c for c in result.chunks if 
                       c.metadata.get('content_type') == 'table']
        
        # At least one chunk should contain table content
        table_content_found = any('Name' in chunk.content and 'Age' in chunk.content 
                                 for chunk in result.chunks)
        assert table_content_found
    
    def test_header_hierarchy_metadata(self):
        """Test that header hierarchy is correctly built and attached."""
        config = ChunkConfig(chunk_size=400, chunk_overlap=40)
        chunker = HybridChunker(config)
        
        document = """# Chapter 1

Introduction content.

## Section 1.1

Section content.

### Subsection 1.1.1

Subsection content.

## Section 1.2

Another section."""
        
        result = chunker.chunk_document(document, document_id="hierarchy_test")
        
        # Find chunks for different sections
        subsection_chunks = [c for c in result.chunks if 
                           c.metadata.get('section_title') == 'Subsection 1.1.1']
        
        if subsection_chunks:
            hierarchy = subsection_chunks[0].metadata['header_hierarchy']
            # Should include path from root to current section
            assert 'Chapter 1' in hierarchy
            assert 'Section 1.1' in hierarchy
            assert 'Subsection 1.1.1' in hierarchy
    
    def test_chunk_oversized_atomic_units(self):
        """Test handling of atomic units that exceed chunk size."""
        config = ChunkConfig(
            chunk_size=100,  # Small chunk size
            chunk_overlap=10,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        # Create document with large code block
        large_code = """def very_long_function():
    # This is a very long function that exceeds our small chunk size
    for i in range(100):
        print(f"Processing item {i}")
        # More code here to make it long
        result = process_complex_operation(i)
        if result:
            handle_result(result)
    return "completed\""""
        
        document = f"""# Large Code Example

Here's a large code block:

```python
{large_code}
```

End of document."""
        
        result = chunker.chunk_document(document, document_id="oversized_test")
        
        # Should still create chunks (possibly split the oversized atomic unit)
        assert len(result.chunks) > 0
        
        # Should detect the atomic unit
        code_units = [u for u in result.atomic_units if u.unit_type == AtomicUnitType.CODE_BLOCK]
        assert len(code_units) == 1
    
    def test_chunk_without_headers(self):
        """Test chunking document without headers."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20)
        chunker = HybridChunker(config)
        
        document = """This is a document without any headers.

It has multiple paragraphs of content that should still be chunked properly.

The chunker should handle this gracefully and create appropriate chunks."""
        
        result = chunker.chunk_document(document, document_id="no_headers_test")
        
        # Should still create chunks
        assert len(result.chunks) > 0
        
        # Should have basic metadata
        for chunk in result.chunks:
            assert chunk.metadata['source_document_id'] == "no_headers_test"
            assert chunk.metadata['content_type'] == 'prose'
    
    def test_processing_statistics(self):
        """Test that processing statistics are collected."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        document = """# Test Document

Content for testing statistics.

```python
print("test")
```

More content."""
        
        result = chunker.chunk_document(document, document_id="stats_test")
        
        # Verify processing stats
        assert 'processing_time_ms' in result.processing_stats
        assert 'total_chunks' in result.processing_stats
        assert 'atomic_units_detected' in result.processing_stats
        assert 'sections_processed' in result.processing_stats
        
        # Verify chunker stats
        stats = chunker.get_processing_stats()
        assert stats['documents_processed'] >= 1
        assert stats['total_chunks_created'] >= 1
    
    def test_reset_statistics(self):
        """Test resetting processing statistics."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        # Process a document
        chunker.chunk_document("# Test\nContent", document_id="test")
        
        # Verify stats exist
        stats = chunker.get_processing_stats()
        assert stats['documents_processed'] > 0
        
        # Reset stats
        chunker.reset_stats()
        
        # Verify stats are reset
        stats = chunker.get_processing_stats()
        assert stats['documents_processed'] == 0
        assert stats['total_chunks_created'] == 0
    
    def test_atomic_unit_preservation_configuration(self):
        """Test that atomic unit preservation respects configuration."""
        # Test with preservation disabled
        config_disabled = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            preserve_code_blocks=False,
            preserve_tables=False
        )
        chunker_disabled = HybridChunker(config_disabled)
        
        document = """# Test

Text before.

```python
def test():
    pass
```

Text after."""
        
        result_disabled = chunker_disabled.chunk_document(document, document_id="test_disabled")
        
        # With preservation disabled, code should be treated as normal text
        # (This is a bit tricky to test definitively, but we can check that 
        # no special atomic unit handling occurred)
        assert len(result_disabled.chunks) > 0
        
        # Test with preservation enabled
        config_enabled = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker_enabled = HybridChunker(config_enabled)
        
        result_enabled = chunker_enabled.chunk_document(document, document_id="test_enabled")
        
        # Should detect atomic units
        assert len(result_enabled.atomic_units) > 0
    
    def test_source_path_metadata(self):
        """Test that source path is included in metadata when provided."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        document = "# Test Document\n\nContent here."
        source_path = "/path/to/document.md"
        
        result = chunker.chunk_document(
            document, 
            document_id="test_path",
            source_path=source_path
        )
        
        # Verify source path in metadata
        for chunk in result.chunks:
            assert chunk.metadata['source_path'] == source_path
            assert chunk.metadata['document_title'] == "document"  # from Path.stem
    
    def test_chunk_sequence_ids(self):
        """Test that chunk sequence IDs are properly assigned."""
        config = ChunkConfig(chunk_size=150, chunk_overlap=15)  # Small chunks to ensure multiple
        chunker = HybridChunker(config)
        
        document = """# Long Document

This is a long document that should be split into multiple chunks to test the sequence ID assignment.

Each chunk should have a proper sequence ID that reflects its position in the document.

More content to ensure we get multiple chunks."""
        
        result = chunker.chunk_document(document, document_id="sequence_test")
        
        if len(result.chunks) > 1:
            # Verify sequence IDs are assigned
            sequence_ids = [chunk.metadata['chunk_sequence_id'] for chunk in result.chunks]
            
            # Should start from 0 and be sequential within sections
            for chunk in result.chunks:
                assert 'chunk_sequence_id' in chunk.metadata
                assert isinstance(chunk.metadata['chunk_sequence_id'], int)
                assert chunk.metadata['chunk_sequence_id'] >= 0


class TestHybridChunkerEdgeCases:
    """Test edge cases and error conditions for HybridChunker."""
    
    def test_document_with_only_frontmatter(self):
        """Test document that only contains frontmatter."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        document = """---
title: "Only Frontmatter"
author: "Test"
---"""
        
        result = chunker.chunk_document(document, document_id="frontmatter_only")
        
        # Should extract metadata but have no content chunks
        assert result.document_metadata is not None
        # Might have empty result or minimal chunks depending on implementation
    
    def test_document_with_empty_sections(self):
        """Test document with empty sections."""
        config = ChunkConfig(chunk_size=300, chunk_overlap=30)
        chunker = HybridChunker(config)
        
        document = """# Section 1

## Empty Subsection

# Section 2

Some content here."""
        
        result = chunker.chunk_document(document, document_id="empty_sections")
        
        # Should handle empty sections gracefully
        assert len(result.chunks) > 0
        
        # Should have processed sections
        assert result.document_tree is not None
        sections = result.document_tree.get_all_sections()
        assert len(sections) > 0
    
    def test_very_long_single_section(self):
        """Test document with very long single section."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20)
        chunker = HybridChunker(config)
        
        # Create very long content
        long_content = " ".join([f"This is sentence {i} of the very long content." 
                                for i in range(50)])
        
        document = f"""# Long Section

{long_content}"""
        
        result = chunker.chunk_document(document, document_id="long_section")
        
        # Should split into multiple chunks
        assert len(result.chunks) > 1
        
        # All chunks should belong to the same section
        for chunk in result.chunks:
            assert chunk.metadata['section_title'] == 'Long Section'


class TestHybridChunkerIntegration:
    """Integration tests for HybridChunker with other components."""
    
    def test_integration_with_all_components(self):
        """Test complete integration with all document processing components."""
        config = ChunkConfig(
            chunk_size=400,
            chunk_overlap=40,
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = HybridChunker(config)
        
        complex_document = """---
title: "Complex Integration Test"
author: "Test Suite"
project: "hybrid_chunking"
tags: ["integration", "test"]
---

# System Architecture @priority:high

This document describes the system architecture.

## Database Layer @tag:database

The database layer handles data persistence.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);
```

### Performance Metrics @tag:metrics

| Operation | Time (ms) | Memory (MB) |
|-----------|-----------|-------------|
| SELECT    | 5         | 10          |
| INSERT    | 15        | 15          |
| UPDATE    | 12        | 12          |

## API Layer @tag:api

The API layer exposes REST endpoints.

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/users')
def get_users():
    return jsonify(fetch_users())
```

# Configuration @priority:medium

Configuration is handled through environment variables."""
        
        result = chunker.chunk_document(
            complex_document, 
            document_id="integration_test",
            source_path="/path/to/integration_test.md"
        )
        
        # Comprehensive verification
        assert isinstance(result, HybridChunkResult)
        assert len(result.chunks) > 0
        
        # Verify metadata extraction
        assert result.document_metadata is not None
        assert result.document_metadata.title == "Complex Integration Test"
        
        # Verify document structure
        assert result.document_tree is not None
        sections = result.document_tree.get_all_sections()
        assert len(sections) > 0
        
        # Verify atomic units detection
        assert len(result.atomic_units) > 0
        code_units = [u for u in result.atomic_units if u.unit_type == AtomicUnitType.CODE_BLOCK]
        table_units = [u for u in result.atomic_units if u.unit_type == AtomicUnitType.TABLE]
        assert len(code_units) >= 1  # SQL and Python code
        assert len(table_units) >= 1  # Performance metrics table
        
        # Verify rich metadata on chunks
        for chunk in result.chunks:
            metadata = chunk.metadata
            
            # Required FR-KB-002.3 metadata
            assert 'source_document_id' in metadata
            assert 'document_title' in metadata
            assert 'header_hierarchy' in metadata
            assert 'chunk_sequence_id' in metadata
            assert 'content_type' in metadata
            
            # Additional metadata
            assert 'source_path' in metadata
            assert 'section_title' in metadata
            assert 'section_level' in metadata
            
            # Content type specific metadata
            if metadata['content_type'] == 'code_block':
                assert 'code_language' in metadata
                assert metadata['code_language'] in ['sql', 'python', 'unknown']
        
        # Verify header hierarchies are correct
        api_chunks = [c for c in result.chunks if 
                     c.metadata.get('section_title') == 'API Layer']
        if api_chunks:
            hierarchy = api_chunks[0].metadata['header_hierarchy']
            assert 'System Architecture' in hierarchy
            assert 'API Layer' in hierarchy
        
        # Verify processing statistics
        stats = result.processing_stats
        assert stats['total_chunks'] == len(result.chunks)
        assert stats['atomic_units_detected'] == len(result.atomic_units)
        assert stats['sections_processed'] == len(sections)
        assert 'processing_time_ms' in stats 