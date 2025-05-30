"""Tests for integration between metadata extraction components."""

import pytest
from research_agent_backend.core.document_processor import (
    MetadataExtractor, MetadataRegistry, MetadataQuery,
    HeaderBasedSplitter, MarkdownParser
)


class TestMetadataIntegration:
    """Tests for integration between metadata extraction components."""
    
    def test_full_metadata_extraction_pipeline(self):
        """Test complete metadata extraction from document to registry."""
        content = """---
title: "Integration Test Document"
author: "Test Author"
tags: ["integration", "testing"]
version: 1.0
---

# Integration Test

This document has @tag:important and @priority:high inline metadata.

<!-- @status: ready_for_review -->

The content includes various metadata sources."""
        
        extractor = MetadataExtractor()
        registry = MetadataRegistry()
        
        # Extract all metadata
        result = extractor.extract_all(content, document_id="integration_test")
        
        # Register in registry
        registry.register(result.document_metadata)
        
        # Verify complete extraction
        doc = registry.get_document("integration_test")
        assert doc.title == "Integration Test Document"
        assert doc.author == "Test Author"
        assert "integration" in doc.tags
        assert doc.frontmatter["version"] == 1.0
        assert "important" in [tag.value for tag in doc.inline_tags if tag.key == "tag"]
        assert doc.inline_metadata["status"] == "ready_for_review"
    
    def test_metadata_extraction_with_document_chunking(self):
        """Test metadata extraction integration with document chunking pipeline."""
        content = """---
title: "Chunked Document"
project: "metadata_system"
---

# Main Section @tag:section1

Content for first section.

## Subsection @priority:medium

More content here.

# Second Section @tag:section2

Final content."""
        
        # Extract metadata first
        extractor = MetadataExtractor()
        metadata_result = extractor.extract_all(content, document_id="chunked_test_doc")
        
        # Then process with existing chunking pipeline
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        tree = splitter.split_and_build_tree(metadata_result.content_without_frontmatter)
        
        # Verify that both metadata and structure are preserved
        assert metadata_result.document_metadata.title == "Chunked Document"
        assert len(tree.root.children) >= 1  # Should have sections
        
        # Verify inline metadata was extracted from sections
        section1_tags = [tag for tag in metadata_result.document_metadata.inline_tags if tag.value == "section1"]
        assert len(section1_tags) > 0
    
    def test_metadata_search_and_retrieval(self):
        """Test searching and retrieving documents based on metadata."""
        # Create documents with different metadata
        docs_content = [
            """---
title: "Python Tutorial"
author: "Alice"
difficulty: beginner
tags: ["python", "tutorial"]
---
# Learning Python
Content about Python basics.""",
            
            """---
title: "Advanced Python"
author: "Bob"  
difficulty: advanced
tags: ["python", "advanced"]
---
# Advanced Python Concepts
Complex Python topics.""",
            
            """---
title: "JavaScript Guide"
author: "Alice"
difficulty: intermediate  
tags: ["javascript", "web"]
---
# JavaScript Fundamentals
Web development with JS."""
        ]
        
        extractor = MetadataExtractor()
        registry = MetadataRegistry()
        
        # Process all documents
        for i, content in enumerate(docs_content):
            result = extractor.extract_all(content, document_id=f"doc{i+1}")
            registry.register(result.document_metadata)
        
        # Test various search scenarios
        query = MetadataQuery()
        
        # Find all Python documents
        python_docs = query.find_by_tags(registry, ["python"])
        assert len(python_docs) == 2
        
        # Find Alice's documents
        alice_docs = query.find_by_author(registry, "Alice")
        assert len(alice_docs) == 2
        
        # Find beginner-level documents
        beginner_docs = query.find_by_metadata(registry, "frontmatter.difficulty", "beginner")
        assert len(beginner_docs) == 1
        assert beginner_docs[0].title == "Python Tutorial" 