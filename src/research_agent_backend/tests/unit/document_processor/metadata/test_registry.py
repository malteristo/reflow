"""Tests for metadata registry and query system."""

import pytest
from core.document_processor import MetadataRegistry, DocumentMetadata


class TestMetadataRegistry:
    """Tests for metadata registry and query system."""
    
    def test_metadata_registry_creation(self):
        """Test creating a MetadataRegistry."""
        registry = MetadataRegistry()
        assert isinstance(registry, MetadataRegistry)
    
    def test_register_document_metadata(self):
        """Test registering document metadata in registry."""
        registry = MetadataRegistry()
        
        metadata = DocumentMetadata(
            document_id="doc1",
            title="Test Document",
            author="John Doe",
            tags=["python", "testing"],
            frontmatter={"version": "1.0", "status": "draft"},
            inline_metadata={"priority": "high", "category": "technical"}
        )
        
        registry.register(metadata)
        assert registry.get_document_count() == 1
        assert registry.has_document("doc1") == True
    
    def test_query_by_tags(self):
        """Test querying documents by tags."""
        registry = MetadataRegistry()
        
        # Register multiple documents
        doc1 = DocumentMetadata(
            document_id="doc1",
            title="Python Guide",
            tags=["python", "tutorial"]
        )
        doc2 = DocumentMetadata(
            document_id="doc2", 
            title="Testing Guide",
            tags=["python", "testing"]
        )
        doc3 = DocumentMetadata(
            document_id="doc3",
            title="JavaScript Guide", 
            tags=["javascript", "tutorial"]
        )
        
        registry.register(doc1)
        registry.register(doc2)
        registry.register(doc3)
        
        # Query by tag
        python_docs = registry.query_by_tags(["python"])
        assert len(python_docs) == 2
        assert "doc1" in [doc.document_id for doc in python_docs]
        assert "doc2" in [doc.document_id for doc in python_docs]
        
        tutorial_docs = registry.query_by_tags(["tutorial"])
        assert len(tutorial_docs) == 2
        assert "doc1" in [doc.document_id for doc in tutorial_docs]
        assert "doc3" in [doc.document_id for doc in tutorial_docs]
    
    def test_query_by_metadata_key_value(self):
        """Test querying documents by metadata key-value pairs."""
        registry = MetadataRegistry()
        
        doc1 = DocumentMetadata(
            document_id="doc1",
            frontmatter={"status": "published", "priority": "high"}
        )
        doc2 = DocumentMetadata(
            document_id="doc2",
            frontmatter={"status": "draft", "priority": "high"}
        )
        doc3 = DocumentMetadata(
            document_id="doc3",
            frontmatter={"status": "published", "priority": "low"}
        )
        
        registry.register(doc1)
        registry.register(doc2)
        registry.register(doc3)
        
        # Query by frontmatter status
        published_docs = registry.query_by_metadata("frontmatter.status", "published")
        assert len(published_docs) == 2
        
        # Query by frontmatter priority
        high_priority_docs = registry.query_by_metadata("frontmatter.priority", "high")
        assert len(high_priority_docs) == 2
    
    def test_query_by_author(self):
        """Test querying documents by author."""
        registry = MetadataRegistry()
        
        doc1 = DocumentMetadata(document_id="doc1", author="Alice")
        doc2 = DocumentMetadata(document_id="doc2", author="Bob")
        doc3 = DocumentMetadata(document_id="doc3", author="Alice")
        
        registry.register(doc1)
        registry.register(doc2)
        registry.register(doc3)
        
        alice_docs = registry.query_by_author("Alice")
        assert len(alice_docs) == 2
        assert all(doc.author == "Alice" for doc in alice_docs)
    
    def test_complex_query_combinations(self):
        """Test complex query combinations with multiple criteria."""
        registry = MetadataRegistry()
        
        # Register documents with rich metadata
        docs = [
            DocumentMetadata(
                document_id="doc1",
                author="Alice",
                tags=["python", "advanced"],
                frontmatter={"status": "published", "type": "tutorial"}
            ),
            DocumentMetadata(
                document_id="doc2",
                author="Bob", 
                tags=["python", "beginner"],
                frontmatter={"status": "draft", "type": "tutorial"}
            ),
            DocumentMetadata(
                document_id="doc3",
                author="Alice",
                tags=["javascript", "advanced"],
                frontmatter={"status": "published", "type": "guide"}
            )
        ]
        
        for doc in docs:
            registry.register(doc)
        
        # Complex query: Alice's published tutorials
        results = registry.query_complex(
            author="Alice",
            tags_any=["python", "javascript"],
            metadata_filters={"frontmatter.status": "published"}
        )
        
        assert len(results) >= 1
        assert all(doc.author == "Alice" for doc in results)
    
    def test_metadata_aggregation(self):
        """Test metadata aggregation and statistics."""
        registry = MetadataRegistry()
        
        docs = [
            DocumentMetadata(document_id="doc1", tags=["python"], author="Alice"),
            DocumentMetadata(document_id="doc2", tags=["python", "testing"], author="Bob"),
            DocumentMetadata(document_id="doc3", tags=["javascript"], author="Alice"),
        ]
        
        for doc in docs:
            registry.register(doc)
        
        stats = registry.get_statistics()
        
        assert stats["total_documents"] == 3
        assert stats["unique_authors"] == 2
        assert "python" in stats["tag_frequencies"]
        assert stats["tag_frequencies"]["python"] == 2
        assert "Alice" in stats["author_frequencies"]
        assert stats["author_frequencies"]["Alice"] == 2
    
    def test_registry_update_and_removal(self):
        """Test updating and removing documents from registry."""
        registry = MetadataRegistry()
        
        # Register initial document
        doc = DocumentMetadata(
            document_id="doc1",
            title="Original Title",
            tags=["old_tag"]
        )
        registry.register(doc)
        
        # Update document
        updated_doc = DocumentMetadata(
            document_id="doc1",
            title="Updated Title",
            tags=["new_tag"]
        )
        registry.update(updated_doc)
        
        retrieved = registry.get_document("doc1")
        assert retrieved.title == "Updated Title"
        assert "new_tag" in retrieved.tags
        
        # Remove document
        registry.remove("doc1")
        assert registry.has_document("doc1") == False
        assert registry.get_document_count() == 0 