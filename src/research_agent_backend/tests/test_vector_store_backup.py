"""
Unit tests for ChromaDB Vector Store implementation.

Tests the ChromaDBManager class functionality including database initialization,
collection management, document insertion, and querying operations.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from src.research_agent_backend.core.vector_store import (
    ChromaDBManager,
    create_chroma_manager,
    get_default_collection_types,
)
from src.research_agent_backend.exceptions.vector_store_exceptions import (
    VectorStoreError,
    CollectionNotFoundError,
    DocumentInsertionError,
    QueryError,
)
from src.research_agent_backend.utils.config import ConfigManager


class TestChromaDBManager:
    """Test suite for ChromaDBManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Use in-memory database for testing
        self.manager = ChromaDBManager(in_memory=True)
        
        # Sample test data
        self.test_chunks = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are used in deep learning applications."
        ]
        
        # Sample embeddings (simplified for testing)
        self.test_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        
        self.test_metadata = [
            {"source": "doc1.md", "section": "intro"},
            {"source": "doc1.md", "section": "overview"},
            {"source": "doc2.md", "section": "technical"}
        ]
        
        # Counter for unique collection names
        import time
        self.test_counter = int(time.time() * 1000000) % 1000000
    
    def teardown_method(self):
        """Clean up after each test method."""
        if self.manager and self.manager.is_connected:
            try:
                # Reset database to clean up all collections
                self.manager.reset_database()
            except Exception:
                pass  # Ignore errors during cleanup
            self.manager.close()
    
    def get_unique_collection_name(self, base_name: str = "test_collection") -> str:
        """Generate a unique collection name for each test."""
        self.test_counter += 1
        return f"{base_name}_{self.test_counter}"
    
    def test_initialization(self):
        """Test ChromaDBManager initialization."""
        assert self.manager is not None
        assert not self.manager.is_connected  # Not connected until first use
        
        # Test lazy initialization
        client = self.manager.client
        assert client is not None
        assert self.manager.is_connected
    
    def test_health_check(self):
        """Test database health check functionality."""
        health = self.manager.health_check()
        
        assert health['status'] == 'healthy'
        assert health['connected'] is True
        assert health['collections_count'] == 0  # No collections initially
        assert isinstance(health['collections'], list)
        assert 'timestamp' in health
        assert health['errors'] == []
    
    def test_create_collection(self):
        """Test collection creation."""
        collection_name = self.get_unique_collection_name()
        metadata = {"type": "test", "description": "Test collection"}
        
        collection = self.manager.create_collection(
            name=collection_name,
            metadata=metadata
        )
        
        assert collection is not None
        assert collection.name == collection_name
        
        # Verify collection appears in listings
        collections = self.manager.list_collections()
        assert len(collections) >= 1
        collection_names = [c['name'] for c in collections]
        assert collection_name in collection_names
    
    def test_get_collection(self):
        """Test getting an existing collection."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection first
        self.manager.create_collection(collection_name)
        
        # Get collection
        collection = self.manager.get_collection(collection_name)
        assert collection.name == collection_name
        
        # Test getting non-existent collection
        with pytest.raises(CollectionNotFoundError):
            self.manager.get_collection("nonexistent_collection")
    
    def test_delete_collection(self):
        """Test collection deletion."""
        collection_name = self.get_unique_collection_name()
        
        # Create and then delete collection
        self.manager.create_collection(collection_name)
        collections_before = len(self.manager.list_collections())
        assert collections_before >= 1
        
        self.manager.delete_collection(collection_name)
        collections_after = len(self.manager.list_collections())
        assert collections_after == collections_before - 1
        
        # Test deleting non-existent collection
        with pytest.raises(CollectionNotFoundError):
            self.manager.delete_collection("nonexistent_collection")
    
    def test_add_documents(self):
        """Test adding documents to a collection."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection
        self.manager.create_collection(collection_name)
        
        # Add documents
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=self.test_chunks,
            embeddings=self.test_embeddings,
            metadata=self.test_metadata
        )
        
        # Verify documents were added
        stats = self.manager.get_collection_stats(collection_name)
        assert stats['document_count'] == len(self.test_chunks)
    
    def test_add_documents_validation(self):
        """Test document addition input validation."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        # Test mismatched chunks and embeddings
        with pytest.raises(DocumentInsertionError):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=self.test_chunks,
                embeddings=self.test_embeddings[:2]  # Fewer embeddings
            )
        
        # Test adding to non-existent collection
        with pytest.raises(CollectionNotFoundError):
            self.manager.add_documents(
                collection_name="nonexistent_collection",
                chunks=self.test_chunks,
                embeddings=self.test_embeddings
            )
    
    def test_query_collection(self):
        """Test querying a collection."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection and add documents
        self.manager.create_collection(collection_name)
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=self.test_chunks,
            embeddings=self.test_embeddings,
            metadata=self.test_metadata
        )
        
        # Query collection
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = self.manager.query_collection(
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=2
        )
        
        assert 'ids' in results
        assert 'documents' in results
        assert 'metadatas' in results
        assert 'distances' in results
        assert len(results['ids']) <= 2  # Requested k=2
        assert results['collection'] == collection_name
    
    def test_query_nonexistent_collection(self):
        """Test querying a non-existent collection."""
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with pytest.raises(CollectionNotFoundError):
            self.manager.query_collection(
                collection_name="nonexistent_collection",
                query_embedding=query_embedding
            )
    
    def test_get_collection_stats(self):
        """Test getting collection statistics."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection and add documents
        self.manager.create_collection(collection_name)
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=self.test_chunks,
            embeddings=self.test_embeddings
        )
        
        stats = self.manager.get_collection_stats(collection_name)
        
        assert stats['name'] == collection_name
        assert stats['document_count'] == len(self.test_chunks)
        assert 'id' in stats
        assert 'metadata' in stats
        assert 'timestamp' in stats


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_chroma_manager(self):
        """Test the factory function for creating ChromaDBManager."""
        manager = create_chroma_manager(in_memory=True)
        
        assert isinstance(manager, ChromaDBManager)
        assert manager.persist_directory is None  # In-memory
        
        # Test with persistent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager_persistent = create_chroma_manager(
                persist_directory=temp_dir,
                in_memory=False
            )
            assert manager_persistent.persist_directory == temp_dir
    
    def test_get_default_collection_types(self):
        """Test getting default collection type configurations."""
        collection_types = get_default_collection_types()
        
        assert isinstance(collection_types, dict)
        assert 'fundamental' in collection_types
        assert 'project-specific' in collection_types
        assert 'general' in collection_types
        
        # Verify structure
        for type_name, config in collection_types.items():
            assert 'description' in config
            assert 'metadata' in config
            assert 'type' in config['metadata']
            assert 'searchable' in config['metadata']
            assert 'priority' in config['metadata']


# Integration test with configuration
class TestConfigurationIntegration:
    """Test ChromaDBManager integration with configuration system."""
    
    def test_with_config_manager(self):
        """Test ChromaDBManager with explicit ConfigManager."""
        # Create a minimal config for testing
        config_manager = ConfigManager()
        
        manager = ChromaDBManager(
            config_manager=config_manager,
            in_memory=True
        )
        
        assert manager.config_manager is config_manager
        
        # Test configuration access
        vector_config = manager.config_manager.get('vector_store', {})
        assert isinstance(vector_config, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 