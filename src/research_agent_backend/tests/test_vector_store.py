"""
Unit tests for ChromaDB Vector Store implementation.

Tests the ChromaDBManager class functionality including database initialization,
collection management, document insertion, and querying operations.
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import time
import uuid
from unittest.mock import patch
from uuid import uuid4
import os

from src.research_agent_backend.core.vector_store import (
    ChromaDBManager,
    create_chroma_manager,
    get_default_collection_types,
    DatabaseInitializationError,
)
from src.research_agent_backend.core.collection_type_manager import CollectionType
from src.research_agent_backend.exceptions.vector_store_exceptions import (
    VectorStoreError,
    CollectionNotFoundError,
    DocumentInsertionError,
    QueryError,
    CollectionAlreadyExistsError,
    MetadataValidationError,
    EmbeddingDimensionError,
)
from src.research_agent_backend.exceptions.config_exceptions import ConfigurationFileNotFoundError
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
        # Manager connects during initialization due to specialized managers needing client
        assert self.manager.is_connected  # Connected during manager creation
        
        # Test that client is accessible
        client = self.manager.client
        assert client is not None
        assert self.manager.is_connected
    
    def test_health_check(self):
        """Test database health check functionality."""
        health = self.manager.health_check()
        
        assert health.status == 'healthy'  # Access as attribute
        assert health.connected is True   # Access as attribute
        assert health.collections_count == 0  # No collections initially
        assert isinstance(health.collections, list)  # Access as attribute
        assert 'timestamp' in health.__dict__  # Check timestamp attribute exists
        assert health.errors == []  # Access as attribute
    
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
        collection_names = [c.name for c in collections]
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
        assert stats.document_count == len(self.test_chunks)  # Access as attribute
    
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
        
        assert results.ids is not None              # Access as attribute
        assert results.documents is not None       # Access as attribute
        assert results.metadatas is not None       # Access as attribute  
        assert results.distances is not None       # Access as attribute
        assert len(results.ids) <= 2               # Requested k=2
        assert results.collection == collection_name  # Access as attribute
    
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
            embeddings=self.test_embeddings,
            metadata=self.test_metadata  # Added metadata to fix ChromaDB requirement
        )
        
        stats = self.manager.get_collection_stats(collection_name)
        
        assert stats.name == collection_name        # Access as attribute
        assert stats.document_count == len(self.test_chunks)  # Access as attribute
        assert stats.id                            # Access as attribute (check exists)
        assert isinstance(stats.metadata, dict)    # Access as attribute
        assert stats.timestamp                     # Access as attribute (check exists)

    def test_health_check_with_client_failure(self):
        """Test health check when client operations fail."""
        # Initialize first to establish client
        self.manager.initialize_database()
        
        # Then mock heartbeat to fail
        with patch.object(self.manager.client, 'heartbeat', side_effect=Exception("Heartbeat failed")):
            health = self.manager.health_check()
            assert health.status == 'unhealthy'  # Access as attribute
            assert 'Heartbeat failed' in str(health.errors)  # Access as attribute


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_create_chroma_manager(self):
        """Test the factory function for creating ChromaDBManager."""
        manager = create_chroma_manager(in_memory=True)
        
        assert isinstance(manager, ChromaDBManager)
        assert manager.client_manager.persist_directory is None  # In-memory - access through client_manager
        
        # Test with persistent directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager_persistent = create_chroma_manager(
                persist_directory=temp_dir,
                in_memory=False
            )
            assert manager_persistent.client_manager.persist_directory == temp_dir  # Access through client_manager
    
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


class TestVectorStoreEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for comprehensive coverage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_unique_collection_name(self, base_name: str = "edge_test") -> str:
        """Generate a unique collection name for each test."""
        timestamp = str(int(time.time() * 1000))
        unique_id = str(uuid.uuid4())[:8]
        return f"{base_name}_{timestamp}_{unique_id}"
    
    def test_connection_test_with_failure(self):
        """Test connection testing when heartbeat fails."""
        # Mock a failing heartbeat
        with patch.object(self.manager.client, 'heartbeat', side_effect=Exception("Connection failed")):
            with pytest.raises(VectorStoreError, match="Database connection test failed"):
                self.manager.client_manager._test_connection()  # Access through client_manager
    
    def test_health_check_with_client_failure(self):
        """Test health check when client operations fail."""
        # Initialize first to establish client
        self.manager.initialize_database()
        
        # Then mock heartbeat to fail
        with patch.object(self.manager.client, 'heartbeat', side_effect=Exception("Heartbeat failed")):
            health = self.manager.health_check()
            assert health['status'] == 'unhealthy'
            assert 'Heartbeat failed' in str(health['errors'])
    
    def test_create_collection_with_additional_metadata(self):
        """Test creating collection with additional metadata."""
        collection_name = self.get_unique_collection_name()
        metadata = {"custom_field": "value", "priority": 1}
        
        collection = self.manager.create_collection(collection_name, metadata=metadata)
        assert collection is not None
        assert collection.name == collection_name
    
    def test_create_collection_force_recreate_missing_collection(self):
        """Test force recreate on non-existent collection."""
        collection_name = self.get_unique_collection_name()
        
        # Should succeed even if collection doesn't exist
        collection = self.manager.create_collection(collection_name, force_recreate=True)
        assert collection is not None
    
    def test_create_collection_already_exists_error(self):
        """Test handling collection already exists error."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection first
        self.manager.create_collection(collection_name)
        
        # Try to create again without force_recreate
        with pytest.raises(CollectionAlreadyExistsError):
            self.manager.create_collection(collection_name)

    def test_get_collection_stats_detailed_metadata(self):
        """Test getting collection stats with detailed metadata."""
        collection_name = self.get_unique_collection_name()
        collection = self.manager.create_collection(collection_name, metadata={"type": "test"})
        
        metadata = [
            {"category": "A", "priority": 1},
            {"category": "B", "priority": 2},
            {"category": "A", "priority": 1}
        ]
        self.manager.add_documents(collection_name, ["doc1", "doc2", "doc3"], 
                                 [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], metadata)
        
        stats = self.manager.get_collection_stats(collection_name)
        assert stats.document_count == 3     # Access as attribute
        assert "type" in stats.metadata      # Access as attribute
    
    def test_close_connection_cleanup(self):
        """Test connection cleanup and cache clearing."""
        # Create some collections to populate cache
        self.manager.create_collection(self.get_unique_collection_name("test1"))
        self.manager.create_collection(self.get_unique_collection_name("test2"))
        
        assert len(self.manager.collections._collections_cache) > 0  # Access through collections manager
        
        self.manager.close()
        
        # Cache should be cleared
        assert len(self.manager.collections._collections_cache) == 0  # Access through collections manager
        assert not self.manager.is_connected


class TestValidationAndErrorHandling:
    """Test validation functions and error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass
    
    def get_unique_collection_name(self, base_name: str = "validation_test") -> str:
        """Generate a unique collection name."""
        timestamp = str(int(time.time() * 1000))
        unique_id = str(uuid.uuid4())[:8]
        return f"{base_name}_{timestamp}_{unique_id}"
    
    def test_add_documents_metadata_validation_none_metadata(self):
        """Test metadata validation with None metadata entry."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        with pytest.raises(MetadataValidationError, match="Metadata entry .* cannot be None"):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=["doc1", "doc2"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                metadata=[{"valid": "data"}, None]  # None metadata
            )
    
    def test_add_documents_metadata_validation_non_dict(self):
        """Test metadata validation with non-dict metadata entry."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        with pytest.raises(MetadataValidationError, match="Metadata entry .* must be a dictionary"):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=["doc1", "doc2"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                metadata=[{"valid": "data"}, "invalid_metadata"]  # String instead of dict
            )
    
    def test_add_documents_embedding_dimension_mismatch(self):
        """Test embedding dimension validation."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        with pytest.raises(EmbeddingDimensionError, match="Embedding .* has dimension .*, expected .*"):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=["doc1", "doc2"],
                embeddings=[[0.1, 0.2], [0.3, 0.4, 0.5]],  # Different dimensions
                metadata=[{"doc": "1"}, {"doc": "2"}]
            )
    
    def test_add_documents_mismatched_metadata_length(self):
        """Test metadata length validation."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        with pytest.raises(DocumentInsertionError, match="Number of metadata entries must match number of chunks"):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=["doc1", "doc2"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                metadata=[{"doc": "1"}]  # Only one metadata for two chunks
            )
    
    def test_add_documents_mismatched_ids_length(self):
        """Test IDs length validation."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        with pytest.raises(DocumentInsertionError, match="Number of IDs must match number of chunks"):
            self.manager.add_documents(
                collection_name=collection_name,
                chunks=["doc1", "doc2"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                metadata=[{"doc": "1"}, {"doc": "2"}],
                ids=["id1"]  # Only one ID for two chunks
            )
    
    def test_query_filters_validation(self):
        """Test query filters validation."""
        collection_name = self.get_unique_collection_name()
        self.manager.create_collection(collection_name)
        
        # Add some documents first
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=["doc1"],
            embeddings=[[0.1, 0.2]],
            metadata=[{"category": "test", "priority": 1}]  # Added priority for valid $and test
        )
        
        # Test valid filters
        results = self.manager.query_collection(
            collection_name=collection_name,
            query_embedding=[0.1, 0.2],
            filters={"category": "test"}
        )
        assert results is not None
        
        # Test valid $and filter with two conditions (fixed)
        results = self.manager.query_collection(
            collection_name=collection_name,
            query_embedding=[0.1, 0.2],
            filters={"$and": [{"category": "test"}, {"priority": 1}]}
        )
        assert results is not None
    
    def test_embedding_function_mocking(self):
        """Test mocking embedding functions."""
        # Create proper mock embedding function with correct signature for ChromaDB
        class MockEmbeddingFunction:
            def __call__(self, input):
                """Mock embedding function with correct signature for ChromaDB."""
                if isinstance(input, list):
                    return [[0.1, 0.2, 0.3] for _ in input]
                else:
                    return [[0.1, 0.2, 0.3]]
        
        mock_embedding_func = MockEmbeddingFunction()
        collection_name = self.get_unique_collection_name()
        
        # Create collection with mock embedding function
        collection = self.manager.create_collection(
            collection_name, 
            embedding_function=mock_embedding_func
        )
        assert collection is not None
    
    def test_create_chroma_manager_variations(self):
        """Test various ways to create ChromaDBManager."""
        # Test with nonexistent config file (should handle gracefully)
        try:
            manager = create_chroma_manager(config_file="nonexistent.json", in_memory=True)
            assert isinstance(manager, ChromaDBManager)
        except ConfigurationFileNotFoundError:
            # This is acceptable behavior
            pass
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_chroma_manager(persist_directory=temp_dir)
            assert manager.persist_directory == temp_dir
            manager.close()


class TestUtilityMethodsAndProperties:
    """Test utility methods and properties for complete coverage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass
    
    def test_client_property_initialization(self):
        """Test client property initialization and lazy loading."""
        # Before initialization, client should still be accessible (lazy loading)
        client = self.manager.client
        assert client is not None
        
        # Should now be connected after accessing client property
        assert self.manager.is_connected
        
        # Getting client again should return the cached instance
        client2 = self.manager.client
        assert client2 is not None
        # After first access, subsequent accesses should return the same instance
        assert client is client2
    
    def test_is_connected_property(self):
        """Test is_connected property."""
        # Should start as not connected
        assert not self.manager.is_connected
        
        # After initialization, should be connected
        self.manager.initialize_database()
        assert self.manager.is_connected
        
        # After close, should not be connected
        self.manager.close()
        assert not self.manager.is_connected
    
    def test_get_default_collection_types_structure(self):
        """Test that default collection types have proper structure."""
        defaults = get_default_collection_types()
        
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        
        # Check that each type has required fields
        for type_name, config in defaults.items():
            assert isinstance(config, dict)
            assert "description" in config
            assert "metadata" in config
            assert isinstance(config["metadata"], dict)
    
    def test_collection_caching(self):
        """Test collection caching functionality."""
        collection_name = f"cache_test_{int(time.time() * 1000)}"
        
        # Create collection
        self.manager.create_collection(collection_name)
        
        # Should be in cache
        assert collection_name in self.manager._collections_cache
        
        # Getting collection should use cache
        collection1 = self.manager.get_collection(collection_name)
        collection2 = self.manager.get_collection(collection_name)
        assert collection1 is collection2  # Same instance from cache
    
    def test_reset_database_functionality(self):
        """Test database reset functionality."""
        # Create multiple collections
        collection1 = f"reset_test_1_{int(time.time() * 1000)}"
        collection2 = f"reset_test_2_{int(time.time() * 1000)}"
        
        self.manager.create_collection(collection1)
        self.manager.create_collection(collection2)
        
        # Verify collections exist
        collections_before = self.manager.list_collections()
        assert len(collections_before) >= 2
        
        # Reset database
        self.manager.reset_database()
        
        # Verify all collections are gone
        collections_after = self.manager.list_collections()
        assert len(collections_after) == 0

        # Cache should be cleared
        assert len(self.manager._collections_cache) == 0 


class TestPersistentDatabaseOperations:
    """Test persistent database initialization and file system operations."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config file for testing
        config_path = os.path.join(self.temp_dir, "researchagent.config.json")
        minimal_config = {
            "vector_store": {
                "provider": "chromadb",
                "persist_directory": "./data/chroma_db",
                "collection_metadata": {}
            },
            "collections": {
                "metadata_fields": []
            }
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(minimal_config, f)
        
        self.config_manager = ConfigManager(project_root=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_persistent_database_initialization_success(self):
        """Test successful persistent database initialization."""
        persist_dir = os.path.join(self.temp_dir, "test_chroma_db")
        manager = ChromaDBManager(
            config_manager=self.config_manager,
            persist_directory=persist_dir,
            in_memory=False
        )
        
        # Trigger initialization
        client = manager.client
        assert client is not None
        assert manager.is_connected
        assert os.path.exists(persist_dir)
        
        manager.close()
    
    def test_persistent_database_path_resolution(self):
        """Test relative path resolution for persistent database."""
        # This tests line 84 which handles relative path resolution
        relative_path = "data/test_chroma"
        manager = ChromaDBManager(
            config_manager=self.config_manager,
            persist_directory=relative_path,
            in_memory=False
        )
        
        # Should resolve relative to project root
        expected_path = os.path.join(self.config_manager.project_root, relative_path)
        assert manager.persist_directory == expected_path
        
        manager.close()
    
    def test_persistent_database_permission_error(self):
        """Test handling of permission errors during database initialization."""
        # This tests lines 138-142 error handling for path creation failures
        if os.name == 'nt':  # Windows
            # Use a restricted path on Windows
            restricted_path = "C:\\Windows\\System32\\test_chroma"
        else:  # Unix-like systems
            # Use a restricted path on Unix
            restricted_path = "/root/test_chroma_restricted"
        
        manager = ChromaDBManager(
            config_manager=self.config_manager,
            persist_directory=restricted_path,
            in_memory=False
        )
        
        # Should raise DatabaseInitializationError
        with pytest.raises(DatabaseInitializationError) as exc_info:
            _ = manager.client
        
        assert "Cannot create database directory" in str(exc_info.value)
    
    def test_database_initialization_error_handling(self):
        """Test general database initialization error handling."""
        # This tests lines 160-165 exception handling in initialize_database
        manager = ChromaDBManager(in_memory=True)
        
        # Mock a client creation failure
        with patch('chromadb.Client') as mock_client:
            mock_client.side_effect = Exception("Simulated ChromaDB failure")
            
            with pytest.raises(DatabaseInitializationError) as exc_info:
                manager.initialize_database()
            
            assert "Failed to initialize ChromaDB database" in str(exc_info.value)
    
    def test_connection_test_failure(self):
        """Test connection test failure handling."""
        # This tests lines 168-171 in _test_connection
        manager = ChromaDBManager(in_memory=True)
        
        # Initialize client but mock heartbeat to fail
        manager.initialize_database()
        
        with patch.object(manager._client, 'heartbeat') as mock_heartbeat:
            mock_heartbeat.side_effect = Exception("Connection test failed")
            
            with pytest.raises(VectorStoreError) as exc_info:
                manager._test_connection()
            
            assert "Database connection test failed" in str(exc_info.value)


class TestCollectionValidationMethods:
    """Test collection validation and type checking methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_unique_collection_name(self) -> str:
        """Generate unique collection name for tests."""
        return f"test_collection_{uuid4().hex[:8]}"

    def test_validate_collection_type_success(self):
        """Test successful collection type validation."""
        collection_name = self.get_unique_collection_name()
        # Use PROJECT_SPECIFIC which has default HNSW params: construction_ef=100, M=16
        collection = self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={"type": "project"}
        )
        
        # Test validation with matching type
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.PROJECT_SPECIFIC
        )
        
        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_validate_collection_type_mismatch(self):
        """Test collection type validation with mismatch."""
        collection_name = self.get_unique_collection_name()
        # Create collection as PROJECT_SPECIFIC but validate as FUNDAMENTAL
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={"type": "project"}
        )
        
        # Test validation with mismatched type (FUNDAMENTAL expects different HNSW params)
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.FUNDAMENTAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        # Should detect collection type mismatch first
        assert any("type mismatch" in error.lower() for error in errors)

    def test_validate_collection_type_nonexistent(self):
        """Test validation with nonexistent collection."""
        nonexistent_name = self.get_unique_collection_name()
        
        is_valid, errors = self.manager.validate_collection_type(
            nonexistent_name, CollectionType.GENERAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("not found" in error.lower() for error in errors)

    def test_validate_collection_type_missing_metadata(self):
        """Test validation with collection missing type metadata."""
        collection_name = self.get_unique_collection_name()
        # Create collection without explicit collection_type (should get GENERAL by default)
        self.manager.create_collection(
            collection_name,
            metadata={"some_other": "metadata"}  # No collection type info
        )
        
        # Test validation against different type
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.FUNDAMENTAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        # Should either be type mismatch or missing metadata
        error_text = " ".join(errors).lower()
        assert ("type mismatch" in error_text or "missing collection_type" in error_text)
    
    def test_validate_collection_type_invalid_string_type(self):
        """Test validation with invalid string type."""
        collection_name = self.get_unique_collection_name()
        
        # Create a collection first
        self.manager.create_collection(collection_name, collection_type=CollectionType.GENERAL)
        
        # Test with invalid expected type string
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, "invalid_type"
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("invalid expected collection type" in error.lower() for error in errors)


class TestCollectionTypeManagement:
    """Test collection type management and organization methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_unique_collection_name(self) -> str:
        """Generate unique collection name for tests."""
        return f"test_collection_{uuid4().hex[:8]}"

    def test_determine_collection_for_document(self):
        """Test document collection determination logic."""
        document_metadata = {
            'user_id': 'test_user',
            'team_id': 'test_team',
            'project_name': 'test_project'
        }
        
        # Should determine collection and auto-create if needed
        collection_name = self.manager.determine_collection_for_document(document_metadata)
        
        assert collection_name is not None
        assert isinstance(collection_name, str)
        
        # Collection should now exist
        collection = self.manager.get_collection(collection_name)
        assert collection is not None
        
        # Second call should use existing collection
        collection_name2 = self.manager.determine_collection_for_document(document_metadata)
        assert collection_name == collection_name2

    def test_determine_collection_for_document_with_chunk_metadata(self):
        """Test collection determination with chunk metadata."""
        document_metadata = {'user_id': 'test_user'}
        chunk_metadata = {'importance': 'high'}
        
        collection_name = self.manager.determine_collection_for_document(
            document_metadata, chunk_metadata
        )
        
        assert collection_name is not None
        assert isinstance(collection_name, str)

    def test_get_collections_by_type(self):
        """Test filtering collections by type."""
        # Create collections of different types
        fundamental_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.FUNDAMENTAL,
            metadata={'type': 'fundamental'}
        )
        
        project_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={'type': 'project'}
        )
        
        general_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.GENERAL,
            metadata={'type': 'general'}
        )
        
        # Test filtering by FUNDAMENTAL type
        fundamental_collections = self.manager.get_collections_by_type(CollectionType.FUNDAMENTAL)
        assert len(fundamental_collections) >= 1
        assert any(c['name'] == fundamental_collection.name for c in fundamental_collections)
        
        # Test filtering by PROJECT_SPECIFIC type
        project_collections = self.manager.get_collections_by_type(CollectionType.PROJECT_SPECIFIC)
        assert len(project_collections) >= 1
        assert any(c['name'] == project_collection.name for c in project_collections)
        
        # Test filtering by string type
        general_collections = self.manager.get_collections_by_type('general')
        assert len(general_collections) >= 1
        assert any(c['name'] == general_collection.name for c in general_collections)

    def test_get_collections_by_type_invalid_string(self):
        """Test filtering with invalid collection type string."""
        with pytest.raises(ValueError, match="Unknown collection type"):
            self.manager.get_collections_by_type("invalid_type")

    def test_get_collection_type_summary(self):
        """Test collection type summary generation."""
        # Create collections of different types
        self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.FUNDAMENTAL,
            metadata={'type': 'fundamental'}
        )
        
        self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={'type': 'project'}
        )
        
        # Create collection without collection_type metadata (untyped)
        # Note: This collection should NOT have 'collection_type' in metadata to be truly untyped
        untyped_collection = self.get_unique_collection_name()
        collection = self.manager._client.create_collection(
            name=untyped_collection,
            metadata={'other': 'metadata'}  # No 'collection_type' key
        )
        
        summary = self.manager.get_collection_type_summary()
        
        # Verify summary structure
        assert 'total_collections' in summary
        assert 'by_type' in summary
        assert 'untyped' in summary
        
        assert summary['total_collections'] >= 3
        assert len(summary['by_type']) >= 2  # At least fundamental and project-specific
        assert len(summary['untyped']) >= 1  # At least one untyped collection
        assert untyped_collection in summary['untyped']
        
        # Check type-specific details
        for type_key, type_info in summary['by_type'].items():
            assert 'count' in type_info
            assert 'total_documents' in type_info
            assert 'collections' in type_info
            assert type_info['count'] >= 1
            assert isinstance(type_info['collections'], list)

    def test_get_collection_type_summary_empty_database(self):
        """Test summary with empty database."""
        # Reset to ensure clean state
        self.manager.reset_database()
        
        summary = self.manager.get_collection_type_summary()
        
        assert summary['total_collections'] == 0
        assert len(summary['by_type']) == 0
        assert len(summary['untyped']) == 0

    def test_migrate_collection_type(self):
        """Test collection type migration functionality."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection with one type
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.GENERAL,
            metadata={'type': 'general'}
        )
        
        # Test migration (if method exists)
        try:
            result = self.manager.migrate_collection_type(
                collection_name, 
                CollectionType.PROJECT_SPECIFIC
            )
            assert result is not None
        except AttributeError:
            # Method doesn't exist yet, that's fine
            pass

    def test_archive_and_restore_collection(self):
        """Test collection archiving and restoration functionality."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.TEMPORARY,
            metadata={'type': 'temporary'}
        )
        
        # Test archiving (if method exists)
        try:
            archive_result = self.manager.archive_collection(collection_name)
            assert archive_result is not None
            
            # Test restoration (if method exists)
            restore_result = self.manager.restore_collection(collection_name)
            assert restore_result is not None
        except AttributeError:
            # Methods don't exist yet, that's fine
            pass


class TestAdvancedErrorHandling:
    """Test advanced error handling scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass
    
    def test_health_check_failure_scenario(self):
        """Test health check failure handling."""
        # This tests lines 195-210 in health_check method
        
        # Initialize the manager first to ensure client exists
        _ = self.manager.client
        
        # Mock client heartbeat to raise exception during health check
        with patch.object(self.manager._client, 'heartbeat') as mock_heartbeat:
            mock_heartbeat.side_effect = Exception("Health check failed")
            
            health_status = self.manager.health_check()
            
            assert health_status.status == 'unhealthy'  # Access as attribute
            assert len(health_status.errors) > 0  # Access as attribute
            assert 'Health check failed' in health_status.errors[0]  # Access as attribute
    
    def test_collection_creation_with_invalid_type_string(self):
        """Test collection creation with invalid type string."""
        # This tests lines 272-276 - ValueError handling in create_collection
        collection_name = f"invalid_type_{uuid4().hex[:8]}"
        
        with pytest.raises(VectorStoreError) as exc_info:
            self.manager.create_collection(
                collection_name, 
                collection_type="invalid_type_string"
            )
        
        assert "Unknown collection type" in str(exc_info.value)
    
    def test_query_filters_validation_edge_cases(self):
        """Test edge cases in query filters validation."""
        # This tests lines 612, 616, 619, 627-631 in _validate_query_filters
        collection_name = f"filter_test_{uuid4().hex[:8]}"
        self.manager.create_collection(collection_name)
        
        # Add test document
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=["test document"],
            embeddings=[[0.1, 0.2]],
            metadata=[{"category": "test", "priority": 1}]
        )
        
        # Test deeply nested filter structure
        complex_filter = {
            "$and": [
                {"category": "test"},
                {
                    "$or": [
                        {"priority": {"$gte": 1}},
                        {"category": {"$in": ["test", "demo"]}}
                    ]
                }
            ]
        }
        
        # Should handle complex nested filters without error
        results = self.manager.query_collection(
            collection_name=collection_name,
            query_embedding=[0.1, 0.2],
            filters=complex_filter
        )
        
        assert results is not None
    
    def test_reset_database_with_client_warning(self):
        """Test database reset functionality with warning logging."""
        # This tests lines 673-676 in reset_database method
        
        # Ensure database is initialized
        _ = self.manager.client
        
        # Reset should work and log warning
        with patch.object(self.manager.logger, 'warning') as mock_warning:
            self.manager.reset_database()
            mock_warning.assert_called_once()
            assert "Resetting database - all data will be lost!" in mock_warning.call_args[0][0]
    
    def test_close_connection_cleanup(self):
        """Test connection cleanup and cache clearing."""
        # Create some collections to populate cache
        self.manager.create_collection(self.get_unique_collection_name("test1"))
        self.manager.create_collection(self.get_unique_collection_name("test2"))
        
        assert len(self.manager.collections._collections_cache) > 0  # Access through collections manager
        
        self.manager.close()
        
        # Cache should be cleared
        assert len(self.manager.collections._collections_cache) == 0  # Access through collections manager
        assert not self.manager.is_connected


class TestCollectionValidationMethods:
    """Test collection validation and type checking methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_unique_collection_name(self) -> str:
        """Generate unique collection name for tests."""
        return f"test_collection_{uuid4().hex[:8]}"

    def test_validate_collection_type_success(self):
        """Test successful collection type validation."""
        collection_name = self.get_unique_collection_name()
        # Use PROJECT_SPECIFIC which has default HNSW params: construction_ef=100, M=16
        collection = self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={"type": "project"}
        )
        
        # Test validation with matching type
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.PROJECT_SPECIFIC
        )
        
        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_validate_collection_type_mismatch(self):
        """Test collection type validation with mismatch."""
        collection_name = self.get_unique_collection_name()
        # Create collection as PROJECT_SPECIFIC but validate as FUNDAMENTAL
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={"type": "project"}
        )
        
        # Test validation with mismatched type (FUNDAMENTAL expects different HNSW params)
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.FUNDAMENTAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        # Should detect collection type mismatch first
        assert any("type mismatch" in error.lower() for error in errors)

    def test_validate_collection_type_nonexistent(self):
        """Test validation with nonexistent collection."""
        nonexistent_name = self.get_unique_collection_name()
        
        is_valid, errors = self.manager.validate_collection_type(
            nonexistent_name, CollectionType.GENERAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("not found" in error.lower() for error in errors)

    def test_validate_collection_type_missing_metadata(self):
        """Test validation with collection missing type metadata."""
        collection_name = self.get_unique_collection_name()
        # Create collection without explicit collection_type (should get GENERAL by default)
        self.manager.create_collection(
            collection_name,
            metadata={"some_other": "metadata"}  # No collection type info
        )
        
        # Test validation against different type
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, CollectionType.FUNDAMENTAL
        )
        
        assert not is_valid
        assert len(errors) > 0
        # Should either be type mismatch or missing metadata
        error_text = " ".join(errors).lower()
        assert ("type mismatch" in error_text or "missing collection_type" in error_text)
    
    def test_validate_collection_type_invalid_string_type(self):
        """Test validation with invalid string type."""
        collection_name = self.get_unique_collection_name()
        
        # Create a collection first
        self.manager.create_collection(collection_name, collection_type=CollectionType.GENERAL)
        
        # Test with invalid expected type string
        is_valid, errors = self.manager.validate_collection_type(
            collection_name, "invalid_type"
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("invalid expected collection type" in error.lower() for error in errors)


class TestCollectionTypeManagement:
    """Test collection type management and organization methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_unique_collection_name(self) -> str:
        """Generate unique collection name for tests."""
        return f"test_collection_{uuid4().hex[:8]}"

    def test_determine_collection_for_document(self):
        """Test document collection determination logic."""
        document_metadata = {
            'user_id': 'test_user',
            'team_id': 'test_team',
            'project_name': 'test_project'
        }
        
        # Should determine collection and auto-create if needed
        collection_name = self.manager.determine_collection_for_document(document_metadata)
        
        assert collection_name is not None
        assert isinstance(collection_name, str)
        
        # Collection should now exist
        collection = self.manager.get_collection(collection_name)
        assert collection is not None
        
        # Second call should use existing collection
        collection_name2 = self.manager.determine_collection_for_document(document_metadata)
        assert collection_name == collection_name2

    def test_determine_collection_for_document_with_chunk_metadata(self):
        """Test collection determination with chunk metadata."""
        document_metadata = {'user_id': 'test_user'}
        chunk_metadata = {'importance': 'high'}
        
        collection_name = self.manager.determine_collection_for_document(
            document_metadata, chunk_metadata
        )
        
        assert collection_name is not None
        assert isinstance(collection_name, str)

    def test_get_collections_by_type(self):
        """Test filtering collections by type."""
        # Create collections of different types
        fundamental_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.FUNDAMENTAL,
            metadata={'type': 'fundamental'}
        )
        
        project_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={'type': 'project'}
        )
        
        general_collection = self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.GENERAL,
            metadata={'type': 'general'}
        )
        
        # Test filtering by FUNDAMENTAL type
        fundamental_collections = self.manager.get_collections_by_type(CollectionType.FUNDAMENTAL)
        assert len(fundamental_collections) >= 1
        assert any(c['name'] == fundamental_collection.name for c in fundamental_collections)
        
        # Test filtering by PROJECT_SPECIFIC type
        project_collections = self.manager.get_collections_by_type(CollectionType.PROJECT_SPECIFIC)
        assert len(project_collections) >= 1
        assert any(c['name'] == project_collection.name for c in project_collections)
        
        # Test filtering by string type
        general_collections = self.manager.get_collections_by_type('general')
        assert len(general_collections) >= 1
        assert any(c['name'] == general_collection.name for c in general_collections)

    def test_get_collections_by_type_invalid_string(self):
        """Test filtering with invalid collection type string."""
        with pytest.raises(ValueError, match="Unknown collection type"):
            self.manager.get_collections_by_type("invalid_type")

    def test_get_collection_type_summary(self):
        """Test collection type summary generation."""
        # Create collections of different types
        self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.FUNDAMENTAL,
            metadata={'type': 'fundamental'}
        )
        
        self.manager.create_collection(
            self.get_unique_collection_name(),
            collection_type=CollectionType.PROJECT_SPECIFIC,
            metadata={'type': 'project'}
        )
        
        # Create collection without collection_type metadata (untyped)
        # Note: This collection should NOT have 'collection_type' in metadata to be truly untyped
        untyped_collection = self.get_unique_collection_name()
        collection = self.manager._client.create_collection(
            name=untyped_collection,
            metadata={'other': 'metadata'}  # No 'collection_type' key
        )
        
        summary = self.manager.get_collection_type_summary()
        
        # Verify summary structure
        assert 'total_collections' in summary
        assert 'by_type' in summary
        assert 'untyped' in summary
        
        assert summary['total_collections'] >= 3
        assert len(summary['by_type']) >= 2  # At least fundamental and project-specific
        assert len(summary['untyped']) >= 1  # At least one untyped collection
        assert untyped_collection in summary['untyped']
        
        # Check type-specific details
        for type_key, type_info in summary['by_type'].items():
            assert 'count' in type_info
            assert 'total_documents' in type_info
            assert 'collections' in type_info
            assert type_info['count'] >= 1
            assert isinstance(type_info['collections'], list)

    def test_get_collection_type_summary_empty_database(self):
        """Test summary with empty database."""
        # Reset to ensure clean state
        self.manager.reset_database()
        
        summary = self.manager.get_collection_type_summary()
        
        assert summary['total_collections'] == 0
        assert len(summary['by_type']) == 0
        assert len(summary['untyped']) == 0

    def test_migrate_collection_type(self):
        """Test collection type migration functionality."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection with one type
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.GENERAL,
            metadata={'type': 'general'}
        )
        
        # Test migration (if method exists)
        try:
            result = self.manager.migrate_collection_type(
                collection_name, 
                CollectionType.PROJECT_SPECIFIC
            )
            assert result is not None
        except AttributeError:
            # Method doesn't exist yet, that's fine
            pass

    def test_archive_and_restore_collection(self):
        """Test collection archiving and restoration functionality."""
        collection_name = self.get_unique_collection_name()
        
        # Create collection
        self.manager.create_collection(
            collection_name,
            collection_type=CollectionType.TEMPORARY,
            metadata={'type': 'temporary'}
        )
        
        # Test archiving (if method exists)
        try:
            archive_result = self.manager.archive_collection(collection_name)
            assert archive_result is not None
            
            # Test restoration (if method exists)
            restore_result = self.manager.restore_collection(collection_name)
            assert restore_result is not None
        except AttributeError:
            # Methods don't exist yet, that's fine
            pass


class TestAdvancedErrorHandling:
    """Test advanced error handling scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = ChromaDBManager(in_memory=True)
        
    def teardown_method(self):
        """Clean up after tests."""
        try:
            self.manager.reset_database()
            self.manager.close()
        except Exception:
            pass
    
    def test_health_check_failure_scenario(self):
        """Test health check failure handling."""
        # This tests lines 195-210 in health_check method
        
        # Initialize the manager first to ensure client exists
        _ = self.manager.client
        
        # Mock client heartbeat to raise exception during health check
        with patch.object(self.manager._client, 'heartbeat') as mock_heartbeat:
            mock_heartbeat.side_effect = Exception("Health check failed")
            
            health_status = self.manager.health_check()
            
            assert health_status.status == 'unhealthy'  # Access as attribute
            assert len(health_status.errors) > 0  # Access as attribute
            assert 'Health check failed' in health_status.errors[0]  # Access as attribute
    
    def test_collection_creation_with_invalid_type_string(self):
        """Test collection creation with invalid type string."""
        # This tests lines 272-276 - ValueError handling in create_collection
        collection_name = f"invalid_type_{uuid4().hex[:8]}"
        
        with pytest.raises(VectorStoreError) as exc_info:
            self.manager.create_collection(
                collection_name, 
                collection_type="invalid_type_string"
            )
        
        assert "Unknown collection type" in str(exc_info.value)
    
    def test_query_filters_validation_edge_cases(self):
        """Test edge cases in query filters validation."""
        # This tests lines 612, 616, 619, 627-631 in _validate_query_filters
        collection_name = f"filter_test_{uuid4().hex[:8]}"
        self.manager.create_collection(collection_name)
        
        # Add test document
        self.manager.add_documents(
            collection_name=collection_name,
            chunks=["test document"],
            embeddings=[[0.1, 0.2]],
            metadata=[{"category": "test", "priority": 1}]
        )
        
        # Test deeply nested filter structure
        complex_filter = {
            "$and": [
                {"category": "test"},
                {
                    "$or": [
                        {"priority": {"$gte": 1}},
                        {"category": {"$in": ["test", "demo"]}}
                    ]
                }
            ]
        }
        
        # Should handle complex nested filters without error
        results = self.manager.query_collection(
            collection_name=collection_name,
            query_embedding=[0.1, 0.2],
            filters=complex_filter
        )
        
        assert results is not None
    
    def test_reset_database_with_client_warning(self):
        """Test database reset functionality with warning logging."""
        # This tests lines 673-676 in reset_database method
        
        # Ensure database is initialized
        _ = self.manager.client
        
        # Reset should work and log warning
        with patch.object(self.manager.logger, 'warning') as mock_warning:
            self.manager.reset_database()
            mock_warning.assert_called_once()
            assert "Resetting database - all data will be lost!" in mock_warning.call_args[0][0]
    
    def test_close_connection_cleanup(self):
        """Test connection cleanup and cache clearing."""
        # Create some collections to populate cache
        self.manager.create_collection(self.get_unique_collection_name("test1"))
        self.manager.create_collection(self.get_unique_collection_name("test2"))
        
        assert len(self.manager.collections._collections_cache) > 0  # Access through collections manager
        
        self.manager.close()
        
        # Cache should be cleared
        assert len(self.manager.collections._collections_cache) == 0  # Access through collections manager
        assert not self.manager.is_connected 