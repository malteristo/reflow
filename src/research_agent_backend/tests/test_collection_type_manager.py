"""
Tests for CollectionTypeManager module.

This module tests collection type-specific configuration management,
HNSW parameters, and data routing logic for different collection types.

Implements TDD principles for collection type management functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from ..core.collection_type_manager import (
    CollectionTypeManager,
    CollectionTypeConfig,
    create_collection_type_manager
)
from ..models.metadata_schema import (
    CollectionType,
    CollectionMetadata,
    AccessPermission
)
from ..utils.config import ConfigManager


class TestCollectionTypeConfig:
    """Test CollectionTypeConfig dataclass functionality."""
    
    def test_collection_type_config_creation(self):
        """Test creating a CollectionTypeConfig instance."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.FUNDAMENTAL,
            description="Test fundamental collection",
            default_name_prefix="test_fundamental",
            embedding_dimension=384,
            distance_metric="cosine"
        )
        
        assert config.collection_type == CollectionType.FUNDAMENTAL
        assert config.description == "Test fundamental collection"
        assert config.default_name_prefix == "test_fundamental"
        assert config.embedding_dimension == 384
        assert config.distance_metric == "cosine"
        assert config.hnsw_construction_ef == 100  # Default value
        assert config.hnsw_m == 16  # Default value
    
    def test_to_chromadb_metadata(self):
        """Test conversion to ChromaDB metadata format."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.REFERENCE,
            description="Reference collection",
            default_name_prefix="reference",
            hnsw_construction_ef=200,
            hnsw_m=32,
            distance_metric="cosine"
        )
        
        metadata = config.to_chromadb_metadata()
        
        expected_metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:M": 32,
            "hnsw:search_ef": 50,  # Default value
            "collection_type": "reference",
            "batch_insert_size": 100,  # Default value
            "enable_auto_compaction": True  # Default value
        }
        
        assert metadata == expected_metadata
    
    def test_create_collection_metadata(self):
        """Test creating CollectionMetadata from config."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.PROJECT_SPECIFIC,
            description="Project collection",
            default_name_prefix="project",
            embedding_dimension=512,
            distance_metric="ip",
            hnsw_construction_ef=150,
            hnsw_m=24
        )
        
        collection_metadata = config.create_collection_metadata(
            collection_name="test_project_collection",
            owner_id="user123",
            team_id="team456"
        )
        
        assert isinstance(collection_metadata, CollectionMetadata)
        assert collection_metadata.collection_name == "test_project_collection"
        assert collection_metadata.collection_type == CollectionType.PROJECT_SPECIFIC
        assert collection_metadata.description == "Project collection"
        assert collection_metadata.embedding_dimension == 512
        assert collection_metadata.distance_metric == "ip"
        assert collection_metadata.hnsw_construction_ef == 150
        assert collection_metadata.hnsw_m == 24
        assert collection_metadata.owner_id == "user123"
        assert collection_metadata.team_id == "team456"


class TestCollectionTypeManager:
    """Test CollectionTypeManager functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager for testing."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get.return_value = {}  # Return empty dict for collection_types
        return mock_config
    
    @pytest.fixture
    def collection_type_manager(self, mock_config_manager):
        """Create CollectionTypeManager instance for testing."""
        return CollectionTypeManager(mock_config_manager)
    
    def test_initialization(self, collection_type_manager):
        """Test CollectionTypeManager initialization."""
        assert collection_type_manager is not None
        assert isinstance(collection_type_manager.config_manager, Mock)
        
        # Check that all default collection types are initialized
        all_types = collection_type_manager.get_all_collection_types()
        expected_types = [
            CollectionType.FUNDAMENTAL,
            CollectionType.PROJECT_SPECIFIC,
            CollectionType.GENERAL,
            CollectionType.REFERENCE,
            CollectionType.TEMPORARY
        ]
        
        assert len(all_types) == len(expected_types)
        for expected_type in expected_types:
            assert expected_type in all_types
    
    def test_get_collection_config_with_enum(self, collection_type_manager):
        """Test getting collection config with enum parameter."""
        config = collection_type_manager.get_collection_config(CollectionType.FUNDAMENTAL)
        
        assert config.collection_type == CollectionType.FUNDAMENTAL
        assert config.description == "Core foundational knowledge for long-term reference"
        assert config.default_name_prefix == "fundamental"
        assert config.hnsw_construction_ef == 200  # Optimized for read-heavy
        assert config.hnsw_m == 32  # Higher for better recall
        assert config.allow_public_access is True
    
    def test_get_collection_config_with_string(self, collection_type_manager):
        """Test getting collection config with string parameter."""
        config = collection_type_manager.get_collection_config("project-specific")
        
        assert config.collection_type == CollectionType.PROJECT_SPECIFIC
        assert config.description == "Project-specific knowledge and documentation"
        assert config.default_name_prefix == "project"
        assert config.hnsw_construction_ef == 100  # Balanced read/write
        assert config.hnsw_m == 16  # Balanced performance
        assert config.allow_public_access is False
    
    def test_get_collection_config_unknown_type(self, collection_type_manager):
        """Test getting config for unknown collection type."""
        with pytest.raises(ValueError, match="Unknown collection type: unknown"):
            collection_type_manager.get_collection_config("unknown")
    
    def test_determine_collection_type_reference(self, collection_type_manager):
        """Test determining collection type for reference materials."""
        document_metadata = {
            'document_type': 'reference',
            'source_path': '/docs/reference/api.md',
            'user_id': 'user123'
        }
        
        collection_type = collection_type_manager.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.REFERENCE
    
    def test_determine_collection_type_fundamental(self, collection_type_manager):
        """Test determining collection type for fundamental knowledge."""
        document_metadata = {
            'document_type': 'pdf',
            'source_path': '/fundamental/standards/iso.pdf',
            'user_id': 'user123'
        }
        
        collection_type = collection_type_manager.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.FUNDAMENTAL
    
    def test_determine_collection_type_project_specific(self, collection_type_manager):
        """Test determining collection type for project-specific content."""
        document_metadata = {
            'document_type': 'markdown',
            'source_path': '/project/myapp/docs/readme.md',
            'user_id': 'user123',
            'team_id': 'team456'
        }
        
        collection_type = collection_type_manager.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.PROJECT_SPECIFIC
    
    def test_determine_collection_type_temporary(self, collection_type_manager):
        """Test determining collection type for temporary content."""
        document_metadata = {
            'document_type': 'text',
            'source_path': '/temp/scratch_notes.txt',
            'user_id': 'user123'
        }
        
        collection_type = collection_type_manager.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.TEMPORARY
    
    def test_determine_collection_type_general_default(self, collection_type_manager):
        """Test determining collection type defaults to general."""
        document_metadata = {
            'document_type': 'markdown',
            'source_path': '/random/document.md',
            'user_id': 'user123'
        }
        
        collection_type = collection_type_manager.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.GENERAL
    
    def test_create_collection_name_basic(self, collection_type_manager):
        """Test creating basic collection name."""
        name = collection_type_manager.create_collection_name(CollectionType.GENERAL)
        assert name == "general"
    
    def test_create_collection_name_project_specific(self, collection_type_manager):
        """Test creating project-specific collection name."""
        name = collection_type_manager.create_collection_name(
            CollectionType.PROJECT_SPECIFIC,
            project_name="My App Project"
        )
        assert name == "project_my_app_project"
    
    def test_create_collection_name_with_suffix(self, collection_type_manager):
        """Test creating collection name with suffix."""
        name = collection_type_manager.create_collection_name(
            CollectionType.REFERENCE,
            suffix="v2"
        )
        assert name == "reference_v2"
    
    def test_create_collection_name_sanitization(self, collection_type_manager):
        """Test collection name sanitization."""
        name = collection_type_manager.create_collection_name(
            CollectionType.PROJECT_SPECIFIC,
            project_name="My-Project@#$%",
            suffix="test!@#"
        )
        assert name == "project_my_project_test"
    
    def test_validate_collection_for_type_valid(self, collection_type_manager):
        """Test validating a collection that matches its type."""
        collection_metadata = CollectionMetadata(
            collection_name="test_fundamental",
            collection_type=CollectionType.FUNDAMENTAL,
            embedding_dimension=384,
            distance_metric="cosine",
            hnsw_construction_ef=200,
            hnsw_m=32
        )
        
        is_valid, errors = collection_type_manager.validate_collection_for_type(
            collection_metadata,
            CollectionType.FUNDAMENTAL
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_collection_for_type_invalid(self, collection_type_manager):
        """Test validating a collection that doesn't match its type."""
        collection_metadata = CollectionMetadata(
            collection_name="test_collection",
            collection_type=CollectionType.GENERAL,  # Wrong type
            embedding_dimension=512,  # Wrong dimension
            distance_metric="ip",  # Wrong metric
            hnsw_construction_ef=50,  # Wrong HNSW setting
            hnsw_m=8  # Wrong HNSW setting
        )
        
        is_valid, errors = collection_type_manager.validate_collection_for_type(
            collection_metadata,
            CollectionType.FUNDAMENTAL
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("Collection type mismatch" in error for error in errors)
        assert any("Embedding dimension mismatch" in error for error in errors)
        assert any("Distance metric mismatch" in error for error in errors)
    
    def test_get_collection_summary(self, collection_type_manager):
        """Test getting collection type summary."""
        summary = collection_type_manager.get_collection_summary()
        
        assert 'total_types' in summary
        assert 'types' in summary
        assert summary['total_types'] == 5  # Five default types
        
        # Check that all default types are present
        type_keys = summary['types'].keys()
        expected_types = ['fundamental', 'project-specific', 'general', 'reference', 'temporary']
        
        for expected_type in expected_types:
            assert expected_type in type_keys
        
        # Check fundamental type details
        fundamental_info = summary['types']['fundamental']
        assert fundamental_info['description'] == "Core foundational knowledge for long-term reference"
        assert fundamental_info['embedding_dimension'] == 384
        assert fundamental_info['distance_metric'] == "cosine"
        assert fundamental_info['hnsw_construction_ef'] == 200
        assert fundamental_info['allow_public_access'] is True
    
    def test_load_custom_configurations(self, mock_config_manager):
        """Test loading custom configurations from config."""
        # Setup mock to return custom config
        custom_config = {
            'general': {
                'hnsw_construction_ef': 75,
                'batch_insert_size': 150
            }
        }
        mock_config_manager.get.return_value = custom_config
        
        manager = CollectionTypeManager(mock_config_manager)
        
        # Verify custom configuration was applied
        general_config = manager.get_collection_config(CollectionType.GENERAL)
        assert general_config.hnsw_construction_ef == 75
        assert general_config.batch_insert_size == 150


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_create_collection_type_manager_default(self):
        """Test creating manager with default config."""
        with patch('src.research_agent_backend.core.collection_type_manager.ConfigManager') as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get.return_value = {}
            
            manager = create_collection_type_manager()
            
            assert isinstance(manager, CollectionTypeManager)
            assert manager.config_manager == mock_config
    
    def test_create_collection_type_manager_with_config(self):
        """Test creating manager with provided config."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get.return_value = {}
        
        manager = create_collection_type_manager(mock_config)
        
        assert isinstance(manager, CollectionTypeManager)
        assert manager.config_manager == mock_config


class TestCollectionTypeIntegration:
    """Integration tests for collection type functionality."""
    
    @pytest.fixture
    def manager_with_config(self):
        """Create manager with realistic configuration."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get.return_value = {
            'reference': {
                'hnsw_construction_ef': 250,
                'max_documents_per_collection': 1500
            }
        }
        return CollectionTypeManager(config_manager)
    
    def test_end_to_end_collection_workflow(self, manager_with_config):
        """Test complete workflow from document to collection configuration."""
        # Document metadata simulating a reference document
        document_metadata = {
            'document_type': 'pdf',
            'source_path': '/docs/api_reference.pdf',
            'user_id': 'user123',
            'team_id': 'public'
        }
        
        # Determine collection type
        collection_type = manager_with_config.determine_collection_type(document_metadata)
        assert collection_type == CollectionType.REFERENCE
        
        # Get configuration for the determined type
        config = manager_with_config.get_collection_config(collection_type)
        assert config.collection_type == CollectionType.REFERENCE
        assert config.hnsw_construction_ef == 250  # Custom config applied
        assert config.max_documents_per_collection == 1500  # Custom config applied
        
        # Create collection name
        collection_name = manager_with_config.create_collection_name(collection_type)
        assert collection_name == "reference"
        
        # Create collection metadata
        collection_metadata = config.create_collection_metadata(
            collection_name=collection_name,
            owner_id="user123"
        )
        
        # Validate the metadata
        is_valid, errors = manager_with_config.validate_collection_for_type(
            collection_metadata,
            collection_type
        )
        assert is_valid is True
        assert len(errors) == 0 