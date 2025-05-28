"""
Tests for metadata schema module.

This module tests all metadata schema functionality including enums, data classes,
validation, serialization, and ChromaDB integration.
"""

import json
import pytest
from datetime import datetime
from uuid import uuid4

from research_agent_backend.models.metadata_schema import (
    DocumentType,
    ContentType,
    CollectionType,
    AccessPermission,
    HeaderHierarchy,
    DocumentMetadata,
    ChunkMetadata,
    CollectionMetadata,
    MetadataValidator,
    create_document_metadata,
    create_chunk_metadata,
    create_collection_metadata,
    DEFAULT_COLLECTION_TYPES,
)


class TestEnums:
    """Test enum functionality."""
    
    def test_document_type_string_conversion(self):
        """Test DocumentType enum string conversion."""
        assert str(DocumentType.MARKDOWN) == "markdown"
        assert str(DocumentType.PDF) == "pdf"
        assert str(DocumentType.UNKNOWN) == "unknown"
    
    def test_content_type_string_conversion(self):
        """Test ContentType enum string conversion."""
        assert str(ContentType.PROSE) == "prose"
        assert str(ContentType.CODE_BLOCK) == "code-block"
        assert str(ContentType.TABLE) == "table"
    
    def test_collection_type_string_conversion(self):
        """Test CollectionType enum string conversion."""
        assert str(CollectionType.FUNDAMENTAL) == "fundamental"
        assert str(CollectionType.PROJECT_SPECIFIC) == "project-specific"
        assert str(CollectionType.GENERAL) == "general"
    
    def test_access_permission_string_conversion(self):
        """Test AccessPermission enum string conversion."""
        assert str(AccessPermission.READ) == "read"
        assert str(AccessPermission.WRITE) == "write"
        assert str(AccessPermission.ADMIN) == "admin"
        assert str(AccessPermission.OWNER) == "owner"


class TestHeaderHierarchy:
    """Test HeaderHierarchy functionality."""
    
    def test_header_hierarchy_creation(self):
        """Test creating header hierarchy."""
        hierarchy = HeaderHierarchy()
        assert hierarchy.levels == []
        assert hierarchy.depths == []
        assert hierarchy.get_path() == ""
    
    def test_add_header(self):
        """Test adding headers to hierarchy."""
        hierarchy = HeaderHierarchy()
        hierarchy.add_header("Introduction", 1)
        hierarchy.add_header("Overview", 2)
        
        assert hierarchy.levels == ["Introduction", "Overview"]
        assert hierarchy.depths == [1, 2]
        assert hierarchy.get_path() == "Introduction > Overview"
    
    def test_get_context_at_depth(self):
        """Test getting context at specific depth."""
        hierarchy = HeaderHierarchy()
        hierarchy.add_header("Main", 1)
        hierarchy.add_header("Sub", 2)
        hierarchy.add_header("Detail", 3)
        
        context = hierarchy.get_context_at_depth(2)
        assert context == ["Main", "Sub"]
    
    def test_to_dict_conversion(self):
        """Test converting hierarchy to dictionary."""
        hierarchy = HeaderHierarchy()
        hierarchy.add_header("Chapter 1", 1)
        hierarchy.add_header("Section A", 2)
        
        result = hierarchy.to_dict()
        expected = {
            "levels": ["Chapter 1", "Section A"],
            "depths": [1, 2],
            "path": "Chapter 1 > Section A"
        }
        assert result == expected
    
    def test_from_dict_creation(self):
        """Test creating hierarchy from dictionary."""
        data = {
            "levels": ["Main", "Sub"],
            "depths": [1, 2]
        }
        
        hierarchy = HeaderHierarchy.from_dict(data)
        assert hierarchy.levels == ["Main", "Sub"]
        assert hierarchy.depths == [1, 2]
        assert hierarchy.get_path() == "Main > Sub"
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        hierarchy = HeaderHierarchy()
        hierarchy.add_header("Test", 1)
        
        json_str = hierarchy.to_json()
        restored = HeaderHierarchy.from_json(json_str)
        
        assert restored.levels == hierarchy.levels
        assert restored.depths == hierarchy.depths
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        hierarchy = HeaderHierarchy.from_json("invalid json")
        assert hierarchy.levels == []
        assert hierarchy.depths == []


class TestDocumentMetadata:
    """Test DocumentMetadata functionality."""
    
    def test_document_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            title="Test Document",
            document_type=DocumentType.MARKDOWN,
            source_path="/path/to/doc.md"
        )
        
        assert metadata.title == "Test Document"
        assert metadata.document_type == DocumentType.MARKDOWN
        assert metadata.source_path == "/path/to/doc.md"
        assert isinstance(metadata.document_id, str)
        assert isinstance(metadata.created_at, datetime)
    
    def test_update_timestamp(self):
        """Test timestamp update functionality."""
        metadata = DocumentMetadata()
        original_time = metadata.updated_at
        
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.001)
        
        metadata.update_timestamp()
        assert metadata.updated_at > original_time
    
    def test_to_dict_conversion(self):
        """Test converting document metadata to dictionary."""
        metadata = DocumentMetadata(
            title="Test",
            document_type=DocumentType.PDF,
            tags=["tag1", "tag2"]
        )
        
        result = metadata.to_dict()
        
        assert result["title"] == "Test"
        assert result["document_type"] == "pdf"
        assert result["tags"] == ["tag1", "tag2"]
        assert "document_id" in result
        assert "created_at" in result


class TestChunkMetadata:
    """Test ChunkMetadata functionality."""
    
    def test_chunk_metadata_creation(self):
        """Test creating chunk metadata."""
        metadata = ChunkMetadata(
            source_document_id="doc123",
            document_title="Test Doc",
            chunk_sequence_id=1,
            content_type=ContentType.PROSE
        )
        
        assert metadata.source_document_id == "doc123"
        assert metadata.document_title == "Test Doc"
        assert metadata.chunk_sequence_id == 1
        assert metadata.content_type == ContentType.PROSE
        assert isinstance(metadata.chunk_id, str)
    
    def test_chromadb_metadata_conversion(self):
        """Test converting to ChromaDB-compatible metadata."""
        hierarchy = HeaderHierarchy()
        hierarchy.add_header("Test Header", 1)
        
        metadata = ChunkMetadata(
            source_document_id="doc123",
            document_title="Test Document",
            chunk_sequence_id=5,
            content_type=ContentType.CODE_BLOCK,
            code_language="python",
            header_hierarchy=hierarchy,
            access_permissions=[AccessPermission.READ, AccessPermission.WRITE]
        )
        
        chromadb_meta = metadata.to_chromadb_metadata()
        
        # Check all fields are present and correct types
        assert chromadb_meta["source_document_id"] == "doc123"
        assert chromadb_meta["document_title"] == "Test Document"
        assert chromadb_meta["chunk_sequence_id"] == 5
        assert chromadb_meta["content_type"] == "code-block"
        assert chromadb_meta["code_language"] == "python"
        assert chromadb_meta["access_permissions"] == "read,write"
        
        # Check header hierarchy is JSON string
        assert isinstance(chromadb_meta["header_hierarchy"], str)
        hierarchy_data = json.loads(chromadb_meta["header_hierarchy"])
        assert hierarchy_data["levels"] == ["Test Header"]
        
        # Verify all values are ChromaDB-compatible types
        for key, value in chromadb_meta.items():
            assert isinstance(value, (str, int, float, bool)), f"Field {key} has invalid type {type(value)}"
    
    def test_from_chromadb_metadata(self):
        """Test creating ChunkMetadata from ChromaDB metadata."""
        chromadb_meta = {
            "chunk_id": "chunk123",
            "source_document_id": "doc456",
            "document_title": "Source Doc",
            "chunk_sequence_id": 3,
            "content_type": "table",
            "code_language": "javascript",
            "header_hierarchy": '{"levels": ["Chapter 1"], "depths": [1], "path": "Chapter 1"}',
            "chunk_size": 512,
            "user_id": "user123",
            "access_permissions": "read,admin",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-02T00:00:00"
        }
        
        metadata = ChunkMetadata.from_chromadb_metadata(chromadb_meta)
        
        assert metadata.chunk_id == "chunk123"
        assert metadata.source_document_id == "doc456"
        assert metadata.document_title == "Source Doc"
        assert metadata.chunk_sequence_id == 3
        assert metadata.content_type == ContentType.TABLE
        assert metadata.code_language == "javascript"
        assert metadata.chunk_size == 512
        assert metadata.user_id == "user123"
        assert AccessPermission.READ in metadata.access_permissions
        assert AccessPermission.ADMIN in metadata.access_permissions
        
        # Test header hierarchy restoration
        assert metadata.header_hierarchy.levels == ["Chapter 1"]
        assert metadata.header_hierarchy.depths == [1]
    
    def test_round_trip_chromadb_conversion(self):
        """Test round-trip conversion to/from ChromaDB metadata."""
        original = ChunkMetadata(
            source_document_id="doc789",
            document_title="Round Trip Test",
            chunk_sequence_id=10,
            content_type=ContentType.LIST,
            user_id="user456"
        )
        
        # Convert to ChromaDB format and back
        chromadb_meta = original.to_chromadb_metadata()
        restored = ChunkMetadata.from_chromadb_metadata(chromadb_meta)
        
        assert restored.source_document_id == original.source_document_id
        assert restored.document_title == original.document_title
        assert restored.chunk_sequence_id == original.chunk_sequence_id
        assert restored.content_type == original.content_type
        assert restored.user_id == original.user_id
    
    def test_to_dict_conversion(self):
        """Test converting chunk metadata to dictionary."""
        metadata = ChunkMetadata(
            source_document_id="doc123",
            document_title="Test",
            content_type=ContentType.PROSE
        )
        
        result = metadata.to_dict()
        
        assert result["source_document_id"] == "doc123"
        assert result["document_title"] == "Test"
        assert result["content_type"] == "prose"
        assert "chunk_id" in result
        assert "header_hierarchy" in result


class TestCollectionMetadata:
    """Test CollectionMetadata functionality."""
    
    def test_collection_metadata_creation(self):
        """Test creating collection metadata."""
        metadata = CollectionMetadata(
            collection_name="test_collection",
            collection_type=CollectionType.FUNDAMENTAL,
            description="Test collection"
        )
        
        assert metadata.collection_name == "test_collection"
        assert metadata.collection_type == CollectionType.FUNDAMENTAL
        assert metadata.description == "Test collection"
        assert metadata.hnsw_construction_ef == 100
        assert metadata.hnsw_m == 16
    
    def test_update_stats(self):
        """Test updating collection statistics."""
        metadata = CollectionMetadata()
        original_time = metadata.updated_at
        
        import time
        time.sleep(0.001)
        
        metadata.update_stats(document_count=10, chunk_count=50, size_bytes=1024)
        
        assert metadata.document_count == 10
        assert metadata.chunk_count == 50
        assert metadata.total_size_bytes == 1024
        assert metadata.updated_at > original_time
    
    def test_add_team_permission(self):
        """Test adding team permissions."""
        metadata = CollectionMetadata()
        permissions = [AccessPermission.READ, AccessPermission.WRITE]
        
        metadata.add_team_permission("user123", permissions)
        
        assert "user123" in metadata.team_permissions
        assert metadata.team_permissions["user123"] == permissions
    
    def test_to_dict_conversion(self):
        """Test converting collection metadata to dictionary."""
        metadata = CollectionMetadata(
            collection_name="test",
            collection_type=CollectionType.PROJECT_SPECIFIC
        )
        metadata.add_team_permission("user1", [AccessPermission.ADMIN])
        
        result = metadata.to_dict()
        
        assert result["collection_name"] == "test"
        assert result["collection_type"] == "project-specific"
        assert result["team_permissions"]["user1"] == ["admin"]


class TestMetadataValidator:
    """Test MetadataValidator functionality."""
    
    def test_validate_string_field(self):
        """Test string field validation."""
        # Normal string
        result = MetadataValidator.validate_string_field("test", "field1")
        assert result == "test"
        
        # None value
        result = MetadataValidator.validate_string_field(None, "field2")
        assert result == ""
        
        # Non-string value
        result = MetadataValidator.validate_string_field(123, "field3")
        assert result == "123"
        
        # Whitespace trimming
        result = MetadataValidator.validate_string_field("  test  ", "field4")
        assert result == "test"
        
        # Length limit
        long_string = "a" * 2000
        result = MetadataValidator.validate_string_field(long_string, "field5", max_length=100)
        assert len(result) == 100
    
    def test_validate_integer_field(self):
        """Test integer field validation."""
        # Normal integer
        result = MetadataValidator.validate_integer_field(42, "field1")
        assert result == 42
        
        # None value
        result = MetadataValidator.validate_integer_field(None, "field2")
        assert result == 0
        
        # String integer
        result = MetadataValidator.validate_integer_field("123", "field3")
        assert result == 123
        
        # Invalid value
        result = MetadataValidator.validate_integer_field("invalid", "field4")
        assert result == 0
        
        # Bounds checking
        result = MetadataValidator.validate_integer_field(-5, "field5", min_value=0)
        assert result == 0
        
        result = MetadataValidator.validate_integer_field(200, "field6", max_value=100)
        assert result == 100
    
    def test_validate_enum_field(self):
        """Test enum field validation."""
        # Valid enum value
        result = MetadataValidator.validate_enum_field(ContentType.PROSE, ContentType, "field1")
        assert result == ContentType.PROSE
        
        # Valid string value
        result = MetadataValidator.validate_enum_field("prose", ContentType, "field2")
        assert result == ContentType.PROSE
        
        # Case insensitive match
        result = MetadataValidator.validate_enum_field("PROSE", ContentType, "field3")
        assert result == ContentType.PROSE
        
        # Invalid value with default
        result = MetadataValidator.validate_enum_field("invalid", ContentType, "field4", ContentType.UNKNOWN)
        assert result == ContentType.UNKNOWN
        
        # None value
        result = MetadataValidator.validate_enum_field(None, ContentType, "field5")
        assert result in ContentType
    
    def test_validate_chunk_metadata(self):
        """Test chunk metadata validation."""
        metadata = ChunkMetadata(
            chunk_id="  test_id  ",
            source_document_id="doc123",
            document_title="Test Document",
            chunk_sequence_id="5",  # String that should be converted to int
            content_type="prose",   # String that should be converted to enum
            code_language="  python  "
        )
        
        validated = MetadataValidator.validate_chunk_metadata(metadata)
        
        assert validated.chunk_id == "test_id"  # Trimmed
        assert validated.chunk_sequence_id == 5  # Converted to int
        assert validated.content_type == ContentType.PROSE  # Converted to enum
        assert validated.code_language == "python"  # Trimmed


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_document_metadata(self):
        """Test document metadata factory function."""
        metadata = create_document_metadata(
            title="Test Doc",
            source_path="/path/to/doc.md",
            document_type="markdown",
            user_id="user123"
        )
        
        assert metadata.title == "Test Doc"
        assert metadata.source_path == "/path/to/doc.md"
        assert metadata.document_type == DocumentType.MARKDOWN
        assert metadata.user_id == "user123"
    
    def test_create_chunk_metadata(self):
        """Test chunk metadata factory function."""
        metadata = create_chunk_metadata(
            source_document_id="doc123",
            document_title="Test Document",
            chunk_sequence_id=1,
            content_type="code-block",
            user_id="user456"
        )
        
        assert metadata.source_document_id == "doc123"
        assert metadata.document_title == "Test Document"
        assert metadata.chunk_sequence_id == 1
        assert metadata.content_type == ContentType.CODE_BLOCK
        assert metadata.user_id == "user456"
    
    def test_create_collection_metadata(self):
        """Test collection metadata factory function."""
        metadata = create_collection_metadata(
            collection_name="test_collection",
            collection_type="fundamental",
            owner_id="owner123"
        )
        
        assert metadata.collection_name == "test_collection"
        assert metadata.collection_type == CollectionType.FUNDAMENTAL
        assert metadata.owner_id == "owner123"


class TestDefaultCollectionTypes:
    """Test default collection type definitions."""
    
    def test_default_collection_types_structure(self):
        """Test that default collection types are properly defined."""
        assert "fundamental" in DEFAULT_COLLECTION_TYPES
        assert "project-specific" in DEFAULT_COLLECTION_TYPES
        assert "general" in DEFAULT_COLLECTION_TYPES
        assert "reference" in DEFAULT_COLLECTION_TYPES
        
        # Check structure of each type
        for name, config in DEFAULT_COLLECTION_TYPES.items():
            assert "type" in config
            assert "description" in config
            assert isinstance(config["type"], CollectionType)
            assert isinstance(config["description"], str)
    
    def test_collection_type_consistency(self):
        """Test that collection type values match enum values."""
        fundamental_config = DEFAULT_COLLECTION_TYPES["fundamental"]
        assert fundamental_config["type"] == CollectionType.FUNDAMENTAL
        
        project_config = DEFAULT_COLLECTION_TYPES["project-specific"]
        assert project_config["type"] == CollectionType.PROJECT_SPECIFIC
        
        general_config = DEFAULT_COLLECTION_TYPES["general"]
        assert general_config["type"] == CollectionType.GENERAL
        
        reference_config = DEFAULT_COLLECTION_TYPES["reference"]
        assert reference_config["type"] == CollectionType.REFERENCE


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_header_hierarchy_json(self):
        """Test handling empty header hierarchy JSON."""
        metadata = ChunkMetadata()
        chromadb_meta = metadata.to_chromadb_metadata()
        
        # Header hierarchy should be empty JSON object
        hierarchy_json = chromadb_meta["header_hierarchy"]
        hierarchy = HeaderHierarchy.from_json(hierarchy_json)
        assert hierarchy.levels == []
        assert hierarchy.depths == []
    
    def test_malformed_access_permissions(self):
        """Test handling malformed access permissions in ChromaDB metadata."""
        chromadb_meta = {
            "access_permissions": "read,invalid_permission,write"
        }
        
        metadata = ChunkMetadata.from_chromadb_metadata(chromadb_meta)
        
        # Should only include valid permissions
        valid_permissions = [perm for perm in metadata.access_permissions]
        assert AccessPermission.READ in valid_permissions
        assert AccessPermission.WRITE in valid_permissions
        assert len(valid_permissions) == 2  # Invalid permission should be filtered out
    
    def test_invalid_datetime_strings(self):
        """Test handling invalid datetime strings in ChromaDB metadata."""
        chromadb_meta = {
            "created_at": "invalid_datetime",
            "updated_at": "2023-13-45T25:75:99"  # Invalid date
        }
        
        metadata = ChunkMetadata.from_chromadb_metadata(chromadb_meta)
        
        # Should use current time for invalid timestamps
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime) 