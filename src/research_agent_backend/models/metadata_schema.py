"""
Metadata Schema for Research Agent Vector Database.

This module defines comprehensive metadata schemas for documents, chunks, and collections
supporting ChromaDB integration and future team scalability features.

Implements FR-KB-002: Rich metadata extraction and storage.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification for metadata."""
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    CODE = "code"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class ContentType(Enum):
    """Content type classification for chunks."""
    PROSE = "prose"
    CODE_BLOCK = "code-block"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    METADATA_BLOCK = "metadata-block"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class CollectionType(Enum):
    """Collection type classification."""
    FUNDAMENTAL = "fundamental"
    PROJECT_SPECIFIC = "project-specific"
    GENERAL = "general"
    REFERENCE = "reference"
    TEMPORARY = "temporary"
    
    def __str__(self) -> str:
        return self.value


class AccessPermission(Enum):
    """Access permission levels for future team features."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class HeaderHierarchy:
    """
    Represents hierarchical header structure for Markdown documents.
    
    Implements FR-KB-002.1: Header hierarchy tracking for context preservation.
    """
    levels: List[str] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    
    def add_header(self, text: str, depth: int) -> None:
        """Add a header to the hierarchy."""
        self.levels.append(text.strip())
        self.depths.append(depth)
    
    def get_path(self) -> str:
        """Get formatted header path for display."""
        return " > ".join(self.levels) if self.levels else ""
    
    def get_context_at_depth(self, max_depth: int) -> List[str]:
        """Get header context up to specified depth."""
        return [level for level, depth in zip(self.levels, self.depths) if depth <= max_depth]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "levels": self.levels,
            "depths": self.depths,
            "path": self.get_path()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeaderHierarchy':
        """Create from dictionary."""
        hierarchy = cls()
        hierarchy.levels = data.get("levels", [])
        hierarchy.depths = data.get("depths", [])
        return hierarchy
    
    def to_json(self) -> str:
        """Convert to JSON string for ChromaDB storage."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HeaderHierarchy':
        """Create from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse header hierarchy JSON: {e}")
            return cls()


@dataclass
class DocumentMetadata:
    """
    Metadata for source documents.
    
    Supports future team scalability and comprehensive document tracking.
    """
    document_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    source_path: str = ""
    file_size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Team scalability fields
    user_id: str = ""
    team_id: Optional[str] = None
    access_permissions: List[AccessPermission] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_timestamp(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": str(self.document_type),
            "source_path": self.source_path,
            "file_size_bytes": self.file_size_bytes,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "access_permissions": [str(perm) for perm in self.access_permissions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ChunkMetadata:
    """
    Metadata for document chunks stored in vector database.
    
    Implements FR-KB-002: Rich metadata for document chunks including
    source_document_id, document_title, header_hierarchy, chunk_sequence_id,
    content_type, and code_language.
    """
    # Core chunk identification
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    source_document_id: str = ""
    document_title: str = ""
    chunk_sequence_id: int = 0
    
    # Content classification
    content_type: ContentType = ContentType.PROSE
    code_language: Optional[str] = None
    
    # Markdown structure context
    header_hierarchy: HeaderHierarchy = field(default_factory=HeaderHierarchy)
    
    # Chunk properties
    chunk_size: int = 0
    start_char_index: Optional[int] = None
    end_char_index: Optional[int] = None
    
    # Team scalability fields
    user_id: str = ""
    team_id: Optional[str] = None
    access_permissions: List[AccessPermission] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_timestamp(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_chromadb_metadata(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Convert to ChromaDB-compatible metadata.
        
        ChromaDB only accepts str, int, float, or bool values for metadata.
        Complex objects are JSON serialized to strings.
        """
        metadata = {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "document_title": self.document_title,
            "chunk_sequence_id": self.chunk_sequence_id,
            "content_type": str(self.content_type),
            "code_language": self.code_language or "",
            "header_hierarchy": self.header_hierarchy.to_json(),
            "chunk_size": self.chunk_size,
            "start_char_index": self.start_char_index or 0,
            "end_char_index": self.end_char_index or 0,
            "user_id": self.user_id,
            "team_id": self.team_id or "",
            "access_permissions": ",".join(str(perm) for perm in self.access_permissions),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        # Ensure all values are ChromaDB-compatible types
        validated_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                validated_metadata[key] = value
            else:
                validated_metadata[key] = str(value)
        
        return validated_metadata
    
    @classmethod
    def from_chromadb_metadata(cls, metadata: Dict[str, Any]) -> 'ChunkMetadata':
        """Create ChunkMetadata from ChromaDB metadata."""
        chunk_metadata = cls()
        
        # Basic fields
        chunk_metadata.chunk_id = metadata.get("chunk_id", "")
        chunk_metadata.source_document_id = metadata.get("source_document_id", "")
        chunk_metadata.document_title = metadata.get("document_title", "")
        chunk_metadata.chunk_sequence_id = int(metadata.get("chunk_sequence_id", 0))
        
        # Content type
        content_type_str = metadata.get("content_type", "prose")
        try:
            chunk_metadata.content_type = ContentType(content_type_str)
        except ValueError:
            chunk_metadata.content_type = ContentType.UNKNOWN
        
        chunk_metadata.code_language = metadata.get("code_language") or None
        
        # Header hierarchy
        hierarchy_json = metadata.get("header_hierarchy", "")
        if hierarchy_json:
            chunk_metadata.header_hierarchy = HeaderHierarchy.from_json(hierarchy_json)
        
        # Chunk properties
        chunk_metadata.chunk_size = int(metadata.get("chunk_size", 0))
        chunk_metadata.start_char_index = int(metadata.get("start_char_index", 0)) or None
        chunk_metadata.end_char_index = int(metadata.get("end_char_index", 0)) or None
        
        # Team fields
        chunk_metadata.user_id = metadata.get("user_id", "")
        chunk_metadata.team_id = metadata.get("team_id") or None
        
        # Access permissions
        permissions_str = metadata.get("access_permissions", "")
        if permissions_str:
            permission_strs = permissions_str.split(",")
            permissions = []
            for perm_str in permission_strs:
                try:
                    permissions.append(AccessPermission(perm_str.strip()))
                except ValueError:
                    continue
            chunk_metadata.access_permissions = permissions
        
        # Timestamps
        try:
            chunk_metadata.created_at = datetime.fromisoformat(metadata.get("created_at", ""))
        except (ValueError, TypeError):
            chunk_metadata.created_at = datetime.utcnow()
        
        try:
            chunk_metadata.updated_at = datetime.fromisoformat(metadata.get("updated_at", ""))
        except (ValueError, TypeError):
            chunk_metadata.updated_at = datetime.utcnow()
        
        return chunk_metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "document_title": self.document_title,
            "chunk_sequence_id": self.chunk_sequence_id,
            "content_type": str(self.content_type),
            "code_language": self.code_language,
            "header_hierarchy": self.header_hierarchy.to_dict(),
            "chunk_size": self.chunk_size,
            "start_char_index": self.start_char_index,
            "end_char_index": self.end_char_index,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "access_permissions": [str(perm) for perm in self.access_permissions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class CollectionMetadata:
    """
    Metadata for vector database collections.
    
    Supports collection management and team-based access control.
    """
    collection_name: str = ""
    collection_type: CollectionType = CollectionType.GENERAL
    description: str = ""
    
    # Vector database configuration
    embedding_model: str = ""
    embedding_dimension: int = 0
    distance_metric: str = "cosine"
    
    # HNSW parameters
    hnsw_construction_ef: int = 100
    hnsw_m: int = 16
    
    # Collection statistics
    document_count: int = 0
    chunk_count: int = 0
    total_size_bytes: int = 0
    
    # Team scalability
    owner_id: str = ""
    team_id: Optional[str] = None
    team_permissions: Dict[str, List[AccessPermission]] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_indexed_at: Optional[datetime] = None
    
    def update_stats(self, document_count: int = None, chunk_count: int = None, size_bytes: int = None) -> None:
        """Update collection statistics."""
        if document_count is not None:
            self.document_count = document_count
        if chunk_count is not None:
            self.chunk_count = chunk_count
        if size_bytes is not None:
            self.total_size_bytes = size_bytes
        self.updated_at = datetime.utcnow()
    
    def add_team_permission(self, user_id: str, permissions: List[AccessPermission]) -> None:
        """Add team member permissions."""
        self.team_permissions[user_id] = permissions
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "collection_type": str(self.collection_type),
            "description": self.description,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.distance_metric,
            "hnsw_construction_ef": self.hnsw_construction_ef,
            "hnsw_m": self.hnsw_m,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "total_size_bytes": self.total_size_bytes,
            "owner_id": self.owner_id,
            "team_id": self.team_id,
            "team_permissions": {
                user_id: [str(perm) for perm in perms]
                for user_id, perms in self.team_permissions.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        }


class MetadataValidator:
    """
    Utility class for metadata validation and normalization.
    
    Ensures metadata conforms to ChromaDB constraints and schema requirements.
    """
    
    @staticmethod
    def validate_string_field(value: Any, field_name: str, max_length: int = 1000) -> str:
        """Validate and normalize string fields."""
        if value is None:
            return ""
        
        if not isinstance(value, str):
            value = str(value)
        
        # Normalize whitespace
        value = value.strip()
        
        # Check length
        if len(value) > max_length:
            logger.warning(f"Field '{field_name}' truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]
        
        return value
    
    @staticmethod
    def validate_integer_field(value: Any, field_name: str, min_value: int = 0, max_value: int = None) -> int:
        """Validate and normalize integer fields."""
        if value is None:
            return 0
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for field '{field_name}': {value}, using 0")
            return 0
        
        # Check bounds
        if int_value < min_value:
            logger.warning(f"Field '{field_name}' value {int_value} below minimum {min_value}")
            return min_value
        
        if max_value is not None and int_value > max_value:
            logger.warning(f"Field '{field_name}' value {int_value} above maximum {max_value}")
            return max_value
        
        return int_value
    
    @staticmethod
    def validate_enum_field(value: Any, enum_class: type, field_name: str, default_value: Any = None):
        """Validate and normalize enum fields."""
        if value is None:
            return default_value if default_value is not None else list(enum_class)[0]
        
        if isinstance(value, enum_class):
            return value
        
        if isinstance(value, str):
            try:
                return enum_class(value)
            except ValueError:
                # Try case-insensitive match
                for enum_member in enum_class:
                    if enum_member.value.lower() == value.lower():
                        return enum_member
        
        logger.warning(f"Invalid enum value for field '{field_name}': {value}")
        return default_value if default_value is not None else list(enum_class)[0]
    
    @classmethod
    def validate_chunk_metadata(cls, metadata: ChunkMetadata) -> ChunkMetadata:
        """Validate and normalize chunk metadata."""
        # Validate string fields
        metadata.chunk_id = cls.validate_string_field(metadata.chunk_id, "chunk_id", 100)
        metadata.source_document_id = cls.validate_string_field(metadata.source_document_id, "source_document_id", 100)
        metadata.document_title = cls.validate_string_field(metadata.document_title, "document_title", 500)
        metadata.user_id = cls.validate_string_field(metadata.user_id, "user_id", 100)
        metadata.team_id = cls.validate_string_field(metadata.team_id, "team_id", 100) or None
        
        # Validate integer fields
        metadata.chunk_sequence_id = cls.validate_integer_field(metadata.chunk_sequence_id, "chunk_sequence_id")
        metadata.chunk_size = cls.validate_integer_field(metadata.chunk_size, "chunk_size")
        
        # Validate enum fields
        metadata.content_type = cls.validate_enum_field(
            metadata.content_type, ContentType, "content_type", ContentType.PROSE
        )
        
        # Validate code language
        if metadata.code_language:
            metadata.code_language = cls.validate_string_field(metadata.code_language, "code_language", 50)
        
        return metadata


# Factory functions for common use cases

def create_document_metadata(
    title: str,
    source_path: str,
    document_type: Union[DocumentType, str] = DocumentType.UNKNOWN,
    user_id: str = "",
    **kwargs
) -> DocumentMetadata:
    """Factory function to create document metadata with validation."""
    if isinstance(document_type, str):
        document_type = MetadataValidator.validate_enum_field(
            document_type, DocumentType, "document_type", DocumentType.UNKNOWN
        )
    
    return DocumentMetadata(
        title=MetadataValidator.validate_string_field(title, "title", 500),
        source_path=MetadataValidator.validate_string_field(source_path, "source_path", 1000),
        document_type=document_type,
        user_id=MetadataValidator.validate_string_field(user_id, "user_id", 100),
        **kwargs
    )


def create_chunk_metadata(
    source_document_id: str,
    document_title: str,
    chunk_sequence_id: int,
    content_type: Union[ContentType, str] = ContentType.PROSE,
    user_id: str = "",
    **kwargs
) -> ChunkMetadata:
    """Factory function to create chunk metadata with validation."""
    if isinstance(content_type, str):
        content_type = MetadataValidator.validate_enum_field(
            content_type, ContentType, "content_type", ContentType.PROSE
        )
    
    metadata = ChunkMetadata(
        source_document_id=source_document_id,
        document_title=document_title,
        chunk_sequence_id=chunk_sequence_id,
        content_type=content_type,
        user_id=user_id,
        **kwargs
    )
    
    return MetadataValidator.validate_chunk_metadata(metadata)


def create_collection_metadata(
    collection_name: str,
    collection_type: Union[CollectionType, str] = CollectionType.GENERAL,
    owner_id: str = "",
    **kwargs
) -> CollectionMetadata:
    """Factory function to create collection metadata with validation."""
    if isinstance(collection_type, str):
        collection_type = MetadataValidator.validate_enum_field(
            collection_type, CollectionType, "collection_type", CollectionType.GENERAL
        )
    
    return CollectionMetadata(
        collection_name=MetadataValidator.validate_string_field(collection_name, "collection_name", 100),
        collection_type=collection_type,
        owner_id=MetadataValidator.validate_string_field(owner_id, "owner_id", 100),
        **kwargs
    )


# Default collection types for configuration
DEFAULT_COLLECTION_TYPES = {
    "fundamental": {
        "type": CollectionType.FUNDAMENTAL,
        "description": "Core knowledge and fundamental concepts",
    },
    "project-specific": {
        "type": CollectionType.PROJECT_SPECIFIC,
        "description": "Project-specific knowledge and documentation",
    },
    "general": {
        "type": CollectionType.GENERAL,
        "description": "General knowledge and miscellaneous content",
    },
    "reference": {
        "type": CollectionType.REFERENCE,
        "description": "Reference materials and external documentation",
    },
} 