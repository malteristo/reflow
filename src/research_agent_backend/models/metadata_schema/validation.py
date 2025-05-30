"""
Metadata validation utilities and factory functions.

This module provides validation logic and factory functions for creating
metadata objects with proper validation and defaults.

Ensures metadata conforms to ChromaDB constraints and schema requirements.
"""

import logging
from typing import Any, Union, Dict

from .enums import DocumentType, ContentType, CollectionType
from .document_metadata import DocumentMetadata
from .chunk_metadata import ChunkMetadata
from .collection_metadata import CollectionMetadata


logger = logging.getLogger(__name__)


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