"""
Metadata Schema Package for Research Agent Vector Database.

This package provides comprehensive metadata schemas for documents, chunks, and collections
supporting ChromaDB integration and future team scalability features.

Implements FR-KB-002: Rich metadata extraction and storage.

Public API:
- Enums: DocumentType, ContentType, CollectionType, AccessPermission
- Data Classes: DocumentMetadata, ChunkMetadata, CollectionMetadata, HeaderHierarchy
- Validation: MetadataValidator
- Factory Functions: create_document_metadata, create_chunk_metadata, create_collection_metadata
- Constants: DEFAULT_COLLECTION_TYPES
"""

# Enumeration types
from .enums import (
    DocumentType,
    ContentType,
    CollectionType,
    AccessPermission
)

# Header hierarchy class
from .header_hierarchy import HeaderHierarchy

# Metadata classes
from .document_metadata import DocumentMetadata
from .chunk_metadata import ChunkMetadata
from .collection_metadata import CollectionMetadata

# Validation and factory functions
from .validation import (
    MetadataValidator,
    create_document_metadata,
    create_chunk_metadata,
    create_collection_metadata,
    DEFAULT_COLLECTION_TYPES
)

# For backward compatibility, ensure all original module exports are available
__all__ = [
    # Enums
    'DocumentType',
    'ContentType',
    'CollectionType',
    'AccessPermission',
    
    # Data classes
    'HeaderHierarchy',
    'DocumentMetadata',
    'ChunkMetadata',
    'CollectionMetadata',
    
    # Validation and utilities
    'MetadataValidator',
    'create_document_metadata',
    'create_chunk_metadata',
    'create_collection_metadata',
    'DEFAULT_COLLECTION_TYPES'
] 