"""
Metadata Schema for Research Agent Vector Database.

This module provides backwards compatibility for the metadata schema system
by importing from the modular metadata_schema package.

Implements FR-KB-002: Rich metadata extraction and storage.
"""

# Import all public API from the modular package
from .metadata_schema import (
    # Enumeration types
    DocumentType,
    ContentType,
    CollectionType,
    AccessPermission,
    
    # Data classes
    HeaderHierarchy,
    DocumentMetadata,
    ChunkMetadata,
    CollectionMetadata,
    
    # Validation and factory functions
    MetadataValidator,
    create_document_metadata,
    create_chunk_metadata,
    create_collection_metadata,
    DEFAULT_COLLECTION_TYPES
)

# Module-level imports for logging compatibility
import logging

# For compatibility, also provide the original module exports
__all__ = [
    'DocumentType',
    'ContentType',
    'CollectionType',
    'AccessPermission',
    'HeaderHierarchy',
    'DocumentMetadata',
    'ChunkMetadata',
    'CollectionMetadata',
    'MetadataValidator',
    'create_document_metadata',
    'create_chunk_metadata',
    'create_collection_metadata',
    'DEFAULT_COLLECTION_TYPES'
] 