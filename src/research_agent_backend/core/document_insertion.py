"""
Document Insertion Manager for Research Agent Vector Database.

This module provides backwards compatibility for the DocumentInsertionManager
by importing from the modular document_insertion package.

Implements FR-KB-002: Document insertion with rich metadata.
Implements FR-ST-002: Vector database operations with transaction support.
"""

# Import all public API from the modular package
from .document_insertion import (
    # Main manager class
    DocumentInsertionManager,
    
    # Exception classes
    InsertionError,
    ValidationError,
    TransactionError,
    
    # Result classes
    InsertionResult,
    BatchInsertionResult,
    
    # Factory function
    create_document_insertion_manager
)

# For compatibility, also export service components
from .document_insertion import (
    DocumentValidator,
    DocumentPreparationService,
    DocumentChunker,
    ChunkMetadataFactory,
    EmbeddingService,
    TransactionManager
)

# Ensure all exports are available at module level
__all__ = [
    'DocumentInsertionManager',
    'InsertionError',
    'ValidationError',
    'TransactionError',
    'InsertionResult',
    'BatchInsertionResult',
    'create_document_insertion_manager',
    'DocumentValidator',
    'DocumentPreparationService',
    'DocumentChunker',
    'ChunkMetadataFactory',
    'EmbeddingService',
    'TransactionManager'
] 