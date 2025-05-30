"""
Document Insertion Package for Research Agent Vector Database.

This package provides comprehensive document insertion capabilities with
modular architecture for maintainability and testability.

Public API:
- DocumentInsertionManager: Main orchestration class
- Exception classes: InsertionError, ValidationError, TransactionError
- Result classes: InsertionResult, BatchInsertionResult
- Factory function: create_document_insertion_manager
"""

# Main manager class
from .manager import DocumentInsertionManager

# Exception classes and result models
from .exceptions import (
    InsertionError,
    ValidationError, 
    TransactionError,
    InsertionResult,
    BatchInsertionResult
)

# Service components for advanced usage
from .validation import DocumentValidator, DocumentPreparationService
from .chunking import DocumentChunker, ChunkMetadataFactory
from .embeddings import EmbeddingService
from .transactions import TransactionManager

# Factory function for backward compatibility
from typing import Optional, Any
from ...core.vector_store import ChromaDBManager
from ...utils.config import ConfigManager


def create_document_insertion_manager(
    config_file: Optional[str] = None,
    vector_store: Optional[ChromaDBManager] = None,
    **kwargs
) -> DocumentInsertionManager:
    """
    Create DocumentInsertionManager with default configuration.
    
    Args:
        config_file: Optional config file path
        vector_store: Optional vector store instance
        **kwargs: Additional arguments for DocumentInsertionManager
        
    Returns:
        Configured DocumentInsertionManager instance
    """
    # Create config manager
    config_manager = ConfigManager() if config_file is None else ConfigManager(config_file)
    
    # Create vector store if not provided
    if vector_store is None:
        from ...core.vector_store import create_chroma_manager
        vector_store = create_chroma_manager(config_file=config_file)
    
    # Create data preparation manager
    from ...core.data_preparation import create_data_preparation_manager
    data_prep_manager = create_data_preparation_manager(config_manager=config_manager)
    
    # Create collection type manager
    from ...core.collection_type_manager import create_collection_type_manager
    collection_type_manager = create_collection_type_manager(config_manager=config_manager)
    
    return DocumentInsertionManager(
        vector_store=vector_store,
        data_preparation_manager=data_prep_manager,
        config_manager=config_manager,
        collection_type_manager=collection_type_manager,
        **kwargs
    )


__all__ = [
    # Main class
    'DocumentInsertionManager',
    
    # Exception classes
    'InsertionError',
    'ValidationError', 
    'TransactionError',
    
    # Result classes
    'InsertionResult',
    'BatchInsertionResult',
    
    # Service components
    'DocumentValidator',
    'DocumentPreparationService',
    'DocumentChunker',
    'ChunkMetadataFactory',
    'EmbeddingService',
    'TransactionManager',
    
    # Factory function
    'create_document_insertion_manager'
] 