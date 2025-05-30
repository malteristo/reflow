"""
Integration Pipeline Package for Research Agent Vector Database.

This package provides comprehensive integration pipeline capabilities with
modular architecture for maintainability and testability.

Public API:
- DocumentProcessingPipeline: Main document processing orchestration
- IntegratedSearchEngine: Enhanced search engine with ranking
- DataPreparationManager: Data preparation and validation
- CollectionTypeManager: Collection configuration management
- Data models: ProcessingResult, SearchResult, StorageResult, Collection, MockChunk
- Integration support: apply_integration_patches, remove_integration_patches
"""

# Main pipeline classes
from .document_pipeline import DocumentProcessingPipeline
from .search_engine import IntegratedSearchEngine

# Supporting manager classes
from .managers import DataPreparationManager, CollectionTypeManager

# Data models and result objects
from .models import (
    ProcessingResult,
    SearchResult,
    StorageResult,
    Collection,
    MockChunk
)

# Integration support functions
from .integration_support import (
    apply_integration_patches,
    remove_integration_patches
)

# For backward compatibility, ensure all original module exports are available
__all__ = [
    # Main pipeline classes
    'DocumentProcessingPipeline',
    'IntegratedSearchEngine',
    
    # Supporting managers
    'DataPreparationManager',
    'CollectionTypeManager',
    
    # Data models
    'ProcessingResult',
    'SearchResult',
    'StorageResult',
    'Collection',
    'MockChunk',
    
    # Integration support
    'apply_integration_patches',
    'remove_integration_patches'
] 