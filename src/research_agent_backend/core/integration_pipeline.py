"""
Integration pipeline for end-to-end document processing workflows.

This module provides backwards compatibility for the integration pipeline system
by importing from the modular integration_pipeline package.

REFACTOR PHASE: Improved implementation with better structure and realistic behavior.
"""

# Import all public API from the modular package
from .integration_pipeline import (
    # Main pipeline classes
    DocumentProcessingPipeline,
    IntegratedSearchEngine,
    
    # Supporting manager classes
    DataPreparationManager,
    CollectionTypeManager,
    
    # Data models and result objects
    ProcessingResult,
    SearchResult,
    StorageResult,
    Collection,
    MockChunk,
    
    # Integration support functions
    apply_integration_patches,
    remove_integration_patches
)

# Module-level imports for consistency with original module
import logging

# For compatibility, also provide the original module exports
__all__ = [
    'DocumentProcessingPipeline',
    'IntegratedSearchEngine', 
    'DataPreparationManager',
    'CollectionTypeManager',
    'ProcessingResult',
    'SearchResult',
    'StorageResult',
    'Collection',
    'MockChunk',
    'apply_integration_patches',
    'remove_integration_patches'
] 