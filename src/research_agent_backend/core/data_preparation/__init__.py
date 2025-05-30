"""
Data Preparation Package for Research Agent.

This package provides comprehensive data preparation capabilities for vector database operations,
organized into focused modules for maintainability and testability.

Modules:
    types: Enums, dataclasses, and configuration types
    cleaning: DataCleaningService for text and metadata processing
    normalization: NormalizationService for numerical data and embeddings
    dimensionality: DimensionalityReductionService for dimensionality reduction
    manager: Main DataPreparationManager class and factory function

Usage:
    from research_agent_backend.core.data_preparation import DataPreparationManager, DataCleaningConfig
    
    # Or import specific components
    from research_agent_backend.core.data_preparation.cleaning import DataCleaningService
    from research_agent_backend.core.data_preparation.normalization import NormalizationService
"""

# Core types and configuration classes
from .types import (
    NormalizationMethod,
    DimensionalityReductionMethod,
    DataCleaningConfig,
    NormalizationConfig,
    DimensionalityReductionConfig,
    DataPreparationResult
)

# Service classes
from .cleaning import DataCleaningService
from .normalization import NormalizationService
from .dimensionality import DimensionalityReductionService

# Main manager and factory
from .manager import DataPreparationManager, create_data_preparation_manager

# Export all public classes and functions
__all__ = [
    # Enums
    "NormalizationMethod",
    "DimensionalityReductionMethod",
    
    # Configuration classes
    "DataCleaningConfig",
    "NormalizationConfig", 
    "DimensionalityReductionConfig",
    
    # Result class
    "DataPreparationResult",
    
    # Service classes
    "DataCleaningService",
    "NormalizationService",
    "DimensionalityReductionService",
    
    # Main manager and factory
    "DataPreparationManager",
    "create_data_preparation_manager"
] 