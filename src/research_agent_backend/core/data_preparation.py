"""
Data Preparation for Research Agent - Backward Compatibility Layer.

This module provides backward compatibility for existing imports while the actual
implementation has been refactored into a modular package structure.

New code should import directly from the data_preparation package:
    from research_agent_backend.core.data_preparation import DataPreparationManager, DataCleaningConfig

This file maintains compatibility for existing code that imports from data_preparation.py:
    from research_agent_backend.core.data_preparation import DataPreparationManager  # Works
    
Refactored Architecture:
- data_preparation/types.py: Enums, dataclasses, and configuration types
- data_preparation/cleaning.py: DataCleaningService for text and metadata processing
- data_preparation/normalization.py: NormalizationService for numerical data and embeddings  
- data_preparation/dimensionality.py: DimensionalityReductionService for reduction algorithms
- data_preparation/manager.py: Main DataPreparationManager class and factory function

Implements FR-KB-003: Data preparation and quality assurance.
"""

# Import all public classes from the modular package for backward compatibility
from .data_preparation import (
    # Enums
    NormalizationMethod,
    DimensionalityReductionMethod,
    
    # Configuration classes
    DataCleaningConfig,
    NormalizationConfig,
    DimensionalityReductionConfig,
    
    # Result class
    DataPreparationResult,
    
    # Service classes
    DataCleaningService,
    NormalizationService,
    DimensionalityReductionService,
    
    # Main manager and factory
    DataPreparationManager,
    create_data_preparation_manager
)

# Maintain the same public API
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