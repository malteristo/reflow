"""
Re-ranking pipeline integration for improving search result precision.

This module provides pipeline components for integrating re-ranking functionality
with existing retrieval pipelines and search engines.

Implements Task 6.4: Integration with Retrieval Pipeline
"""

from .processor import RerankerPipelineProcessor
from .config import PipelineConfig
from .models import PipelineResult
from .enhanced_search import EnhancedSearchEngine

__all__ = [
    'RerankerPipelineProcessor',
    'PipelineConfig', 
    'PipelineResult',
    'EnhancedSearchEngine'
] 