"""
Re-ranking service for improving search result precision using cross-encoder models.

This module provides functionality for re-ranking vector search results using
cross-encoder models to improve precision and relevance scoring, with enhanced
features for keyword highlighting, source attribution, and relevance analysis.

Implements Task 6: Cross-Encoder Re-ranking Service with FR-RQ-008 enhancements
"""

from .service import RerankerService
from .config import RerankerConfig
from .models import (
    RankedResult, 
    HighlightedText, 
    RelevanceIndicators, 
    SourceAttribution
)
from .utils import (
    KeywordHighlighter,
    SourceAttributionExtractor,
    RelevanceAnalyzer
)

__all__ = [
    'RerankerService',
    'RerankerConfig', 
    'RankedResult',
    'HighlightedText',
    'RelevanceIndicators',
    'SourceAttribution',
    'KeywordHighlighter',
    'SourceAttributionExtractor',
    'RelevanceAnalyzer'
] 