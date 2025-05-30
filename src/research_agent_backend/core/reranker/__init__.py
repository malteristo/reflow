"""
Re-ranking service for improving search result precision using cross-encoder models.

This module provides functionality for re-ranking vector search results using
cross-encoder models to improve precision and relevance scoring.

Implements Task 6: Cross-Encoder Re-ranking Service
"""

from .service import RerankerService
from .config import RerankerConfig
from .models import RankedResult

__all__ = [
    'RerankerService',
    'RerankerConfig', 
    'RankedResult'
] 