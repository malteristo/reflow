"""
Data models for re-ranking service results.

Contains data structures for representing re-ranked results with original
metadata preservation and enhanced scoring information.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict
from ..integration_pipeline.models import SearchResult


@dataclass
class RankedResult:
    """
    Result from re-ranking operation with enhanced scoring.
    
    Preserves original search result while adding re-ranking score
    and position information for improved result presentation.
    
    Attributes:
        original_result: The original SearchResult before re-ranking
        rerank_score: New relevance score from cross-encoder (0.0-1.0)
        original_score: Original relevance score from vector search
        rank: Position in re-ranked list (1-based)
        metadata: Additional re-ranking metadata
    """
    original_result: SearchResult
    rerank_score: float
    original_score: float
    rank: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate scores and initialize metadata."""
        if not 0.0 <= self.rerank_score <= 1.0:
            raise ValueError(f"rerank_score must be between 0.0 and 1.0, got: {self.rerank_score}")
        
        if not 0.0 <= self.original_score <= 1.0:
            raise ValueError(f"original_score must be between 0.0 and 1.0, got: {self.original_score}")
        
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got: {self.rank}")
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def content(self) -> str:
        """Get content from original result."""
        return self.original_result.content
    
    @property
    def document_id(self) -> Optional[str]:
        """Get document ID from original result."""
        return self.original_result.document_id
    
    @property
    def chunk_id(self) -> Optional[str]:
        """Get chunk ID from original result."""
        return self.original_result.chunk_id
    
    def get_score_improvement(self) -> float:
        """Calculate improvement from original to rerank score."""
        return self.rerank_score - self.original_score 