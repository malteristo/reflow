"""
Data models for re-ranking service results.

Contains data structures for representing re-ranked results with original
metadata preservation and enhanced scoring information.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from ..integration_pipeline.models import SearchResult


@dataclass 
class HighlightedText:
    """Text with keyword highlights for better result presentation."""
    original_text: str
    highlighted_text: str  # Text with HTML <mark> tags around keywords
    matched_keywords: List[str]
    highlight_positions: List[tuple]  # (start, end) positions of highlights


@dataclass
class RelevanceIndicators:
    """Advanced relevance confidence indicators beyond basic scoring."""
    confidence_level: str  # 'very_high', 'high', 'medium', 'low'
    semantic_similarity: float  # Semantic similarity component (0.0-1.0)
    keyword_density: float  # Keyword match density (0.0-1.0)
    structure_relevance: float  # Document structure relevance (0.0-1.0)
    explanation: str  # Human-readable explanation of relevance


@dataclass
class SourceAttribution:
    """Enhanced source document attribution with context."""
    document_title: Optional[str] = None
    document_path: Optional[str] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    line_numbers: Optional[tuple] = None  # (start_line, end_line)
    context_snippet: Optional[str] = None  # Surrounding context
    document_type: Optional[str] = None  # 'markdown', 'pdf', 'text', etc.


@dataclass
class RankedResult:
    """
    Result from re-ranking operation with enhanced scoring and presentation features.
    
    Preserves original search result while adding re-ranking score, position information,
    keyword highlighting, source attribution, and relevance indicators for improved
    result presentation and user experience.
    
    Attributes:
        original_result: The original SearchResult before re-ranking
        rerank_score: New relevance score from cross-encoder (0.0-1.0)
        original_score: Original relevance score from vector search
        rank: Position in re-ranked list (1-based)
        metadata: Additional re-ranking metadata
        highlighted_content: Content with keyword highlighting (FR-RQ-008)
        source_attribution: Enhanced source information (FR-RQ-008)  
        relevance_indicators: Advanced relevance confidence data (FR-RQ-008)
    """
    original_result: SearchResult
    rerank_score: float
    original_score: float
    rank: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    # Enhanced features for FR-RQ-008 compliance
    highlighted_content: Optional[HighlightedText] = None
    source_attribution: Optional[SourceAttribution] = None
    relevance_indicators: Optional[RelevanceIndicators] = None
    
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
    def display_content(self) -> str:
        """Get content for display (highlighted if available, otherwise original)."""
        if self.highlighted_content:
            return self.highlighted_content.highlighted_text
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
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.relevance_indicators:
            return self.relevance_indicators.confidence_level
        
        # Fallback based on rerank_score
        if self.rerank_score >= 0.9:
            return "very_high"
        elif self.rerank_score >= 0.75:
            return "high"
        elif self.rerank_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_source_display(self) -> str:
        """Get formatted source attribution for display."""
        if not self.source_attribution:
            return f"Document: {self.document_id or 'Unknown'}"
        
        parts = []
        if self.source_attribution.document_title:
            parts.append(self.source_attribution.document_title)
        elif self.source_attribution.document_path:
            parts.append(self.source_attribution.document_path)
        elif self.document_id:
            parts.append(f"Document: {self.document_id}")
        
        if self.source_attribution.section_title:
            parts.append(f"Section: {self.source_attribution.section_title}")
        
        if self.source_attribution.page_number:
            parts.append(f"Page: {self.source_attribution.page_number}")
        
        return " | ".join(parts) if parts else "Unknown Source"
    
    def has_keyword_matches(self) -> bool:
        """Check if result has keyword highlighting."""
        return self.highlighted_content is not None and bool(self.highlighted_content.matched_keywords) 