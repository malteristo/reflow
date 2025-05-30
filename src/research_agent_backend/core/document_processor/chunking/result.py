"""
Chunk Result Module

Contains the ChunkResult class for representing chunking operation results.
Provides comprehensive representation of a chunking operation result with
detailed metadata, validation, and analysis methods.

Components:
- ChunkResult: Comprehensive chunk result container with metadata and analysis
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """
    Comprehensive representation of a chunking operation result.
    
    Contains the chunk content along with detailed metadata about its position,
    relationships to other chunks, boundary information, and quality metrics.
    Provides methods for analysis, validation, and serialization.
    
    This class serves as the primary data container for chunk information and
    supports various operations needed for chunk management, analysis, and
    quality assessment.
    
    Attributes:
        content: The actual text content of the chunk
        start_position: Starting character position in original text (0-indexed)
        end_position: Ending character position in original text (exclusive)
        chunk_index: Sequential index of this chunk (0-based)
        overlap_with_previous: Number of characters overlapping with previous chunk
        overlap_with_next: Number of characters overlapping with next chunk
        boundary_type: Type of boundary used for this chunk (sentence, paragraph, word, etc.)
        source_section: Optional reference to source DocumentSection
        language_detected: Detected language code (if language detection is enabled)
        content_type: Detected content type (prose, code, table, list, etc.)
        quality_score: Quality assessment score (0.0-1.0)
        processing_metadata: Additional metadata from processing pipeline
    
    Example:
        >>> chunk = ChunkResult(
        ...     content="This is a sample chunk.",
        ...     start_position=0,
        ...     end_position=24,
        ...     chunk_index=0,
        ...     boundary_type="sentence"
        ... )
        >>> print(chunk.get_length())
        24
        >>> print(chunk.get_content_type())
        prose
    """
    
    # Core chunk data
    content: str
    start_position: int
    end_position: int
    chunk_index: int
    
    # Overlap information
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    # Boundary and processing metadata
    boundary_type: str = "word"
    source_section: Optional[str] = None  # Title of source DocumentSection
    language_detected: Optional[str] = None
    content_type: str = "prose"
    quality_score: float = 1.0
    
    # Advanced metadata
    processing_metadata: Dict[str, Any] = None
    creation_timestamp: Optional[float] = None
    
    def __post_init__(self) -> None:
        """
        Validate chunk data and initialize computed properties.
        
        Raises:
            ValueError: If chunk data is invalid
        """
        # Initialize mutable defaults
        if self.processing_metadata is None:
            self.processing_metadata = {}
        
        if self.creation_timestamp is None:
            self.creation_timestamp = time.time()
        
        # Validation
        if not isinstance(self.content, str):
            raise ValueError(f"Content must be string, got: {type(self.content)}")
        
        if self.start_position < 0:
            raise ValueError(f"start_position cannot be negative: {self.start_position}")
        
        if self.end_position < self.start_position:
            raise ValueError(f"end_position ({self.end_position}) cannot be less than start_position ({self.start_position})")
        
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index cannot be negative: {self.chunk_index}")
        
        if self.overlap_with_previous < 0 or self.overlap_with_next < 0:
            raise ValueError("Overlap values cannot be negative")
        
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be between 0.0 and 1.0, got: {self.quality_score}")
        
        # Auto-detect content type if not specified or is default
        if self.content_type == "prose":
            self.content_type = self._detect_content_type()
        
        logger.debug(f"ChunkResult validated: index={self.chunk_index}, length={self.get_length()}, type={self.content_type}")
    
    def get_length(self) -> int:
        """
        Get the length of the chunk content.
        
        Returns:
            Length in characters
        """
        return len(self.content)
    
    def get_word_count(self) -> int:
        """
        Get approximate word count of the chunk.
        
        Returns:
            Number of words (whitespace-separated tokens)
        """
        return len(self.content.split())
    
    def get_line_count(self) -> int:
        """
        Get number of lines in the chunk.
        
        Returns:
            Number of lines
        """
        return self.content.count('\n') + 1 if self.content else 0
    
    def has_overlap(self) -> bool:
        """
        Check if this chunk has overlap with neighboring chunks.
        
        Returns:
            True if chunk has any overlap, False otherwise
        """
        return self.overlap_with_previous > 0 or self.overlap_with_next > 0
    
    def get_overlap_ratio(self) -> float:
        """
        Calculate the overlap ratio as a percentage of chunk content.
        
        Returns:
            Overlap ratio (0.0-1.0)
        """
        total_overlap = self.overlap_with_previous + self.overlap_with_next
        content_length = self.get_length()
        
        if content_length == 0:
            return 0.0
        
        return min(1.0, total_overlap / content_length)
    
    def _detect_content_type(self) -> str:
        """
        Automatically detect content type based on content analysis.
        
        Returns:
            Detected content type string
        """
        content = self.content.strip()
        
        if not content:
            return "empty"
        
        # Code block detection
        if content.startswith("```") or "```" in content:
            return "code_block"
        
        # Inline code detection (high ratio of backticks)
        backtick_ratio = content.count("`") / len(content)
        if backtick_ratio > 0.05:
            return "code_heavy"
        
        # List detection
        lines = content.split('\n')
        list_indicators = sum(1 for line in lines if line.strip().startswith(('- ', '* ', '+ ', '1. ', '2. ')))
        if list_indicators > len(lines) * 0.5:
            return "list"
        
        # Table detection
        pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        if pipe_lines > len(lines) * 0.3:
            return "table"
        
        # Header detection
        header_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if header_lines > 0:
            return "header_heavy"
        
        # Link-heavy content
        link_count = content.count('[') + content.count('http')
        if link_count > len(content.split()) * 0.1:
            return "link_heavy"
        
        # Mathematical content
        math_indicators = ['$$', '$', '\\(', '\\)', '\\[', '\\]']
        if any(indicator in content for indicator in math_indicators):
            return "mathematical"
        
        return "prose"
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the chunk content.
        
        Returns:
            Dictionary with content statistics
        """
        content = self.content
        lines = content.split('\n')
        
        # Character analysis
        char_counts = {
            'total': len(content),
            'letters': sum(1 for c in content if c.isalpha()),
            'digits': sum(1 for c in content if c.isdigit()),
            'whitespace': sum(1 for c in content if c.isspace()),
            'punctuation': sum(1 for c in content if not c.isalnum() and not c.isspace())
        }
        
        # Word analysis
        words = content.split()
        word_stats = {
            'count': len(words),
            'avg_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_count': len(set(word.lower() for word in words))
        }
        
        # Line analysis
        line_stats = {
            'count': len(lines),
            'avg_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip())
        }
        
        # Markdown elements
        markdown_stats = {
            'headers': content.count('#'),
            'bold_markers': content.count('**'),
            'italic_markers': content.count('*') - content.count('**') * 2,
            'code_backticks': content.count('`'),
            'links': content.count('['),
            'images': content.count('![')
        }
        
        return {
            'characters': char_counts,
            'words': word_stats,
            'lines': line_stats,
            'markdown': markdown_stats,
            'content_type': self.content_type,
            'estimated_reading_time_seconds': word_stats['count'] * 0.25  # ~240 WPM
        }
    
    def assess_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of the chunk based on various metrics.
        
        Returns:
            Dictionary with quality assessment results
        """
        content = self.content.strip()
        
        if not content:
            return {
                'overall_score': 0.0,
                'issues': ['Empty content'],
                'recommendations': ['Content should not be empty']
            }
        
        issues = []
        recommendations = []
        score_factors = []
        
        # Length assessment
        length = len(content)
        if length < 50:
            issues.append(f"Very short content ({length} chars)")
            recommendations.append("Consider merging with adjacent chunks")
            score_factors.append(0.6)
        elif length > 2000:
            issues.append(f"Very long content ({length} chars)")
            recommendations.append("Consider splitting into smaller chunks")
            score_factors.append(0.8)
        else:
            score_factors.append(1.0)
        
        # Completeness assessment
        if content.endswith(('.', '!', '?', ':', ';')):
            score_factors.append(1.0)
        elif content.endswith(','):
            issues.append("Ends with comma (incomplete sentence)")
            recommendations.append("Adjust boundary to complete sentence")
            score_factors.append(0.7)
        else:
            issues.append("Ends abruptly (incomplete)")
            recommendations.append("Adjust boundary to natural stopping point")
            score_factors.append(0.5)
        
        # Structure assessment
        lines = content.split('\n')
        if len(lines) > 1:
            # Multi-line content should have reasonable structure
            empty_lines = sum(1 for line in lines if not line.strip())
            if empty_lines / len(lines) > 0.5:
                issues.append("Too many empty lines")
                score_factors.append(0.8)
            else:
                score_factors.append(1.0)
        else:
            score_factors.append(1.0)
        
        # Content type consistency
        if self.boundary_type == "sentence" and not any(content.endswith(p) for p in '.!?'):
            issues.append("Boundary type 'sentence' but doesn't end with sentence punctuation")
            score_factors.append(0.7)
        
        # Calculate overall score
        overall_score = sum(score_factors) / len(score_factors) if score_factors else 0.0
        
        return {
            'overall_score': round(overall_score, 2),
            'issues': issues,
            'recommendations': recommendations,
            'score_factors': score_factors
        }
    
    def to_dict(self, include_content: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert chunk result to dictionary for serialization.
        
        Args:
            include_content: Whether to include the actual content text
            include_metadata: Whether to include processing metadata
            
        Returns:
            Dictionary representation
        """
        result = {
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "length": self.get_length(),
            "word_count": self.get_word_count(),
            "line_count": self.get_line_count(),
            "overlap_with_previous": self.overlap_with_previous,
            "overlap_with_next": self.overlap_with_next,
            "boundary_type": self.boundary_type,
            "content_type": self.content_type,
            "quality_score": self.quality_score,
            "has_overlap": self.has_overlap(),
            "overlap_ratio": self.get_overlap_ratio()
        }
        
        if include_content:
            result["content"] = self.content
        
        if self.source_section:
            result["source_section"] = self.source_section
        
        if self.language_detected:
            result["language_detected"] = self.language_detected
        
        if self.creation_timestamp:
            result["creation_timestamp"] = self.creation_timestamp
        
        if include_metadata and self.processing_metadata:
            result["processing_metadata"] = self.processing_metadata.copy()
        
        return result
    
    def get_preview(self, max_length: int = 100) -> str:
        """
        Get a preview of the chunk content.
        
        Args:
            max_length: Maximum length of preview
            
        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_length:
            return self.content
        
        truncated = self.content[:max_length - 3]
        
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only if we don't lose too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def compare_with(self, other: 'ChunkResult') -> Dict[str, Any]:
        """
        Compare this chunk with another chunk.
        
        Args:
            other: Another ChunkResult to compare with
            
        Returns:
            Dictionary with comparison results
        """
        if not isinstance(other, ChunkResult):
            raise TypeError(f"Can only compare with ChunkResult, got: {type(other)}")
        
        return {
            'length_difference': self.get_length() - other.get_length(),
            'word_count_difference': self.get_word_count() - other.get_word_count(),
            'quality_difference': self.quality_score - other.quality_score,
            'same_content_type': self.content_type == other.content_type,
            'same_boundary_type': self.boundary_type == other.boundary_type,
            'index_distance': abs(self.chunk_index - other.chunk_index),
            'position_distance': abs(self.start_position - other.start_position)
        }
    
    def is_adjacent_to(self, other: 'ChunkResult') -> bool:
        """
        Check if this chunk is adjacent to another chunk.
        
        Args:
            other: Another ChunkResult to check
            
        Returns:
            True if chunks are adjacent in the original text
        """
        return (self.end_position == other.start_position or 
                other.end_position == self.start_position)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        preview = self.get_preview(50)
        return (
            f"ChunkResult(index={self.chunk_index}, length={self.get_length()}, "
            f"type={self.content_type}, preview='{preview}')"
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on content and position."""
        if not isinstance(other, ChunkResult):
            return NotImplemented
        return (
            self.content == other.content and
            self.start_position == other.start_position and
            self.end_position == other.end_position
        )
    
    def __hash__(self) -> int:
        """Hash based on content and position for use in sets/dicts."""
        return hash((self.content, self.start_position, self.end_position)) 