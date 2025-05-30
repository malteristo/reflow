"""
Chunking Configuration Module

Contains configuration classes and enums for the chunking system.
Provides comprehensive control over chunking algorithm parameters with validation,
serialization support, and performance tuning options.

Components:
- BoundaryStrategy: Enumeration of boundary detection strategies
- ChunkingMetrics: Protocol for metrics collection
- ChunkConfig: Advanced configuration class for chunking parameters
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Protocol

logger = logging.getLogger(__name__)


class BoundaryStrategy(Enum):
    """
    Enumeration of available boundary detection strategies.
    
    Defines the different approaches for finding optimal chunk boundaries,
    allowing users to customize chunking behavior based on their specific needs.
    """
    INTELLIGENT = "intelligent"  # Use all available boundary types with smart prioritization
    SENTENCE_ONLY = "sentence_only"  # Only respect sentence boundaries
    PARAGRAPH_ONLY = "paragraph_only"  # Only respect paragraph boundaries
    WORD_ONLY = "word_only"  # Only use word boundaries (fastest)
    MARKUP_AWARE = "markup_aware"  # Respect markdown/HTML markup boundaries


class ChunkingMetrics(Protocol):
    """Protocol for chunking metrics collection."""
    
    def record_chunk_created(self, chunk_size: int, boundary_type: str) -> None:
        """Record a chunk creation event."""
        ...
    
    def record_boundary_search(self, search_time_ms: float, boundary_type: str) -> None:
        """Record boundary search performance."""
        ...


@dataclass
class ChunkConfig:
    """
    Advanced configuration class for recursive chunking parameters.
    
    Provides comprehensive control over the chunking algorithm with validation,
    serialization support, and performance tuning options. Supports both basic
    and advanced use cases with sensible defaults.
    
    Key Features:
    - Flexible boundary strategies with intelligent fallbacks
    - Performance tuning parameters for large documents
    - Content-aware options for different text types
    - Validation with detailed error messages
    - JSON serialization for configuration persistence
    - Metrics collection support for optimization
    
    Attributes:
        chunk_size: Maximum size of each chunk in characters (100-10000)
        chunk_overlap: Number of characters to overlap between chunks (0 to chunk_size-1)
        min_chunk_size: Minimum acceptable chunk size to prevent tiny chunks (1 to chunk_size/2)
        preserve_sentences: Whether to respect sentence boundaries when chunking
        preserve_paragraphs: Whether to respect paragraph boundaries when chunking
        preserve_code_blocks: Whether to keep code blocks intact (recommended: True)
        preserve_tables: Whether to keep markdown tables intact
        boundary_strategy: Strategy for boundary detection (see BoundaryStrategy enum)
        max_boundary_search_distance: Maximum characters to search for optimal boundary
        enable_smart_overlap: Use content-aware overlap positioning
        performance_mode: Optimize for speed over boundary quality
        content_type_hints: List of content type hints for specialized processing
        metrics_collector: Optional metrics collection interface
    
    Example:
        >>> # Basic configuration
        >>> config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        
        >>> # Advanced configuration for code documentation
        >>> config = ChunkConfig(
        ...     chunk_size=800,
        ...     chunk_overlap=150,
        ...     preserve_code_blocks=True,
        ...     boundary_strategy=BoundaryStrategy.MARKUP_AWARE,
        ...     content_type_hints=["technical", "code"]
        ... )
        
        >>> # Performance-optimized configuration
        >>> config = ChunkConfig(
        ...     chunk_size=2000,
        ...     chunk_overlap=100,
        ...     boundary_strategy=BoundaryStrategy.WORD_ONLY,
        ...     performance_mode=True
        ... )
    """
    
    # Core chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Content preservation options
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_links: bool = True
    
    # Advanced boundary detection
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.INTELLIGENT
    max_boundary_search_distance: int = 100
    sentence_min_length: int = 10  # Minimum length to consider as sentence
    paragraph_min_length: int = 50  # Minimum length to consider as paragraph
    
    # Performance and optimization
    enable_smart_overlap: bool = True
    performance_mode: bool = False
    cache_boundary_patterns: bool = True
    
    # Content type awareness
    content_type_hints: List[str] = None
    language_code: Optional[str] = None  # For language-specific processing
    
    # Metrics and debugging
    metrics_collector: Optional[ChunkingMetrics] = None
    debug_mode: bool = False
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters with comprehensive checks.
        
        Performs extensive validation of all parameters to ensure they are
        within valid ranges and compatible with each other. Provides detailed
        error messages for debugging configuration issues.
        
        Raises:
            ValueError: If any parameter is invalid or incompatible
            TypeError: If parameter types are incorrect
        """
        # Initialize mutable defaults
        if self.content_type_hints is None:
            self.content_type_hints = []
        
        # Core parameter validation
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive integer, got: {self.chunk_size}")
        
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size should be at least 100 for meaningful chunks, got: {self.chunk_size}")
        
        if self.chunk_size > 10000:
            logger.warning(f"Large chunk_size ({self.chunk_size}) may impact performance")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        
        if self.chunk_overlap > self.chunk_size * 0.5:
            logger.warning(
                f"Large overlap ratio ({self.chunk_overlap/self.chunk_size:.1%}) may cause excessive duplication"
            )
        
        # Min chunk size validation
        if not isinstance(self.min_chunk_size, int) or self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive integer, got: {self.min_chunk_size}")
        
        if self.min_chunk_size > self.chunk_size // 2:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) should not exceed half of chunk_size ({self.chunk_size//2})"
            )
        
        # Boundary strategy validation
        if not isinstance(self.boundary_strategy, BoundaryStrategy):
            raise TypeError(f"boundary_strategy must be BoundaryStrategy enum, got: {type(self.boundary_strategy)}")
        
        # Search distance validation
        if self.max_boundary_search_distance < 10:
            raise ValueError("max_boundary_search_distance should be at least 10")
        
        if self.max_boundary_search_distance > self.chunk_size:
            logger.warning(
                f"max_boundary_search_distance ({self.max_boundary_search_distance}) "
                f"exceeds chunk_size ({self.chunk_size}), limiting to chunk_size"
            )
            self.max_boundary_search_distance = self.chunk_size
        
        # Content type hints validation
        if not isinstance(self.content_type_hints, list):
            raise TypeError(f"content_type_hints must be list, got: {type(self.content_type_hints)}")
        
        # Language code validation
        if self.language_code is not None and not isinstance(self.language_code, str):
            raise TypeError(f"language_code must be string or None, got: {type(self.language_code)}")
        
        # Compatibility checks
        if self.performance_mode and self.boundary_strategy == BoundaryStrategy.INTELLIGENT:
            logger.info("Performance mode enabled, using simplified boundary detection")
            self.boundary_strategy = BoundaryStrategy.WORD_ONLY
        
        logger.debug(
            f"ChunkConfig validated: size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"strategy={self.boundary_strategy.value}, performance={self.performance_mode}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation with all configuration parameters
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "preserve_sentences": self.preserve_sentences,
            "preserve_paragraphs": self.preserve_paragraphs,
            "preserve_code_blocks": self.preserve_code_blocks,
            "preserve_tables": self.preserve_tables,
            "preserve_links": self.preserve_links,
            "boundary_strategy": self.boundary_strategy.value,
            "max_boundary_search_distance": self.max_boundary_search_distance,
            "sentence_min_length": self.sentence_min_length,
            "paragraph_min_length": self.paragraph_min_length,
            "enable_smart_overlap": self.enable_smart_overlap,
            "performance_mode": self.performance_mode,
            "cache_boundary_patterns": self.cache_boundary_patterns,
            "content_type_hints": self.content_type_hints.copy(),
            "language_code": self.language_code,
            "debug_mode": self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration parameters
            
        Returns:
            ChunkConfig instance
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Convert boundary strategy string back to enum
        if "boundary_strategy" in data and isinstance(data["boundary_strategy"], str):
            try:
                data["boundary_strategy"] = BoundaryStrategy(data["boundary_strategy"])
            except ValueError as e:
                raise ValueError(f"Invalid boundary_strategy '{data['boundary_strategy']}': {e}")
        
        # Create instance with validated parameters
        return cls(**data)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert configuration to JSON string.
        
        Args:
            indent: Optional indentation for pretty printing
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkConfig':
        """
        Create configuration from JSON string.
        
        Args:
            json_str: JSON string with configuration
            
        Returns:
            ChunkConfig instance
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
            ValueError: If configuration parameters are invalid
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def copy(self, **overrides) -> 'ChunkConfig':
        """
        Create a copy of this configuration with optional parameter overrides.
        
        Args:
            **overrides: Parameters to override in the copy
            
        Returns:
            New ChunkConfig instance with overrides applied
            
        Example:
            >>> config = ChunkConfig(chunk_size=1000)
            >>> small_config = config.copy(chunk_size=500, chunk_overlap=100)
        """
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)
    
    def get_optimal_settings_for_content_type(self, content_type: str) -> 'ChunkConfig':
        """
        Get optimized configuration for specific content types.
        
        Args:
            content_type: Type of content ("technical", "narrative", "code", "academic")
            
        Returns:
            Optimized ChunkConfig for the content type
        """
        optimizations = {
            "technical": {
                "preserve_code_blocks": True,
                "preserve_tables": True,
                "boundary_strategy": BoundaryStrategy.MARKUP_AWARE,
                "sentence_min_length": 15
            },
            "narrative": {
                "preserve_sentences": True,
                "preserve_paragraphs": True,
                "boundary_strategy": BoundaryStrategy.SENTENCE_ONLY,
                "chunk_size": 1200
            },
            "code": {
                "preserve_code_blocks": True,
                "preserve_sentences": False,
                "boundary_strategy": BoundaryStrategy.MARKUP_AWARE,
                "chunk_overlap": 50
            },
            "academic": {
                "preserve_sentences": True,
                "preserve_paragraphs": True,
                "chunk_size": 800,
                "chunk_overlap": 150
            }
        }
        
        if content_type in optimizations:
            return self.copy(**optimizations[content_type])
        else:
            logger.warning(f"Unknown content type '{content_type}', using default settings")
            return self.copy()
    
    def validate_compatibility_with_text(self, text: str) -> List[str]:
        """
        Validate configuration compatibility with specific text.
        
        Args:
            text: Text to analyze for compatibility
            
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        text_length = len(text)
        
        if text_length < self.chunk_size:
            warnings.append(f"Text length ({text_length}) is smaller than chunk_size ({self.chunk_size})")
        
        if text_length < self.min_chunk_size * 2:
            warnings.append(f"Text is very short, may not benefit from chunking")
        
        # Check for code blocks if preservation is enabled
        if self.preserve_code_blocks and "```" in text:
            code_blocks = text.count("```") // 2
            warnings.append(f"Found {code_blocks} code blocks, ensure chunk_size can accommodate them")
        
        # Check sentence structure if sentence preservation is enabled
        if self.preserve_sentences and text.count('.') < 3:
            warnings.append("Few sentence boundaries found, may not benefit from sentence preservation")
        
        return warnings
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ChunkConfig(size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"strategy={self.boundary_strategy.value}, performance={self.performance_mode})"
        ) 