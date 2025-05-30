"""
Inline Metadata Extraction Module

Handles extraction of inline metadata tags and comments from markdown content.
Supports multiple formats: @tag:value, [[key:value]], {key:value}, HTML comments.

Implements FR-KB-003.2: Inline metadata detection and extraction.
"""

import re
import json
import time
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class InlineTag:
    """Represents an inline tag like @tag:value with position tracking.
    
    Used for simple key-value tags embedded directly in content.
    
    Attributes:
        key: The tag key/name
        value: The tag value
        position: Character position in the source content where tag was found
        line_number: Line number where tag appears (if tracked)
    """
    key: str
    value: str
    position: int
    line_number: Optional[int] = None
    
    def __post_init__(self):
        """Validate tag data."""
        if not self.key or not isinstance(self.key, str):
            raise ValueError(f"key must be a non-empty string, got: {self.key}")
        if not isinstance(self.value, str):
            raise ValueError(f"value must be a string, got: {type(self.value)}")
        if self.position < 0:
            raise ValueError(f"position must be non-negative, got: {self.position}")


@dataclass 
class InlineMetadataItem:
    """Represents an inline metadata item like <!-- @key: value -->.
    
    More complex than InlineTag, can include parsed JSON values and additional metadata.
    
    Attributes:
        key: The metadata key
        value: Raw string value
        position: Character position in source content
        parsed_value: Parsed JSON value (if applicable)
        line_number: Line number where metadata appears (if tracked)
        metadata_type: Type of metadata pattern ("comment", "json_comment", etc.)
    """
    key: str
    value: str
    position: int
    parsed_value: Any = None
    line_number: Optional[int] = None
    metadata_type: str = "comment"
    
    def __post_init__(self):
        """Validate metadata item data."""
        if not self.key or not isinstance(self.key, str):
            raise ValueError(f"key must be a non-empty string, got: {self.key}")
        if not isinstance(self.value, str):
            raise ValueError(f"value must be a string, got: {type(self.value)}")
        if self.position < 0:
            raise ValueError(f"position must be non-negative, got: {self.position}")


@dataclass
class InlineMetadataResult:
    """Result container for inline metadata extraction.
    
    Contains all extracted tags and metadata items with processing statistics.
    
    Attributes:
        tags: List of simple inline tags found
        metadata: List of complex metadata items found
        cleaned_content: Content with metadata removed (if requested)
        patterns_matched: Set of pattern types that had matches
        extraction_time_ms: Time taken for extraction (if tracked)
        total_matches: Total number of items found
    """
    tags: List[InlineTag] = field(default_factory=list)
    metadata: List[InlineMetadataItem] = field(default_factory=list)
    cleaned_content: str = ""
    patterns_matched: Set[str] = field(default_factory=set)
    extraction_time_ms: Optional[float] = None
    
    @property
    def total_matches(self) -> int:
        """Total number of metadata items found."""
        return len(self.tags) + len(self.metadata)
    
    @property 
    def has_metadata(self) -> bool:
        """True if any metadata was found."""
        return self.total_matches > 0
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Convert metadata items to a dictionary."""
        result = {}
        
        # Add simple tags
        for tag in self.tags:
            result[tag.key] = tag.value
        
        # Add complex metadata items
        for item in self.metadata:
            if item.parsed_value is not None:
                result[item.key] = item.parsed_value
            else:
                result[item.key] = item.value
        
        return result


class InlineMetadataExtractor:
    """Extractor for inline metadata tags and comments.
    
    Supports multiple inline metadata formats with configurable extraction
    and performance tracking capabilities.
    """
    
    def __init__(
        self, 
        enable_performance_tracking: bool = False,
        enable_line_tracking: bool = False
    ):
        """Initialize extractor with compiled regex patterns.
        
        Args:
            enable_performance_tracking: Whether to track extraction performance
            enable_line_tracking: Whether to track line numbers for positions
        """
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_line_tracking = enable_line_tracking
        
        # Different metadata patterns - compiled for performance
        self.tag_pattern = re.compile(r'@(\w+):(\w+)')
        self.bracket_pattern = re.compile(r'\[\[(\w+):([^\]]+)\]\]')
        self.brace_pattern = re.compile(r'\{(\w+):([^}]+)\}')
        self.comment_pattern = re.compile(r'<!-- @([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^>]+?) -->')
        self.json_comment_pattern = re.compile(r'<!-- @([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\{[^>]+?\}) -->')
        
        # Performance tracking
        self._performance_stats = {
            'total_extractions': 0,
            'total_items_found': 0,
            'avg_extraction_time_ms': 0.0,
            'pattern_usage': {
                'inline_tags': 0,
                'bracket_tags': 0,
                'brace_tags': 0,
                'html_comments': 0,
                'json_comments': 0
            }
        }
    
    def extract(self, content: str, remove_from_content: bool = False) -> InlineMetadataResult:
        """Extract inline metadata from content with comprehensive tracking.
        
        Args:
            content: Content to extract metadata from
            remove_from_content: Whether to remove found metadata from content
            
        Returns:
            InlineMetadataResult with all found metadata and statistics
        """
        if not isinstance(content, str):
            raise ValueError(f"content must be a string, got {type(content)}")
        
        start_time = time.time() if self.enable_performance_tracking else None
        
        tags = []
        metadata = []
        cleaned_content = content
        patterns_matched = set()
        
        # Line mapping for position-to-line conversion (if enabled)
        line_map = self._build_line_map(content) if self.enable_line_tracking else None
        
        try:
            # Extract @tag:value patterns
            for match in self.tag_pattern.finditer(content):
                tag = InlineTag(
                    key=match.group(1),
                    value=match.group(2), 
                    position=match.start(),
                    line_number=self._get_line_number(match.start(), line_map) if line_map else None
                )
                tags.append(tag)
                patterns_matched.add("inline_tags")
                
                if remove_from_content:
                    cleaned_content = cleaned_content.replace(match.group(0), "")
            
            # Extract [[key:value]] patterns  
            for match in self.bracket_pattern.finditer(content):
                tag = InlineTag(
                    key=match.group(1),
                    value=match.group(2),
                    position=match.start(),
                    line_number=self._get_line_number(match.start(), line_map) if line_map else None
                )
                tags.append(tag)
                patterns_matched.add("bracket_tags")
                
                if remove_from_content:
                    cleaned_content = cleaned_content.replace(match.group(0), "")
            
            # Extract {key:value} patterns
            for match in self.brace_pattern.finditer(content):
                tag = InlineTag(
                    key=match.group(1),
                    value=match.group(2),
                    position=match.start(),
                    line_number=self._get_line_number(match.start(), line_map) if line_map else None
                )
                tags.append(tag)
                patterns_matched.add("brace_tags")
                
                if remove_from_content:
                    cleaned_content = cleaned_content.replace(match.group(0), "")
            
            # Extract JSON metadata from comments first (to avoid duplicate processing)
            json_positions = set()
            for match in self.json_comment_pattern.finditer(content):
                try:
                    parsed_json = json.loads(match.group(2))
                    item = InlineMetadataItem(
                        key=match.group(1),
                        value=match.group(2),
                        position=match.start(),
                        parsed_value=parsed_json,
                        line_number=self._get_line_number(match.start(), line_map) if line_map else None,
                        metadata_type="json_comment"
                    )
                    metadata.append(item)
                    json_positions.add(match.start())
                    patterns_matched.add("json_comments")
                    
                    if remove_from_content:
                        cleaned_content = cleaned_content.replace(match.group(0), "")
                        
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSON metadata at position %d: %s", match.start(), str(e))
            
            # Extract regular comment metadata (skip positions already processed as JSON)
            for match in self.comment_pattern.finditer(content):
                if match.start() not in json_positions:
                    item = InlineMetadataItem(
                        key=match.group(1),
                        value=match.group(2).strip(),
                        position=match.start(),
                        line_number=self._get_line_number(match.start(), line_map) if line_map else None,
                        metadata_type="comment"
                    )
                    metadata.append(item)
                    patterns_matched.add("html_comments")
                    
                    if remove_from_content:
                        cleaned_content = cleaned_content.replace(match.group(0), "")
            
            # Create result with comprehensive statistics
            result = InlineMetadataResult(
                tags=tags,
                metadata=metadata,
                cleaned_content=cleaned_content,
                patterns_matched=patterns_matched
            )
            
            # Add performance tracking
            if self.enable_performance_tracking:
                extraction_time = (time.time() - start_time) * 1000
                result.extraction_time_ms = extraction_time
                self._update_performance_stats(result, extraction_time)
            
            logger.debug("Extracted %d tags and %d metadata items using patterns: %s",
                        len(tags), len(metadata), patterns_matched)
            
            return result
            
        except Exception as e:
            logger.error("Inline metadata extraction failed: %s", str(e))
            raise
    
    def _build_line_map(self, content: str) -> List[int]:
        """Build a mapping from character positions to line numbers."""
        line_map = []
        current_line = 1
        
        for i, char in enumerate(content):
            line_map.append(current_line)
            if char == '\n':
                current_line += 1
        
        return line_map
    
    def _get_line_number(self, position: int, line_map: List[int]) -> int:
        """Get line number for a character position."""
        if line_map and 0 <= position < len(line_map):
            return line_map[position]
        return 1
    
    def _update_performance_stats(self, result: InlineMetadataResult, extraction_time_ms: float):
        """Update performance tracking statistics."""
        stats = self._performance_stats
        stats['total_extractions'] += 1
        stats['total_items_found'] += result.total_matches
        
        # Update running average
        total = stats['total_extractions']
        current_avg = stats['avg_extraction_time_ms']
        stats['avg_extraction_time_ms'] = (
            (current_avg * (total - 1) + extraction_time_ms) / total
        )
        
        # Track pattern usage
        for pattern in result.patterns_matched:
            if pattern in stats['pattern_usage']:
                stats['pattern_usage'][pattern] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return self._performance_stats.copy()
    
    def extract_from_multiple_documents(
        self, 
        documents: List[Tuple[str, str]], 
        remove_from_content: bool = False
    ) -> List[Tuple[str, InlineMetadataResult]]:
        """Extract metadata from multiple documents efficiently.
        
        Args:
            documents: List of (document_id, content) tuples
            remove_from_content: Whether to remove metadata from content
            
        Returns:
            List of (document_id, InlineMetadataResult) tuples
        """
        results = []
        start_time = time.time()
        
        logger.info("Starting batch extraction for %d documents", len(documents))
        
        for doc_id, content in documents:
            try:
                result = self.extract(content, remove_from_content)
                results.append((doc_id, result))
            except Exception as e:
                logger.error("Failed to extract metadata from document %s: %s", doc_id, str(e))
                # Continue with other documents
                continue
        
        batch_time = (time.time() - start_time) * 1000
        logger.info("Completed batch extraction in %.2f ms, processed %d/%d documents",
                   batch_time, len(results), len(documents))
        
        return results
    
    def validate_metadata_syntax(self, content: str) -> Dict[str, List[str]]:
        """Validate metadata syntax and return any issues found.
        
        Args:
            content: Content to validate
            
        Returns:
            Dictionary with pattern names as keys and lists of issues as values
        """
        issues = {
            'json_comments': [],
            'malformed_tags': [],
            'duplicate_keys': []
        }
        
        # Check JSON comment syntax
        for match in self.json_comment_pattern.finditer(content):
            try:
                json.loads(match.group(2))
            except json.JSONDecodeError as e:
                issues['json_comments'].append(
                    f"Line {self._get_line_from_position(content, match.start())}: {str(e)}"
                )
        
        # Check for duplicate keys
        result = self.extract(content)
        all_keys = [tag.key for tag in result.tags] + [item.key for item in result.metadata]
        seen_keys = set()
        for key in all_keys:
            if key in seen_keys:
                issues['duplicate_keys'].append(f"Duplicate key: {key}")
            seen_keys.add(key)
        
        return {k: v for k, v in issues.items() if v}  # Only return non-empty lists
    
    def _get_line_from_position(self, content: str, position: int) -> int:
        """Get line number for a character position."""
        return content[:position].count('\n') + 1 