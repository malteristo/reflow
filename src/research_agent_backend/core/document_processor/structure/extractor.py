"""
Section Extractor Module - Advanced Section Filtering and Analysis

This module provides the SectionExtractor class for extracting and filtering
document sections with various search criteria.

Key Components:
- SectionExtractor: Advanced utility for targeted section extraction

Usage:
    >>> from .tree import DocumentTree
    >>> extractor = SectionExtractor()
    >>> tree = build_sample_tree()
    >>> methods_section = extractor.extract_by_title(tree, "Methods")
"""

import re
from typing import List, Dict, Any, Optional, Callable
import logging

from .tree import DocumentTree
from .section import DocumentSection

logger = logging.getLogger(__name__)


class SectionExtractor:
    """
    Advanced utility class for extracting and filtering document sections.
    
    Provides a comprehensive set of methods for targeted section extraction,
    content filtering, and document analysis. Supports various search criteria
    including title matching, level filtering, pattern matching, and custom
    predicate functions.
    
    Designed for high-performance operations on large document trees with
    caching and optimization features for repeated queries.
    
    Example:
        >>> extractor = SectionExtractor()
        >>> tree = build_sample_tree()
        >>> methods_section = extractor.extract_by_title(tree, "Methods")
        >>> all_level2 = extractor.extract_by_level_range(tree, 2, 2)
    """
    
    def __init__(self, cache_enabled: bool = True) -> None:
        """
        Initialize SectionExtractor with optional caching.
        
        Args:
            cache_enabled: Whether to enable result caching for performance
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        
        logger.debug(f"SectionExtractor initialized with caching: {cache_enabled}")
    
    def _get_cache_key(self, operation: str, *args: Any) -> str:
        """Generate cache key for operation and arguments."""
        return f"{operation}:{hash(str(args))}"
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Cache result if caching is enabled."""
        if self.cache_enabled:
            self._cache[key] = result
    
    def _get_cached_result(self, key: str) -> Any:
        """Get cached result if available."""
        if self.cache_enabled:
            return self._cache.get(key)
        return None
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
        logger.debug("SectionExtractor cache cleared")
    
    def extract_by_title(
        self, 
        tree: DocumentTree, 
        title: str, 
        case_sensitive: bool = True
    ) -> Optional[DocumentSection]:
        """
        Extract a section by title with caching support.
        
        Args:
            tree: DocumentTree to search
            title: Title to find
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Section with matching title, or None if not found
            
        Raises:
            TypeError: If tree is not a DocumentTree
        """
        if not isinstance(tree, DocumentTree):
            raise TypeError(f"Tree must be DocumentTree, got: {type(tree)}")
        
        cache_key = self._get_cache_key("extract_by_title", id(tree), title, case_sensitive)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = tree.find_section_by_title(title, case_sensitive)
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted section by title '{title}': {'found' if result else 'not found'}")
        return result
    
    def extract_by_level_range(
        self, 
        tree: DocumentTree, 
        min_level: int, 
        max_level: int
    ) -> List[DocumentSection]:
        """
        Extract sections within a level range with validation.
        
        Args:
            tree: DocumentTree to search
            min_level: Minimum header level (inclusive)
            max_level: Maximum header level (inclusive)
            
        Returns:
            List of sections within the level range, ordered by document position
            
        Raises:
            ValueError: If level range is invalid
        """
        if min_level > max_level:
            raise ValueError(f"min_level ({min_level}) cannot be greater than max_level ({max_level})")
        
        cache_key = self._get_cache_key("extract_by_level_range", id(tree), min_level, max_level)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = tree.get_sections_by_level_range(min_level, max_level)
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted {len(result)} sections in level range {min_level}-{max_level}")
        return result
    
    def extract_with_children(self, tree: DocumentTree, title: str) -> Optional[DocumentSection]:
        """
        Extract a section and all its children (returns section with full subtree).
        
        Args:
            tree: DocumentTree to search
            title: Title of section to extract
            
        Returns:
            Section with all children preserved, or None if not found
        """
        return tree.find_section_by_title(title)
    
    def extract_by_predicate(
        self, 
        tree: DocumentTree, 
        predicate: Callable[[DocumentSection], bool]
    ) -> List[DocumentSection]:
        """
        Extract sections matching a custom predicate function.
        
        Args:
            tree: DocumentTree to search
            predicate: Function that takes DocumentSection and returns bool
            
        Returns:
            List of sections matching the predicate
            
        Example:
            >>> # Extract sections with long content
            >>> long_sections = extractor.extract_by_predicate(
            ...     tree, lambda s: len(s.content) > 100
            ... )
        """
        if not callable(predicate):
            raise TypeError("Predicate must be callable")
        
        result = [section for section in tree._sections if predicate(section)]
        
        logger.debug(f"Extracted {len(result)} sections matching custom predicate")
        return result
    
    def extract_by_title_pattern(self, tree: DocumentTree, pattern: str) -> List[DocumentSection]:
        """
        Extract sections with titles matching a regex pattern.
        
        Args:
            tree: DocumentTree to search
            pattern: Regex pattern to match against titles
            
        Returns:
            List of sections with matching titles
            
        Raises:
            re.error: If pattern is invalid regex
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise re.error(f"Invalid regex pattern '{pattern}': {e}")
        
        cache_key = self._get_cache_key("extract_by_title_pattern", id(tree), pattern)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = []
        for section in tree._sections:
            if section.title and regex.search(section.title):
                result.append(section)
        
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted {len(result)} sections matching pattern '{pattern}'")
        return result
    
    def extract_by_content_pattern(self, tree: DocumentTree, pattern: str) -> List[DocumentSection]:
        """
        Extract sections with content matching a regex pattern.
        
        Args:
            tree: DocumentTree to search
            pattern: Regex pattern to match against content
            
        Returns:
            List of sections with matching content
        """
        try:
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as e:
            raise re.error(f"Invalid regex pattern '{pattern}': {e}")
        
        result = []
        for section in tree._sections:
            if section.content and regex.search(section.content):
                result.append(section)
        
        logger.debug(f"Extracted {len(result)} sections with content matching pattern")
        return result
    
    def generate_table_of_contents(
        self, 
        tree: DocumentTree,
        max_level: Optional[int] = None,
        include_line_numbers: bool = True,
        include_content_stats: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate enhanced table of contents with configurable options.
        
        Args:
            tree: DocumentTree to analyze
            max_level: Maximum header level to include
            include_line_numbers: Whether to include line numbers
            include_content_stats: Whether to include content statistics
            
        Returns:
            List of TOC entries with requested metadata
        """
        toc = []
        
        for section in tree._sections:
            if not section.title:  # Skip sections without titles
                continue
            
            if max_level is not None and section.level > max_level:
                continue
            
            entry = {
                "title": section.title,
                "level": section.level,
            }
            
            if include_line_numbers:
                entry["line_number"] = section.line_number
            
            if include_content_stats:
                entry.update({
                    "content_length": len(section.content),
                    "children_count": len(section.children),
                    "depth": section.get_depth()
                })
            
            toc.append(entry)
        
        logger.debug(f"Generated TOC with {len(toc)} entries")
        return toc
    
    def extract_outline(self, tree: DocumentTree, max_depth: Optional[int] = None) -> str:
        """
        Generate a textual outline of the document structure.
        
        Args:
            tree: DocumentTree to outline
            max_depth: Maximum depth to include in outline
            
        Returns:
            String representation of document outline
        """
        if not tree.root:
            return "Empty document"
        
        lines = []
        
        def add_section_to_outline(section: DocumentSection, depth: int = 0) -> None:
            if max_depth is not None and depth > max_depth:
                return
            
            indent = "  " * depth
            level_indicator = "#" * section.level if section.level > 0 else "•"
            title = section.title or "(untitled)"
            
            lines.append(f"{indent}{level_indicator} {title}")
            
            for child in section.children:
                add_section_to_outline(child, depth + 1)
        
        add_section_to_outline(tree.root)
        
        return "\n".join(lines)
    
    def find_section_path(self, tree: DocumentTree, target_title: str) -> List[str]:
        """
        Find the path from root to a section with given title.
        
        Args:
            tree: DocumentTree to search
            target_title: Title of target section
            
        Returns:
            List of section titles from root to target (empty if not found)
        """
        target = tree.find_section_by_title(target_title)
        if not target:
            return []
        
        path = []
        current = target
        while current:
            if current.title:  # Only include titled sections in path
                path.insert(0, current.title)
            current = current.parent
        
        return path 