"""
Document Structure Module - Header-based Document Organization

This module implements document structure analysis and hierarchical organization
for markdown documents. It provides tools for splitting documents into logical
sections based on header hierarchy and building tree structures for navigation.

Key Components:
- DocumentSection: Represents a document section with header and content
- DocumentTree: Tree structure for document hierarchy
- HeaderBasedSplitter: Splits documents by headers into sections
- SectionExtractor: Extracts specific sections from document trees

Implements FR-KB-002.1: Hybrid chunking strategy with Markdown-aware processing.

Usage:
    >>> from .markdown_parser import MarkdownParser
    >>> parser = MarkdownParser()
    >>> splitter = HeaderBasedSplitter(parser)
    >>> tree = splitter.split_and_build_tree("# Main\\n\\nContent\\n\\n## Sub\\n\\nMore content")
    >>> print(tree.root.title)
    Main
"""

import re
from typing import List, Dict, Any, Optional, Callable
import logging

# Import from markdown_parser module
from .markdown_parser import MarkdownParser

logger = logging.getLogger(__name__)


class DocumentSection:
    """
    Represents a document section with header information and content.
    
    A section corresponds to a markdown header and all content until the next
    header of equal or higher level. Supports hierarchical organization with
    parent-child relationships for building document trees.
    
    This class is designed to preserve the semantic structure of markdown documents
    while enabling efficient traversal and content extraction operations.
    
    Attributes:
        level: Header level (1-6 for h1-h6, 0 for non-header content)
        title: Section title extracted from header text
        content: Text content of the section (excluding header line)
        line_number: Line number where the section starts in original document
        children: List of child sections (subsections)
        parent: Reference to parent section (None for root sections)
    
    Example:
        >>> section = DocumentSection(
        ...     level=1, 
        ...     title="Introduction", 
        ...     content="This is the intro.",
        ...     line_number=1
        ... )
        >>> child = DocumentSection(level=2, title="Background", content="...", line_number=5)
        >>> section.add_child(child)
        >>> print(section.children[0].title)
        Background
    """
    
    def __init__(
        self,
        level: int,
        title: str,
        content: str,
        line_number: int,
        parent: Optional['DocumentSection'] = None
    ) -> None:
        """
        Initialize a DocumentSection with validation and setup.
        
        Args:
            level: Header level (0-6, where 0 is for non-header content, 1-6 for h1-h6)
            title: Section title from header text (empty string for non-header content)
            content: Section content text (excluding the header line itself)
            line_number: Starting line number in original document (1-indexed)
            parent: Parent section reference (None for root sections)
            
        Raises:
            ValueError: If level is not in valid range 0-6
            TypeError: If any required parameter has wrong type
        """
        # Input validation
        if not isinstance(level, int) or level < 0 or level > 6:
            raise ValueError(f"Header level must be integer 0-6, got: {level}")
        if not isinstance(title, str):
            raise TypeError(f"Title must be string, got: {type(title)}")
        if not isinstance(content, str):
            raise TypeError(f"Content must be string, got: {type(content)}")
        if not isinstance(line_number, int) or line_number < 1:
            raise ValueError(f"Line number must be positive integer, got: {line_number}")
        
        self.level = level
        self.title = title
        self.content = content
        self.line_number = line_number
        self.parent = parent
        self.children: List['DocumentSection'] = []
        
        # Log section creation for debugging
        logger.debug(
            f"Created DocumentSection: level={level}, title='{title}', "
            f"content_length={len(content)}, line={line_number}"
        )
    
    def add_child(self, child: 'DocumentSection') -> None:
        """
        Add a child section to this section with validation.
        
        Establishes bidirectional parent-child relationship and validates
        that the child has appropriate header level (higher than parent).
        
        Args:
            child: Child section to add
            
        Raises:
            TypeError: If child is not a DocumentSection
            ValueError: If child level is not greater than parent level
        """
        if not isinstance(child, DocumentSection):
            raise TypeError(f"Child must be DocumentSection, got: {type(child)}")
        
        # Validate header level hierarchy (child should have higher level than parent)
        if self.level > 0 and child.level > 0 and child.level <= self.level:
            logger.warning(
                f"Child level {child.level} is not greater than parent level {self.level}. "
                f"This may indicate improper document structure."
            )
        
        child.parent = self
        self.children.append(child)
        
        logger.debug(f"Added child '{child.title}' to section '{self.title}'")
    
    def remove_child(self, child: 'DocumentSection') -> bool:
        """
        Remove a child section from this section.
        
        Args:
            child: Child section to remove
            
        Returns:
            True if child was found and removed, False otherwise
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            logger.debug(f"Removed child '{child.title}' from section '{self.title}'")
            return True
        return False
    
    def get_depth(self) -> int:
        """
        Calculate the depth of this section in the document hierarchy.
        
        Depth is determined by traversing parent relationships up to the root.
        Root sections have depth 0, their children have depth 1, etc.
        
        Returns:
            Depth level (0 for root, 1 for first level children, etc.)
            
        Example:
            >>> root = DocumentSection(1, "Root", "", 1)
            >>> child = DocumentSection(2, "Child", "", 3)
            >>> grandchild = DocumentSection(3, "Grandchild", "", 5)
            >>> root.add_child(child)
            >>> child.add_child(grandchild)
            >>> print(grandchild.get_depth())
            2
        """
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_all_content(self, include_headers: bool = False) -> str:
        """
        Get all content including content from child sections.
        
        Recursively collects content from this section and all its children,
        optionally including header text for better context.
        
        Args:
            include_headers: If True, include header titles in the output
            
        Returns:
            Combined content from this section and all children
            
        Example:
            >>> section = DocumentSection(1, "Main", "Main content", 1)
            >>> child = DocumentSection(2, "Sub", "Sub content", 3)
            >>> section.add_child(child)
            >>> print(section.get_all_content())
            Main content
            
            Sub content
        """
        content_parts = []
        
        # Add header if requested
        if include_headers and self.title:
            header_prefix = "#" * self.level if self.level > 0 else ""
            content_parts.append(f"{header_prefix} {self.title}" if header_prefix else self.title)
        
        # Add this section's content
        if self.content.strip():
            content_parts.append(self.content)
        
        # Recursively add children's content
        for child in self.children:
            child_content = child.get_all_content(include_headers)
            if child_content.strip():
                content_parts.append(child_content)
        
        return "\n\n".join(content_parts)
    
    def find_child_by_title(self, title: str, case_sensitive: bool = True) -> Optional['DocumentSection']:
        """
        Find a direct child section by title with optional case sensitivity.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Child section with matching title, or None if not found
            
        Example:
            >>> parent = DocumentSection(1, "Parent", "", 1)
            >>> child = DocumentSection(2, "Methods", "", 3)
            >>> parent.add_child(child)
            >>> found = parent.find_child_by_title("methods", case_sensitive=False)
            >>> print(found.title)
            Methods
        """
        for child in self.children:
            if case_sensitive:
                if child.title == title:
                    return child
            else:
                if child.title.lower() == title.lower():
                    return child
        return None
    
    def get_sibling_sections(self) -> List['DocumentSection']:
        """
        Get all sibling sections (sections with same parent).
        
        Returns:
            List of sibling sections (excluding self)
        """
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert section to dictionary for serialization.
        
        Returns:
            Dictionary representation with all section data
        """
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "line_number": self.line_number,
            "depth": self.get_depth(),
            "children_count": len(self.children),
            "children": [child.to_dict() for child in self.children]
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentSection(level={self.level}, title='{self.title}', "
            f"line={self.line_number}, children={len(self.children)})"
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on content and structure."""
        if not isinstance(other, DocumentSection):
            return NotImplemented
        return (
            self.level == other.level and
            self.title == other.title and
            self.content == other.content and
            self.line_number == other.line_number
        )


class DocumentTree:
    """
    Tree structure representing a complete document hierarchy.
    
    Manages the hierarchical organization of document sections and provides
    efficient methods for navigation, search, and manipulation of the document
    structure. Automatically maintains parent-child relationships and provides
    optimized lookup operations.
    
    The tree structure mirrors the logical organization of a markdown document,
    where headers create a natural hierarchy that can be traversed and analyzed.
    
    Attributes:
        root: Root section of the document (None for empty trees)
        _sections: Flat list of all sections for efficient O(1) lookup operations
        _title_index: Dictionary mapping titles to sections for fast title-based searches
    
    Example:
        >>> tree = DocumentTree()
        >>> tree.add_section(DocumentSection(1, "Chapter 1", "Content", 1))
        >>> tree.add_section(DocumentSection(2, "Section 1.1", "More content", 3))
        >>> print(tree.get_section_count())
        2
    """
    
    def __init__(self, root: Optional[DocumentSection] = None) -> None:
        """
        Initialize a DocumentTree with optional root section.
        
        Args:
            root: Optional root section to start the tree
        """
        self.root = root
        self._sections: List[DocumentSection] = []
        self._title_index: Dict[str, List[DocumentSection]] = {}
        
        if root:
            self._add_section_to_indexes(root)
        
        logger.debug(f"DocumentTree initialized with root: {root.title if root else 'None'}")
    
    def _add_section_to_indexes(self, section: DocumentSection) -> None:
        """
        Add section to internal indexes for efficient lookup.
        
        Args:
            section: Section to add to indexes
        """
        self._sections.append(section)
        
        # Update title index for fast title-based searches
        if section.title:
            if section.title not in self._title_index:
                self._title_index[section.title] = []
            self._title_index[section.title].append(section)
    
    def _remove_section_from_indexes(self, section: DocumentSection) -> None:
        """
        Remove section from internal indexes.
        
        Args:
            section: Section to remove from indexes
        """
        if section in self._sections:
            self._sections.remove(section)
        
        # Update title index
        if section.title in self._title_index:
            self._title_index[section.title] = [
                s for s in self._title_index[section.title] if s != section
            ]
            if not self._title_index[section.title]:
                del self._title_index[section.title]
    
    def add_section(self, section: DocumentSection) -> None:
        """
        Add a section to the tree, automatically determining proper hierarchy.
        
        Uses intelligent hierarchy detection based on header levels to place
        the section in the appropriate location within the tree structure.
        
        Args:
            section: Section to add to the tree
            
        Raises:
            TypeError: If section is not a DocumentSection
        """
        if not isinstance(section, DocumentSection):
            raise TypeError(f"Section must be DocumentSection, got: {type(section)}")
        
        self._add_section_to_indexes(section)
        
        if self.root is None:
            self.root = section
            logger.debug(f"Set '{section.title}' as root section")
            return
        
        # Find appropriate parent based on header levels and document order
        parent = self._find_parent_for_section(section)
        if parent:
            parent.add_child(section)
            logger.debug(f"Added '{section.title}' as child of '{parent.title}'")
        else:
            # Handle sections that should be siblings of root or new roots
            if section.level <= self.root.level:
                # For simplicity in this implementation, we keep the first root
                # In a more sophisticated version, we might create a virtual root
                logger.debug(f"'{section.title}' could be root-level, keeping as orphan")
            else:
                self.root.add_child(section)
                logger.debug(f"Added '{section.title}' as child of root '{self.root.title}'")
    
    def _find_parent_for_section(self, section: DocumentSection) -> Optional[DocumentSection]:
        """
        Find the appropriate parent for a section based on header levels and document order.
        
        Uses the most recent section with a lower header level as the parent,
        which matches the natural document hierarchy expectations.
        
        Args:
            section: Section needing a parent
            
        Returns:
            Appropriate parent section or None if no suitable parent found
        """
        # Find the most recent section with level < section.level
        for i in range(len(self._sections) - 1, -1, -1):
            candidate = self._sections[i]
            if candidate.level < section.level:
                return candidate
        return None
    
    def remove_section(self, section: DocumentSection) -> bool:
        """
        Remove a section from the tree.
        
        Args:
            section: Section to remove
            
        Returns:
            True if section was found and removed, False otherwise
        """
        if section not in self._sections:
            return False
        
        # Remove from parent if it has one
        if section.parent:
            section.parent.remove_child(section)
        
        # If this is the root, find a new root or set to None
        if section == self.root:
            self.root = self._sections[1] if len(self._sections) > 1 else None
        
        # Remove from indexes
        self._remove_section_from_indexes(section)
        
        logger.debug(f"Removed section '{section.title}' from tree")
        return True
    
    def get_section_count(self) -> int:
        """
        Get total number of sections in the tree.
        
        Returns:
            Total section count
        """
        return len(self._sections)
    
    def find_section_by_title(self, title: str, case_sensitive: bool = True) -> Optional[DocumentSection]:
        """
        Find a section by title with optimized lookup.
        
        Uses internal title index for O(1) average case performance.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            First section with matching title, or None if not found
        """
        if case_sensitive:
            sections = self._title_index.get(title, [])
            return sections[0] if sections else None
        else:
            # Fall back to linear search for case-insensitive
            for section in self._sections:
                if section.title.lower() == title.lower():
                    return section
            return None
    
    def find_all_sections_by_title(self, title: str, case_sensitive: bool = True) -> List[DocumentSection]:
        """
        Find all sections with a given title.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            List of sections with matching title
        """
        if case_sensitive:
            return self._title_index.get(title, []).copy()
        else:
            return [
                section for section in self._sections
                if section.title.lower() == title.lower()
            ]
    
    def get_sections_by_level(self, level: int) -> List[DocumentSection]:
        """
        Get all sections at a specific header level with validation.
        
        Args:
            level: Header level to filter by (0-6)
            
        Returns:
            List of sections at the specified level
            
        Raises:
            ValueError: If level is not in valid range
        """
        if not isinstance(level, int) or level < 0 or level > 6:
            raise ValueError(f"Level must be integer 0-6, got: {level}")
        
        return [section for section in self._sections if section.level == level]
    
    def get_sections_by_level_range(self, min_level: int, max_level: int) -> List[DocumentSection]:
        """
        Get sections within a level range.
        
        Args:
            min_level: Minimum level (inclusive)
            max_level: Maximum level (inclusive)
            
        Returns:
            List of sections within the level range
        """
        return [
            section for section in self._sections
            if min_level <= section.level <= max_level
        ]
    
    def traverse_breadth_first(self) -> List[DocumentSection]:
        """
        Traverse the tree in breadth-first order.
        
        Returns:
            List of sections in breadth-first order
        """
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            section = queue.pop(0)
            result.append(section)
            queue.extend(section.children)
        
        return result
    
    def traverse_depth_first(self) -> List[DocumentSection]:
        """
        Traverse the tree in depth-first order.
        
        Returns:
            List of sections in depth-first order
        """
        if not self.root:
            return []
        
        result = []
        
        def dfs(section: DocumentSection) -> None:
            result.append(section)
            for child in section.children:
                dfs(child)
        
        dfs(self.root)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tree to dictionary for serialization with metadata.
        
        Returns:
            Dictionary representation with tree structure and metadata
        """
        return {
            "metadata": {
                "total_sections": len(self._sections),
                "max_depth": max(section.get_depth() for section in self._sections) if self._sections else 0,
                "levels_present": list(set(section.level for section in self._sections)),
                "has_root": self.root is not None
            },
            "root": self.root.to_dict() if self.root else None
        }
    
    def get_table_of_contents(self, max_level: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate table of contents with optional level filtering.
        
        Args:
            max_level: Maximum header level to include (None for all levels)
            
        Returns:
            List of TOC entries with enhanced metadata
        """
        toc = []
        for section in self._sections:
            if section.title and (max_level is None or section.level <= max_level):
                toc.append({
                    "title": section.title,
                    "level": section.level,
                    "line_number": section.line_number,
                    "depth": section.get_depth(),
                    "children_count": len(section.children),
                    "content_length": len(section.content)
                })
        return toc
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentTree(sections={len(self._sections)}, "
            f"root='{self.root.title if self.root else None}')"
        )


class HeaderBasedSplitter:
    """
    High-performance document splitter for markdown header-based sectioning.
    
    Provides efficient algorithms for parsing markdown documents and extracting
    logical sections based on header hierarchy. Uses optimized regex patterns
    and intelligent content extraction to build accurate document representations.
    
    The splitter handles various edge cases including documents without headers,
    nested header hierarchies, and mixed content types while preserving the
    semantic structure of the original document.
    
    Attributes:
        parser: MarkdownParser instance for header detection
        _header_pattern: Compiled regex for header matching (performance optimization)
    
    Example:
        >>> parser = MarkdownParser()
        >>> splitter = HeaderBasedSplitter(parser)
        >>> tree = splitter.split_and_build_tree("# Main\\n\\nContent\\n\\n## Sub\\n\\nMore")
        >>> print(tree.get_section_count())
        2
    """
    
    def __init__(self, parser: MarkdownParser) -> None:
        """
        Initialize HeaderBasedSplitter with parser and optimization setup.
        
        Args:
            parser: MarkdownParser instance to use for header detection
            
        Raises:
            TypeError: If parser is not a MarkdownParser instance
        """
        if not isinstance(parser, MarkdownParser):
            raise TypeError(f"Parser must be MarkdownParser, got: {type(parser)}")
        
        self.parser = parser
        # Pre-compile regex pattern for performance
        self._header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        logger.debug("HeaderBasedSplitter initialized with MarkdownParser")
    
    def split_by_headers(self, document: str, preserve_whitespace: bool = True) -> List[DocumentSection]:
        """
        Split a document into sections based on headers with enhanced processing.
        
        Parses the document line by line, identifying headers and extracting
        content between them. Handles edge cases and provides options for
        whitespace preservation.
        
        Args:
            document: Markdown document text
            preserve_whitespace: Whether to preserve original whitespace formatting
            
        Returns:
            List of DocumentSection objects in document order
            
        Raises:
            TypeError: If document is not a string
            
        Example:
            >>> splitter = HeaderBasedSplitter(MarkdownParser())
            >>> doc = "# Main\\n\\nContent\\n\\n## Sub\\n\\nMore content"
            >>> sections = splitter.split_by_headers(doc)
            >>> print(len(sections))
            2
        """
        if not isinstance(document, str):
            raise TypeError(f"Document must be string, got: {type(document)}")
        
        if not document.strip():
            logger.debug("Empty document provided, returning empty list")
            return []
        
        lines = document.split('\n')
        sections = []
        current_section = None
        content_lines = []
        
        logger.debug(f"Processing document with {len(lines)} lines")
        
        for line_num, line in enumerate(lines, 1):
            # Check if line is a header using pre-compiled regex
            header_match = self._header_pattern.match(line.strip())
            
            if header_match:
                # Save previous section if exists
                if current_section is not None:
                    content = self._process_content_lines(content_lines, preserve_whitespace)
                    current_section.content = content
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = DocumentSection(
                    level=level,
                    title=title,
                    content="",
                    line_number=line_num
                )
                content_lines = []
                
                logger.debug(f"Found header: level={level}, title='{title}', line={line_num}")
            else:
                content_lines.append(line)
        
        # Handle final section or content without headers
        if current_section is not None:
            content = self._process_content_lines(content_lines, preserve_whitespace)
            current_section.content = content
            sections.append(current_section)
        elif content_lines and any(line.strip() for line in content_lines):
            # Document has content but no headers
            content = self._process_content_lines(content_lines, preserve_whitespace)
            sections.append(DocumentSection(
                level=0,
                title="",
                content=content,
                line_number=1
            ))
            logger.debug("Created section for headerless content")
        
        logger.debug(f"Split document into {len(sections)} sections")
        return sections
    
    def _process_content_lines(self, lines: List[str], preserve_whitespace: bool) -> str:
        """
        Process content lines with whitespace handling options.
        
        Args:
            lines: List of content lines
            preserve_whitespace: Whether to preserve whitespace
            
        Returns:
            Processed content string
        """
        if preserve_whitespace:
            return '\n'.join(lines).strip()
        else:
            # Remove excessive whitespace while preserving paragraph breaks
            content = '\n'.join(lines).strip()
            # Normalize multiple newlines to double newlines (paragraph breaks)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            return content
    
    def build_tree(self, sections: List[DocumentSection]) -> DocumentTree:
        """
        Build a DocumentTree from a flat list of sections with validation.
        
        Args:
            sections: List of DocumentSection objects in document order
            
        Returns:
            DocumentTree with proper hierarchy established
            
        Raises:
            TypeError: If sections is not a list or contains non-DocumentSection items
        """
        if not isinstance(sections, list):
            raise TypeError(f"Sections must be list, got: {type(sections)}")
        
        for i, section in enumerate(sections):
            if not isinstance(section, DocumentSection):
                raise TypeError(f"Section {i} is not DocumentSection: {type(section)}")
        
        tree = DocumentTree()
        
        logger.debug(f"Building tree from {len(sections)} sections")
        
        for section in sections:
            tree.add_section(section)
        
        logger.debug(f"Built tree with {tree.get_section_count()} sections")
        return tree
    
    def split_and_build_tree(
        self, 
        document: str, 
        preserve_whitespace: bool = True
    ) -> DocumentTree:
        """
        Split document into sections and build tree in one optimized operation.
        
        Combines splitting and tree building with shared validation and
        performance optimizations for common use cases.
        
        Args:
            document: Markdown document text
            preserve_whitespace: Whether to preserve original whitespace
            
        Returns:
            DocumentTree representing the complete document structure
        """
        sections = self.split_by_headers(document, preserve_whitespace)
        return self.build_tree(sections)
    
    def analyze_document_structure(self, document: str) -> Dict[str, Any]:
        """
        Analyze document structure and provide detailed metrics.
        
        Args:
            document: Markdown document text
            
        Returns:
            Dictionary with structural analysis data
        """
        sections = self.split_by_headers(document)
        tree = self.build_tree(sections)
        
        if not sections:
            return {"structure": "empty", "sections": 0}
        
        levels = [section.level for section in sections if section.level > 0]
        content_lengths = [len(section.content) for section in sections]
        
        return {
            "structure": "hierarchical" if len(set(levels)) > 1 else "flat",
            "total_sections": len(sections),
            "header_sections": len([s for s in sections if s.level > 0]),
            "headerless_sections": len([s for s in sections if s.level == 0]),
            "max_depth": max([s.get_depth() for s in sections]) if sections else 0,
            "levels_used": sorted(set(levels)) if levels else [],
            "avg_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            "total_content_length": sum(content_lengths),
            "table_of_contents": tree.get_table_of_contents()
        }


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