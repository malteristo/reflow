"""
Header-Based Splitter Module - Document Parsing and Section Extraction

This module provides the HeaderBasedSplitter class for parsing markdown documents
and extracting logical sections based on header hierarchy.

Key Components:
- HeaderBasedSplitter: Splits documents by headers into sections

Usage:
    >>> from ..markdown_parser import MarkdownParser
    >>> from .section import DocumentSection
    >>> from .tree import DocumentTree
    >>> parser = MarkdownParser()
    >>> splitter = HeaderBasedSplitter(parser)
    >>> tree = splitter.split_and_build_tree("# Main\\n\\nContent\\n\\n## Sub\\n\\nMore")
"""

import re
from typing import List, Dict, Any
import logging

from ..markdown_parser import MarkdownParser
from .section import DocumentSection
from .tree import DocumentTree

logger = logging.getLogger(__name__)


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
        Convenience method to split document and build tree in one call.
        
        Args:
            document: Markdown document text
            preserve_whitespace: Whether to preserve original whitespace formatting
            
        Returns:
            DocumentTree with document hierarchy
            
        Example:
            >>> splitter = HeaderBasedSplitter(MarkdownParser())
            >>> tree = splitter.split_and_build_tree("# Title\\n\\nContent")
            >>> print(tree.root.title)
            Title
        """
        sections = self.split_by_headers(document, preserve_whitespace)
        return self.build_tree(sections)
    
    def analyze_structure(self, document: str) -> Dict[str, Any]:
        """
        Analyze document structure and provide detailed statistics.
        
        Args:
            document: Markdown document text
            
        Returns:
            Dictionary with structure analysis results including section counts,
            hierarchy information, and content statistics
        """
        sections = self.split_by_headers(document)
        tree = self.build_tree(sections)
        
        levels = [s.level for s in sections if s.level > 0]
        content_lengths = [len(s.content) for s in sections]
        
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