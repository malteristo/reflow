"""
Document Structure Module - Header-based Document Organization

This module provides a compatibility layer for the document structure functionality
that has been refactored into a modular package structure.

All classes and functionality have been moved to focused modules in the structure/
subdirectory while maintaining 100% backward compatibility.

Key Components (imported from modular structure):
- DocumentSection: Individual section representation
- DocumentTree: Tree structure for document hierarchy  
- HeaderBasedSplitter: Document parsing and section extraction
- SectionExtractor: Advanced section filtering and analysis

Implements FR-KB-002.1: Hybrid chunking strategy with Markdown-aware processing.

Usage:
    >>> from .markdown_parser import MarkdownParser
    >>> parser = MarkdownParser()
    >>> splitter = HeaderBasedSplitter(parser)
    >>> tree = splitter.split_and_build_tree("# Main\\n\\nContent\\n\\n## Sub\\n\\nMore content")
    >>> print(tree.root.title)
    Main
"""

# Import all components from the modular structure
from .structure.section import DocumentSection
from .structure.tree import DocumentTree  
from .structure.splitter import HeaderBasedSplitter
from .structure.extractor import SectionExtractor

# Maintain complete public API for backward compatibility
__all__ = [
    'DocumentSection',
    'DocumentTree', 
    'HeaderBasedSplitter',
    'SectionExtractor'
] 