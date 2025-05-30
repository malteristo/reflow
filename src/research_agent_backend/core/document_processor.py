"""
Document Processing Module - Legacy Compatibility Layer

This module provides backward compatibility by re-exporting all functionality
from the modular document_processor package structure. All actual implementation
has been moved to the document_processor/ package for better organization.

For new code, import directly from the specific modules:
- from document_processor.markdown_parser import MarkdownParser
- from document_processor.document_structure import DocumentTree
- from document_processor.chunking import ChunkConfig, RecursiveChunker
- from document_processor.atomic_units import AtomicUnitHandler
- from document_processor.metadata import MetadataExtractor

This file maintains compatibility for existing imports.
"""

# Re-export all functionality from modular structure
from .document_processor import *

# Explicit imports for main classes to ensure availability
from .document_processor import (
    # Markdown parsing
    MarkdownParseError,
    MatchResult,
    Pattern,
    Rule,
    MarkdownParser,
    
    # Document structure
    DocumentSection,
    DocumentTree,
    HeaderBasedSplitter,
    SectionExtractor,
    
    # Chunking system
    ChunkConfig,
    ChunkResult,
    ChunkBoundary,
    RecursiveChunker,
    BoundaryStrategy,
    
    # Atomic units
    AtomicUnit,
    AtomicUnitType,
    AtomicUnitHandler,
    AtomicUnitRegistry,
    CodeBlockHandler,
    TableHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler,
    
    # Metadata extraction
    MetadataExtractor,
    FrontmatterParser,
    InlineMetadataExtractor,
    MetadataRegistry,
)

# Legacy compatibility - ensure all classes are available at module level
__all__ = [
    # Markdown parsing
    'MarkdownParseError',
    'MatchResult', 
    'Pattern',
    'Rule',
    'MarkdownParser',
    
    # Document structure
    'DocumentSection',
    'DocumentTree', 
    'HeaderBasedSplitter',
    'SectionExtractor',
    
    # Chunking system
    'ChunkConfig',
    'ChunkResult',
    'ChunkBoundary', 
    'RecursiveChunker',
    'BoundaryStrategy',
    
    # Atomic units
    'AtomicUnit',
    'AtomicUnitType',
    'AtomicUnitHandler',
    'AtomicUnitRegistry',
    'CodeBlockHandler',
    'TableHandler',
    'ListHandler', 
    'BlockquoteHandler',
    'ParagraphHandler',
    
    # Metadata extraction
    'MetadataExtractor',
    'FrontmatterParser',
    'InlineMetadataExtractor', 
    'MetadataRegistry',
]

