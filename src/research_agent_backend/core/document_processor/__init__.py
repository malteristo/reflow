"""
Document Processor Package

Advanced document processing system with modular architecture following adaptive
organizational principles. Provides comprehensive document parsing, chunking,
atomic unit detection, and metadata extraction capabilities.

Architecture (following FR-KB-002 requirements):
- Markdown-aware parsing with structured content preservation
- Intelligent chunking with boundary detection and overlap management  
- Atomic unit system for granular content handling
- Comprehensive metadata extraction and management
- Backward compatibility maintained through package imports

Components:
- markdown_parser: Core markdown parsing (Pattern, Rule, MarkdownParser)
- document_structure: Document sections and tree building
- chunking/: Advanced chunking engine with boundary detection
- atomic_units/: Content type handlers and registry system
- metadata/: Frontmatter and inline metadata extraction
"""

# Core parsing components
from .markdown_parser import (
    MarkdownParseError,
    MatchResult, 
    Pattern,
    Rule,
    MarkdownParser
)

# Document structure components  
from .document_structure import (
    DocumentSection,
    DocumentTree,
    HeaderBasedSplitter,
    SectionExtractor
)

# Chunking engine components (fully modularized)
from .chunking import (
    BoundaryStrategy,
    ChunkingMetrics,
    ChunkConfig,
    ChunkResult,
    ChunkBoundary,
    RecursiveChunker
)

# Atomic units system
from .atomic_units import (
    AtomicUnitType,
    AtomicUnit,
    AtomicUnitHandler,
    CodeBlockHandler,
    TableHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler,
    AtomicUnitRegistry
)

# Metadata extraction system (fully modularized)
from .metadata import (
    # Frontmatter components
    FrontmatterParser,
    FrontmatterResult,
    FrontmatterParseError,
    
    # Inline metadata
    InlineMetadataExtractor,
    InlineMetadataResult,
    InlineMetadataItem,
    InlineTag,
    
    # Registry and querying
    MetadataRegistry,
    MetadataQuery,
    
    # Core types
    DocumentMetadata,
    MetadataExtractionResult,
    MetadataExtractor
)

# Public API - maintains 100% backward compatibility
__all__ = [
    # Core parsing
    "MarkdownParseError",
    "MatchResult",
    "Pattern", 
    "Rule",
    "MarkdownParser",
    
    # Document structure
    "DocumentSection",
    "DocumentTree", 
    "HeaderBasedSplitter",
    "SectionExtractor",
    
    # Chunking engine
    "BoundaryStrategy",
    "ChunkingMetrics",
    "ChunkConfig",
    "ChunkResult", 
    "ChunkBoundary",
    "RecursiveChunker",
    
    # Atomic units
    "AtomicUnitType",
    "AtomicUnit",
    "AtomicUnitHandler",
    "CodeBlockHandler",
    "TableHandler", 
    "ListHandler",
    "BlockquoteHandler",
    "ParagraphHandler",
    "AtomicUnitRegistry",
    
    # Metadata extraction
    "FrontmatterParser",
    "FrontmatterResult",
    "FrontmatterParseError",
    "InlineMetadataExtractor",
    "InlineMetadataResult", 
    "InlineMetadataItem",
    "InlineTag",
    "MetadataRegistry",
    "MetadataQuery",
    "DocumentMetadata",
    "MetadataExtractionResult",
    "MetadataExtractor"
] 