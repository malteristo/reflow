"""
Metadata Extraction Module

This module provides comprehensive metadata extraction from markdown documents,
including frontmatter parsing, inline metadata detection, and metadata registry
management.

Components:
- frontmatter: YAML/TOML frontmatter parsing
- inline: Inline metadata and tag extraction  
- registry: Metadata storage and querying
- types: Data structures and type definitions

Implements FR-KB-003: Metadata extraction and management.
"""

from .frontmatter import (
    FrontmatterParser,
    FrontmatterResult,
    FrontmatterParseError
)

from .inline import (
    InlineMetadataExtractor,
    InlineMetadataResult,
    InlineMetadataItem,
    InlineTag
)

from .registry import (
    MetadataRegistry,
    MetadataQuery
)

from .types import (
    DocumentMetadata,
    MetadataExtractionResult,
    MetadataExtractor
)

__all__ = [
    # Frontmatter
    'FrontmatterParser',
    'FrontmatterResult', 
    'FrontmatterParseError',
    
    # Inline metadata
    'InlineMetadataExtractor',
    'InlineMetadataResult',
    'InlineMetadataItem',
    'InlineTag',
    
    # Registry and querying
    'MetadataRegistry',
    'MetadataQuery',
    
    # Core types
    'DocumentMetadata',
    'MetadataExtractionResult',
    'MetadataExtractor'
] 