"""
Enumeration types for metadata schema classification.

This module defines all enumeration types used throughout the metadata schema
for consistent type classification and validation.
"""

from enum import Enum


class DocumentType(Enum):
    """Document type classification for metadata."""
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    CODE = "code"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class ContentType(Enum):
    """Content type classification for chunks."""
    PROSE = "prose"
    CODE_BLOCK = "code-block"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    METADATA_BLOCK = "metadata-block"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class CollectionType(Enum):
    """Collection type classification."""
    FUNDAMENTAL = "fundamental"
    PROJECT_SPECIFIC = "project-specific"
    GENERAL = "general"
    REFERENCE = "reference"
    TEMPORARY = "temporary"
    
    def __str__(self) -> str:
        return self.value


class AccessPermission(Enum):
    """Access permission levels for future team features."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"
    
    def __str__(self) -> str:
        return self.value 