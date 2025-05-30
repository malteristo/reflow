"""
Atomic Units System

This module provides atomic content unit detection and processing for markdown documents.
Atomic units represent the smallest meaningful content segments that should be preserved
during document chunking operations.

Components:
- AtomicUnitType: Enumeration of supported atomic unit types
- AtomicUnit: Data structure representing an atomic content unit
- AtomicUnitHandler: Main detection and management system
- Specialized handlers for each unit type
- AtomicUnitRegistry: Registry system for pluggable handlers
"""

from .types import AtomicUnitType, AtomicUnit
from .handlers import (
    CodeBlockHandler,
    TableHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler
)
from .registry import AtomicUnitRegistry
from .handler import AtomicUnitHandler

__all__ = [
    "AtomicUnitType",
    "AtomicUnit", 
    "AtomicUnitHandler",
    "CodeBlockHandler",
    "TableHandler",
    "ListHandler", 
    "BlockquoteHandler",
    "ParagraphHandler",
    "AtomicUnitRegistry"
] 