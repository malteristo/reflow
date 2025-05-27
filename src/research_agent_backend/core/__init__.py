"""
Core modules for Research Agent backend.

This package contains the core business logic components including
vector database management, query processing, and document handling.
"""

from .vector_store import (
    ChromaDBManager,
    create_chroma_manager,
    get_default_collection_types,
)

__all__ = [
    "ChromaDBManager",
    "create_chroma_manager", 
    "get_default_collection_types",
]
