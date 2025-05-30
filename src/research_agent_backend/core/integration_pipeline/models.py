"""
Data structures and result objects for integration pipeline operations.

This module provides data models used throughout the integration pipeline
for representing processing results, search results, and storage outcomes.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProcessingResult:
    """Result from document processing pipeline."""
    document_id: str
    status: str
    chunks_created: int
    embeddings_generated: int
    chunking_strategy: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from integrated search engine."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class StorageResult:
    """Enhanced storage result with detailed metrics."""
    success: bool
    documents_added: int
    processing_time: float = 0.0
    storage_size_bytes: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Collection:
    """Enhanced collection with metadata and statistics."""
    name: str
    config: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    document_count: int = 0
    size_bytes: int = 0


class MockChunk:
    """Enhanced mock chunk with realistic metadata."""
    
    def __init__(self, document_id: str, chunk_index: int = 0, embedding_dim: int = 384):
        """
        Initialize mock chunk with metadata.
        
        Args:
            document_id: Identifier for the source document
            chunk_index: Index of this chunk within the document
            embedding_dim: Dimension of the embedding vector
        """
        self.metadata = type('MockMetadata', (), {
            'document_id': document_id,
            'chunk_index': chunk_index,
            'created_at': time.time(),
            'size': embedding_dim
        })()
        self.embeddings = [0.1 + (chunk_index * 0.01)] * embedding_dim
        self.content = f"Mock chunk {chunk_index} content for document {document_id}" 