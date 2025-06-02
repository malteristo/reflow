"""
Chunking Package - Advanced Document Chunking Components

This package implements sophisticated document chunking functionality with intelligent
boundary detection, overlap management, and content-aware processing.

Implements FR-KB-002.1: Hybrid chunking strategy with content preservation.

Components:
- BoundaryStrategy: Enumeration of boundary detection strategies
- ChunkingMetrics: Protocol for metrics collection
- ChunkConfig: Advanced configuration for chunking parameters
- ChunkResult: Comprehensive chunk result container
- ChunkBoundary: Intelligent boundary detection system
- RecursiveChunker: Main recursive chunking engine
- HybridChunker: Complete FR-KB-002.1 hybrid chunking implementation
- HybridChunkResult: Result container for hybrid chunking operations
"""

from .config import (
    BoundaryStrategy,
    ChunkingMetrics,
    ChunkConfig
)

from .result import ChunkResult

from .boundary import ChunkBoundary

from .chunker import RecursiveChunker

from .hybrid_chunker import HybridChunker, HybridChunkResult

# Public API - maintains backward compatibility
__all__ = [
    "BoundaryStrategy",
    "ChunkingMetrics", 
    "ChunkConfig",
    "ChunkResult",
    "ChunkBoundary",
    "RecursiveChunker",
    "HybridChunker",
    "HybridChunkResult"
] 