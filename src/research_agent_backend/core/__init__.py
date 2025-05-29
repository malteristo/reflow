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

from .embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)

from .local_embedding_service import (
    LocalEmbeddingService,
    ModelCacheManager,
    EmbeddingModelConfig,
)

__all__ = [
    "ChromaDBManager",
    "create_chroma_manager", 
    "get_default_collection_types",
    "EmbeddingService",
    "EmbeddingServiceError",
    "ModelNotFoundError", 
    "EmbeddingDimensionError",
    "BatchProcessingError",
    "LocalEmbeddingService",
    "ModelCacheManager",
    "EmbeddingModelConfig",
]
