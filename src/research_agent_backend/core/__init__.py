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

from .api_embedding_service import (
    APIEmbeddingService,
    APIConfiguration,
    APIError,
    RateLimitError,
    AuthenticationError,
)

from .model_change_detection import (
    ModelFingerprint,
    ModelChangeDetector,
    ModelChangeEvent,
    ModelChangeError,
    FingerprintMismatchError,
    PersistenceError,
    # Configuration integration
    ConfigurationIntegrationHooks,
    auto_register_embedding_service,
    add_config_change_callback,
    trigger_config_change,
    # Query validation
    ModelCompatibilityValidator,
    QueryValidationError,
    validate_query_compatibility,
    set_compatibility_strict_mode,
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
    "APIEmbeddingService",
    "APIConfiguration",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "ModelFingerprint",
    "ModelChangeDetector", 
    "ModelChangeEvent",
    "ModelChangeError",
    "FingerprintMismatchError",
    "PersistenceError",
    # Configuration integration
    "ConfigurationIntegrationHooks",
    "auto_register_embedding_service",
    "add_config_change_callback",
    "trigger_config_change",
    # Query validation
    "ModelCompatibilityValidator",
    "QueryValidationError",
    "validate_query_compatibility",
    "set_compatibility_strict_mode",
]
