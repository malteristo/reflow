"""
Query validation hooks for model compatibility checking.

This module provides query validation functionality to ensure that queries
are only executed against collections that are compatible with the current
embedding model. Prevents inconsistent results from model mismatches.

Part of Task 35 - Model Change Detection Integration Phase 3.
"""

import logging
from typing import List, Optional, Dict, Any, Set

from .detector import ModelChangeDetector
from .fingerprint import ModelFingerprint
from .types import ModelChangeError

logger = logging.getLogger(__name__)


class QueryValidationError(ModelChangeError):
    """Raised when query validation fails due to model incompatibility."""
    pass


class ModelCompatibilityValidator:
    """
    Validates model compatibility for query operations.
    
    Provides hooks to check that queries are only executed against collections
    that were indexed with compatible embedding models.
    
    Features:
        - Model fingerprint compatibility checking
        - Collection compatibility validation
        - Query blocking for incompatible collections
        - Reindexing recommendations
    """
    
    def __init__(self):
        """Initialize the model compatibility validator."""
        self._detector = ModelChangeDetector()
        self._strict_mode = True  # Default to strict compatibility checking
        self._warned_collections: Set[str] = set()  # Track collections we've warned about
        
        logger.debug("Model compatibility validator initialized")
    
    def set_strict_mode(self, strict: bool) -> None:
        """
        Set strict mode for compatibility checking.
        
        Args:
            strict: If True, raise errors for incompatible models.
                   If False, only log warnings.
        """
        self._strict_mode = strict
        logger.debug(f"Model compatibility strict mode set to: {strict}")
    
    def validate_query_compatibility(
        self, 
        collections: List[str], 
        current_embedding_service,
        collection_metadata_getter: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Validate that the current embedding model is compatible with target collections.
        
        Args:
            collections: List of collection names to query
            current_embedding_service: Current embedding service instance
            collection_metadata_getter: Optional function to get collection metadata
                                       Should return dict with 'embedding_model_fingerprint'
            
        Returns:
            Dictionary with validation results:
            - compatible_collections: List of compatible collection names
            - incompatible_collections: List of incompatible collection names
            - warnings: List of warning messages
            - blocked: Boolean indicating if query should be blocked
            
        Raises:
            QueryValidationError: If in strict mode and incompatible collections found
            
        Example:
            >>> validator = ModelCompatibilityValidator()
            >>> def get_metadata(collection_name):
            ...     return {"embedding_model_fingerprint": "abc123"}
            >>> result = validator.validate_query_compatibility(
            ...     ["collection1", "collection2"], 
            ...     embedding_service,
            ...     get_metadata
            ... )
        """
        try:
            # Generate current model fingerprint
            current_fingerprint = current_embedding_service.generate_model_fingerprint()
            
            compatible_collections = []
            incompatible_collections = []
            warnings = []
            
            for collection_name in collections:
                try:
                    # Get collection metadata if getter provided
                    if collection_metadata_getter:
                        metadata = collection_metadata_getter(collection_name)
                        if metadata and 'embedding_model_fingerprint' in metadata:
                            collection_fingerprint = metadata['embedding_model_fingerprint']
                            
                            # Check compatibility using fingerprint checksums
                            if self._is_fingerprint_compatible(current_fingerprint, collection_fingerprint):
                                compatible_collections.append(collection_name)
                            else:
                                incompatible_collections.append(collection_name)
                                if collection_name not in self._warned_collections:
                                    warnings.append(
                                        f"Collection '{collection_name}' was indexed with a different "
                                        f"embedding model. Results may be inconsistent."
                                    )
                                    self._warned_collections.add(collection_name)
                        else:
                            # No fingerprint metadata - assume compatible but warn
                            compatible_collections.append(collection_name)
                            if collection_name not in self._warned_collections:
                                warnings.append(
                                    f"Collection '{collection_name}' has no model fingerprint metadata. "
                                    f"Cannot verify compatibility."
                                )
                                self._warned_collections.add(collection_name)
                    else:
                        # No metadata getter - assume compatible
                        compatible_collections.append(collection_name)
                        
                except Exception as e:
                    logger.warning(f"Failed to check compatibility for collection '{collection_name}': {e}")
                    # Add to compatible collections but with warning
                    compatible_collections.append(collection_name)
                    warnings.append(f"Could not verify compatibility for collection '{collection_name}': {e}")
            
            # Determine if query should be blocked
            blocked = len(incompatible_collections) > 0 and self._strict_mode
            
            # Log warnings
            for warning in warnings:
                logger.warning(warning)
            
            # In strict mode, raise error if incompatible collections found
            if self._strict_mode and incompatible_collections:
                error_msg = (
                    f"Query blocked due to model incompatibility. "
                    f"Incompatible collections: {incompatible_collections}. "
                    f"Please reindex these collections with the current model."
                )
                raise QueryValidationError(error_msg)
            
            result = {
                'compatible_collections': compatible_collections,
                'incompatible_collections': incompatible_collections,
                'warnings': warnings,
                'blocked': blocked,
                'current_model': current_fingerprint.model_name,
                'current_fingerprint': current_fingerprint.checksum
            }
            
            logger.debug(
                f"Query compatibility check: {len(compatible_collections)} compatible, "
                f"{len(incompatible_collections)} incompatible collections"
            )
            
            return result
            
        except QueryValidationError:
            # Re-raise query validation errors
            raise
        except Exception as e:
            logger.error(f"Query compatibility validation failed: {e}")
            # In case of errors, allow query but log warning
            return {
                'compatible_collections': collections,
                'incompatible_collections': [],
                'warnings': [f"Compatibility check failed: {e}"],
                'blocked': False,
                'current_model': 'unknown',
                'current_fingerprint': 'unknown'
            }
    
    def _is_fingerprint_compatible(self, current_fingerprint: ModelFingerprint, stored_fingerprint: str) -> bool:
        """
        Check if the current model fingerprint is compatible with a stored fingerprint.
        
        Args:
            current_fingerprint: Current model fingerprint object
            stored_fingerprint: Stored fingerprint checksum string
            
        Returns:
            True if compatible, False otherwise
        """
        # Simple checksum comparison for now
        # In the future, could implement more sophisticated compatibility rules
        return current_fingerprint.checksum == stored_fingerprint
    
    def create_query_validation_decorator(self, get_collections_metadata: callable):
        """
        Create a decorator for query methods that automatically validates model compatibility.
        
        Args:
            get_collections_metadata: Function to retrieve collection metadata
            
        Returns:
            Decorator function for query methods
            
        Example:
            >>> validator = ModelCompatibilityValidator()
            >>> validate_query = validator.create_query_validation_decorator(get_metadata)
            >>> 
            >>> class QueryService:
            ...     @validate_query
            ...     def query_collections(self, query, collections, embedding_service):
            ...         # Query implementation
            ...         pass
        """
        def decorator(query_method):
            def wrapper(*args, **kwargs):
                # Extract collections and embedding service from arguments
                # This is a simplified implementation - real decorator would need
                # more sophisticated argument parsing
                collections = kwargs.get('collections', [])
                embedding_service = kwargs.get('embedding_service')
                
                if collections and embedding_service:
                    # Validate compatibility
                    self.validate_query_compatibility(
                        collections, 
                        embedding_service, 
                        get_collections_metadata
                    )
                
                # Execute original query method
                return query_method(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_reindexing_recommendations(
        self, 
        incompatible_collections: List[str],
        current_embedding_service
    ) -> Dict[str, Any]:
        """
        Get recommendations for reindexing incompatible collections.
        
        Args:
            incompatible_collections: List of incompatible collection names
            current_embedding_service: Current embedding service
            
        Returns:
            Dictionary with reindexing recommendations
        """
        if not incompatible_collections:
            return {
                'needs_reindexing': False,
                'collections': [],
                'recommendations': []
            }
        
        current_fingerprint = current_embedding_service.generate_model_fingerprint()
        
        recommendations = []
        for collection_name in incompatible_collections:
            recommendations.append({
                'collection': collection_name,
                'action': 'reindex',
                'reason': 'Model fingerprint mismatch',
                'target_model': current_fingerprint.model_name,
                'priority': 'high'
            })
        
        return {
            'needs_reindexing': True,
            'collections': incompatible_collections,
            'recommendations': recommendations,
            'current_model': current_fingerprint.model_name,
            'total_collections': len(incompatible_collections)
        }


# Global instance for easy access
_global_validator_instance: Optional[ModelCompatibilityValidator] = None


def get_compatibility_validator() -> ModelCompatibilityValidator:
    """
    Get the global model compatibility validator instance.
    
    Returns:
        Global ModelCompatibilityValidator instance (singleton pattern)
    """
    global _global_validator_instance
    
    if _global_validator_instance is None:
        _global_validator_instance = ModelCompatibilityValidator()
    
    return _global_validator_instance


def validate_query_compatibility(
    collections: List[str], 
    embedding_service,
    metadata_getter: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate query compatibility.
    
    Args:
        collections: List of collection names to query
        embedding_service: Current embedding service
        metadata_getter: Optional function to get collection metadata
        
    Returns:
        Validation results dictionary
    """
    validator = get_compatibility_validator()
    return validator.validate_query_compatibility(collections, embedding_service, metadata_getter)


def set_compatibility_strict_mode(strict: bool) -> None:
    """
    Convenience function to set strict mode for compatibility checking.
    
    Args:
        strict: Whether to enable strict mode
    """
    validator = get_compatibility_validator()
    validator.set_strict_mode(strict) 