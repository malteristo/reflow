"""
API embedding service model integration.

This module provides model change detection and fingerprinting capabilities
for API-based embedding services, enabling cache invalidation and model lifecycle management.
"""

import hashlib
import logging
from typing import Dict, Any, Optional

from .config import APIConfiguration

logger = logging.getLogger(__name__)


class ModelIntegration:
    """
    Model integration handler for API embedding services.
    
    Provides model change detection and fingerprinting capabilities to support
    cache invalidation and model lifecycle management. Integrates with the
    model change detection system to track configuration changes.
    
    Features:
        - Model fingerprint generation based on configuration
        - Change detection integration
        - Cache invalidation triggers
        - Model metadata collection
    """
    
    def __init__(self, config: APIConfiguration) -> None:
        """
        Initialize model integration handler.
        
        Args:
            config: APIConfiguration instance with validated settings
        """
        self.config = config
        self._cached_dimension: Optional[int] = None
    
    def generate_model_fingerprint(self) -> 'ModelFingerprint':
        """
        Generate a model fingerprint for change detection.
        
        Creates a unique fingerprint based on configuration parameters that affect
        model behavior. Used by the model change detection system to identify
        when the embedding model configuration has changed.
        
        Returns:
            ModelFingerprint object containing model metadata and checksum
            
        Example:
            >>> integration = ModelIntegration(config)
            >>> fingerprint = integration.generate_model_fingerprint()
            >>> print(f"Model: {fingerprint.model_name}, Checksum: {fingerprint.checksum}")
        """
        # Import here to avoid circular imports
        from ..model_change_detection import ModelFingerprint
        
        # Create a checksum based on config parameters that affect model behavior
        checksum_data = (
            f"{self.config.provider}:{self.config.model_name}:{self.config.base_url}:"
            f"{self.config.embedding_dimension}:{self.config.max_batch_size}"
        )
        checksum = hashlib.md5(checksum_data.encode()).hexdigest()
        
        # Get additional model info if available
        try:
            # We can't call get_model_info here due to circular dependency,
            # so we'll use config-based dimension
            dimension = self.config.embedding_dimension
        except Exception:
            dimension = self.config.embedding_dimension
        
        return ModelFingerprint(
            model_name=self.config.model_name,
            model_type="api",
            version="1.0.0",  # Could be enhanced to get actual API version
            checksum=checksum,
            metadata={
                "provider": self.config.provider,
                "base_url": self.config.base_url,
                "dimension": dimension,
                "max_batch_size": self.config.max_batch_size,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout
            }
        )
    
    def check_model_changed(self) -> bool:
        """
        Check if the model configuration has changed since last check.
        
        Integrates with the model change detection system to determine if
        the current configuration differs from the previously registered
        configuration.
        
        Returns:
            True if model configuration has changed, False otherwise
            
        Example:
            >>> integration = ModelIntegration(config)
            >>> if integration.check_model_changed():
            ...     print("Model configuration has changed, invalidating caches")
        """
        from ..model_change_detection import ModelChangeDetector
        
        detector = ModelChangeDetector()
        current_fingerprint = self.generate_model_fingerprint()
        
        changed = detector.detect_change(current_fingerprint)
        
        if changed:
            # Register the new fingerprint
            detector.register_model(current_fingerprint)
        
        return changed
    
    def invalidate_cache_on_change(self, service_instance) -> None:
        """
        Invalidate any cached data if model configuration has changed.
        
        For API services, this mainly clears the cached embedding dimension
        and any other model-specific cached data.
        
        Args:
            service_instance: The APIEmbeddingService instance to invalidate cache for
            
        Example:
            >>> integration = ModelIntegration(config)
            >>> integration.invalidate_cache_on_change(service)
        """
        if self.check_model_changed():
            # Clear cached dimension in the service instance
            if hasattr(service_instance, '_cached_dimension'):
                service_instance._cached_dimension = None
            
            logger.info(f"API model cache cleared due to change detection for {self.config.model_name}")
        else:
            logger.debug(f"No model change detected for {self.config.model_name}, cache preserved")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata for information purposes.
        
        Returns:
            Dictionary containing model metadata including provider,
            configuration, and fingerprint information
        """
        fingerprint = self.generate_model_fingerprint()
        
        return {
            "model_name": self.config.model_name,
            "model_type": "api",
            "provider": self.config.provider,
            "base_url": self.config.base_url,
            "dimension": self.config.embedding_dimension,
            "fingerprint": fingerprint.checksum,
            "metadata": fingerprint.metadata
        } 