"""
Normalization service for numerical data and embeddings.

This module provides comprehensive normalization capabilities for embedding vectors
and numerical metadata using various normalization methods.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union

from .types import NormalizationConfig, NormalizationMethod


class NormalizationService:
    """Service for normalizing numerical data and embeddings."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize normalization service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store fitted scalers for consistency
        self._fitted_scalers = {}
    
    def normalize_embeddings(
        self, 
        embeddings: np.ndarray,
        method: Optional[NormalizationMethod] = None
    ) -> np.ndarray:
        """
        Normalize embedding vectors.
        
        Args:
            embeddings: Array of embedding vectors
            method: Normalization method (uses config default if None)
            
        Returns:
            Normalized embeddings
        """
        if embeddings is None or embeddings.size == 0:
            return embeddings
        
        method = method or self.config.embedding_method
        
        if method == NormalizationMethod.NONE:
            return embeddings
        
        try:
            if method == NormalizationMethod.UNIT_VECTOR:
                return self._normalize_unit_vector(embeddings)
            elif method == NormalizationMethod.MIN_MAX:
                return self._normalize_min_max(embeddings)
            elif method == NormalizationMethod.Z_SCORE:
                return self._normalize_z_score(embeddings)
            elif method == NormalizationMethod.ROBUST:
                return self._normalize_robust(embeddings)
            else:
                self.logger.warning(f"Unknown normalization method: {method}")
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Error normalizing embeddings: {e}")
            return embeddings
    
    def normalize_numerical_metadata(
        self, 
        metadata_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize numerical metadata fields.
        
        Args:
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of metadata with normalized numerical fields
        """
        if not self.config.normalize_numerical_metadata:
            return metadata_list
        
        # Extract numerical fields
        numerical_fields = self._identify_numerical_fields(metadata_list)
        
        if not numerical_fields:
            return metadata_list
        
        # Normalize each numerical field
        normalized_metadata = []
        for metadata in metadata_list:
            normalized_meta = metadata.copy()
            
            for field in numerical_fields:
                if field in metadata and isinstance(metadata[field], (int, float)):
                    normalized_meta[field] = self._normalize_single_value(
                        metadata[field], 
                        field,
                        self.config.numerical_method
                    )
            
            normalized_metadata.append(normalized_meta)
        
        return normalized_metadata
    
    def _normalize_unit_vector(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Handle zero vectors
        if self.config.handle_zero_vectors:
            norms = np.where(norms == 0, 1, norms)
        
        return embeddings / norms
    
    def _normalize_min_max(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply min-max normalization."""
        min_vals = np.min(embeddings, axis=0)
        max_vals = np.max(embeddings, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        
        # Scale to feature range
        normalized = (embeddings - min_vals) / range_vals
        feature_min, feature_max = self.config.feature_range
        normalized = normalized * (feature_max - feature_min) + feature_min
        
        return normalized
    
    def _normalize_z_score(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply z-score (standard) normalization."""
        if self.config.with_centering:
            mean_vals = np.mean(embeddings, axis=0)
            embeddings = embeddings - mean_vals
        
        if self.config.with_scaling:
            std_vals = np.std(embeddings, axis=0)
            std_vals = np.where(std_vals == 0, 1, std_vals)
            embeddings = embeddings / std_vals
        
        return embeddings
    
    def _normalize_robust(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply robust normalization using median and IQR."""
        median_vals = np.median(embeddings, axis=0)
        q75 = np.percentile(embeddings, 75, axis=0)
        q25 = np.percentile(embeddings, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1, iqr)
        
        return (embeddings - median_vals) / iqr
    
    def _identify_numerical_fields(self, metadata_list: List[Dict[str, Any]]) -> List[str]:
        """Identify numerical fields in metadata."""
        numerical_fields = set()
        
        for metadata in metadata_list[:100]:  # Sample first 100 entries
            for key, value in metadata.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Exclude ID fields and timestamps
                    if not key.endswith('_id') and not key.endswith('_at'):
                        numerical_fields.add(key)
        
        return list(numerical_fields)
    
    def _normalize_single_value(
        self, 
        value: Union[int, float], 
        field_name: str, 
        method: NormalizationMethod
    ) -> float:
        """Normalize a single numerical value."""
        # For individual values, we can only apply simple transformations
        if method == NormalizationMethod.MIN_MAX:
            # For single values, we can't determine range, so return as-is
            return float(value)
        elif method == NormalizationMethod.Z_SCORE:
            # For single values, we can't determine mean/std, so return as-is
            return float(value)
        else:
            return float(value) 