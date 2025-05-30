"""
Dimensionality reduction service for high-dimensional data.

This module provides dimensionality reduction capabilities using various algorithms
including PCA, t-SNE, and UMAP for high-dimensional embedding vectors.
"""

import logging
import numpy as np
from typing import Optional

from .types import DimensionalityReductionConfig, DimensionalityReductionMethod


class DimensionalityReductionService:
    """Service for dimensionality reduction of high-dimensional data."""
    
    def __init__(self, config: DimensionalityReductionConfig):
        """Initialize dimensionality reduction service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store fitted reducers for consistency
        self._fitted_reducers = {}
    
    def reduce_dimensions(
        self, 
        embeddings: np.ndarray,
        method: Optional[DimensionalityReductionMethod] = None,
        target_dims: Optional[int] = None
    ) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.
        
        Args:
            embeddings: High-dimensional embedding vectors
            method: Reduction method (uses config default if None)
            target_dims: Target dimensions (uses config default if None)
            
        Returns:
            Reduced embeddings
        """
        if embeddings is None or embeddings.size == 0:
            return embeddings
        
        method = method or self.config.method
        target_dims = target_dims or self.config.target_dimensions
        
        if method == DimensionalityReductionMethod.NONE:
            return embeddings
        
        # Check if reduction is needed
        current_dims = embeddings.shape[1] if embeddings.ndim > 1 else 1
        if target_dims and current_dims <= target_dims:
            return embeddings
        
        try:
            if method == DimensionalityReductionMethod.PCA:
                return self._apply_pca(embeddings, target_dims)
            elif method == DimensionalityReductionMethod.TSNE:
                return self._apply_tsne(embeddings, target_dims)
            elif method == DimensionalityReductionMethod.UMAP:
                return self._apply_umap(embeddings, target_dims)
            else:
                self.logger.warning(f"Unknown dimensionality reduction method: {method}")
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Error in dimensionality reduction: {e}")
            return embeddings
    
    def _apply_pca(self, embeddings: np.ndarray, target_dims: Optional[int]) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        try:
            from sklearn.decomposition import PCA
            
            if target_dims:
                n_components = min(target_dims, embeddings.shape[1], embeddings.shape[0])
            else:
                # Use explained variance ratio to determine components
                pca_temp = PCA()
                pca_temp.fit(embeddings)
                
                cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_var >= self.config.pca_explained_variance_ratio) + 1
                n_components = min(n_components, embeddings.shape[1])
            
            pca = PCA(
                n_components=n_components,
                whiten=self.config.pca_whiten,
                random_state=42
            )
            
            reduced_embeddings = pca.fit_transform(embeddings)
            
            self.logger.info(
                f"PCA reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]} "
                f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
            )
            
            return reduced_embeddings
            
        except ImportError:
            self.logger.error("scikit-learn not available for PCA")
            return embeddings
    
    def _apply_tsne(self, embeddings: np.ndarray, target_dims: Optional[int]) -> np.ndarray:
        """Apply t-SNE dimensionality reduction."""
        try:
            from sklearn.manifold import TSNE
            
            target_dims = target_dims or 2  # t-SNE commonly used for 2D visualization
            
            # t-SNE is computationally expensive, limit input size
            if embeddings.shape[0] > 5000:
                self.logger.warning("Large dataset for t-SNE, consider using PCA first")
            
            tsne = TSNE(
                n_components=target_dims,
                perplexity=min(self.config.tsne_perplexity, embeddings.shape[0] - 1),
                learning_rate=self.config.tsne_learning_rate,
                n_iter=self.config.tsne_n_iter,
                random_state=42
            )
            
            reduced_embeddings = tsne.fit_transform(embeddings)
            
            self.logger.info(
                f"t-SNE reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]}"
            )
            
            return reduced_embeddings
            
        except ImportError:
            self.logger.error("scikit-learn not available for t-SNE")
            return embeddings
    
    def _apply_umap(self, embeddings: np.ndarray, target_dims: Optional[int]) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        try:
            import umap
            
            target_dims = target_dims or 2  # UMAP commonly used for 2D visualization
            
            reducer = umap.UMAP(
                n_components=target_dims,
                n_neighbors=min(self.config.umap_n_neighbors, embeddings.shape[0] - 1),
                min_dist=self.config.umap_min_dist,
                metric=self.config.umap_metric,
                random_state=42
            )
            
            reduced_embeddings = reducer.fit_transform(embeddings)
            
            self.logger.info(
                f"UMAP reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]}"
            )
            
            return reduced_embeddings
            
        except ImportError:
            self.logger.error("umap-learn not available for UMAP")
            return embeddings 