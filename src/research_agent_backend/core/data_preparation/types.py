"""
Types and configuration classes for data preparation.

This module contains all enums, dataclasses, and configuration types used by the 
data preparation system, including cleaning, normalization, and dimensionality reduction configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class NormalizationMethod(Enum):
    """Normalization methods for numerical data."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    UNIT_VECTOR = "unit_vector"
    ROBUST = "robust"
    NONE = "none"


class DimensionalityReductionMethod(Enum):
    """Dimensionality reduction methods."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    NONE = "none"


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning operations."""
    
    # Text cleaning
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    fix_encoding_issues: bool = True
    min_text_length: int = 10
    max_text_length: Optional[int] = None
    
    # Content filtering
    remove_empty_content: bool = True
    remove_duplicate_content: bool = True
    content_similarity_threshold: float = 0.95
    
    # Metadata cleaning
    standardize_metadata_fields: bool = True
    validate_metadata_types: bool = True
    fill_missing_metadata: bool = True
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    # Language processing
    detect_language: bool = True
    filter_languages: Optional[List[str]] = None
    transliterate_non_latin: bool = False


@dataclass
class NormalizationConfig:
    """Configuration for normalization operations."""
    
    # Vector normalization
    embedding_method: NormalizationMethod = NormalizationMethod.UNIT_VECTOR
    preserve_magnitude: bool = False
    handle_zero_vectors: bool = True
    
    # Numerical metadata normalization
    normalize_numerical_metadata: bool = True
    numerical_method: NormalizationMethod = NormalizationMethod.MIN_MAX
    
    # Scaling parameters
    feature_range: Tuple[float, float] = (0.0, 1.0)
    with_centering: bool = True
    with_scaling: bool = True


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction operations."""
    
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.NONE
    target_dimensions: Optional[int] = None
    
    # PCA parameters
    pca_explained_variance_ratio: float = 0.95
    pca_whiten: bool = False
    
    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"


@dataclass
class DataPreparationResult:
    """Result of data preparation operations."""
    
    # Processed data
    cleaned_texts: List[str]
    normalized_embeddings: Optional[np.ndarray] = None
    processed_metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing statistics
    original_count: int = 0
    processed_count: int = 0
    filtered_count: int = 0
    error_count: int = 0
    
    # Quality metrics
    duplicate_removed: int = 0
    empty_removed: int = 0
    invalid_removed: int = 0
    
    # Processing details
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing."""
        if self.original_count == 0:
            return 0.0
        return self.processed_count / self.original_count
    
    @property
    def filter_rate(self) -> float:
        """Calculate filter rate of processing."""
        if self.original_count == 0:
            return 0.0
        return self.filtered_count / self.original_count 