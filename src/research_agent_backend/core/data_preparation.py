"""
Data Preparation and Normalization for Research Agent Vector Database.

This module provides comprehensive data preparation capabilities including
data cleaning, normalization, and dimensionality reduction for vector database insertion.

Implements FR-KB-003: Data preparation and quality assurance.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import re
import unicodedata

from ..models.metadata_schema import (
    ChunkMetadata, 
    CollectionType, 
    ContentType, 
    DocumentType,
    MetadataValidator
)
from ..utils.config import ConfigManager
from .collection_type_manager import CollectionTypeManager


logger = logging.getLogger(__name__)


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


class DataCleaningService:
    """Service for cleaning text and metadata data."""
    
    def __init__(self, config: DataCleaningConfig):
        """Initialize data cleaning service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pre-compile regex patterns for efficiency
        self._extra_whitespace_pattern = re.compile(r'\s+')
        self._control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        self._duplicate_detector = {}  # Simple content hash cache
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text or None if text should be filtered out
        """
        if not text or not isinstance(text, str):
            return None
        
        original_text = text
        
        try:
            # Remove control characters
            if self.config.remove_control_chars:
                text = self._control_char_pattern.sub('', text)
            
            # Normalize Unicode
            if self.config.normalize_unicode:
                text = unicodedata.normalize('NFKC', text)
            
            # Fix common encoding issues
            if self.config.fix_encoding_issues:
                text = self._fix_encoding_issues(text)
            
            # Remove extra whitespace
            if self.config.remove_extra_whitespace:
                text = self._extra_whitespace_pattern.sub(' ', text).strip()
            
            # Length filtering
            if len(text) < self.config.min_text_length:
                return None
            
            if self.config.max_text_length and len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length].rsplit(' ', 1)[0]  # Cut at word boundary
            
            # Duplicate detection
            if self.config.remove_duplicate_content:
                text_hash = hash(text.lower().strip())
                if text_hash in self._duplicate_detector:
                    return None
                self._duplicate_detector[text_hash] = True
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Error cleaning text: {e}")
            return original_text if len(original_text) >= self.config.min_text_length else None
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and standardize metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        cleaned_metadata = {}
        
        for key, value in metadata.items():
            # Standardize field names
            if self.config.standardize_metadata_fields:
                key = self._standardize_field_name(key)
            
            # Handle missing values
            if value is None or value == "":
                if self.config.fill_missing_metadata and key in self.config.default_values:
                    value = self.config.default_values[key]
                elif self.config.fill_missing_metadata:
                    value = self._get_default_value_for_type(key)
            
            # Clean string values
            if isinstance(value, str):
                value = self._clean_metadata_string(value)
            
            # Validate and convert types
            if self.config.validate_metadata_types:
                value = self._validate_metadata_type(key, value)
            
            if value is not None:
                cleaned_metadata[key] = value
        
        return cleaned_metadata
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in text."""
        # Common encoding fixes
        fixes = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '–', 'â€"': '—',
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã ': 'à', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò', 'Ã¹': 'ù'
        }
        
        for old, new in fixes.items():
            text = text.replace(old, new)
        
        return text
    
    def _standardize_field_name(self, field_name: str) -> str:
        """Standardize metadata field names."""
        # Convert to snake_case
        field_name = re.sub(r'([A-Z])', r'_\1', field_name).lower()
        field_name = re.sub(r'[^a-z0-9_]', '_', field_name)
        field_name = re.sub(r'_+', '_', field_name).strip('_')
        return field_name
    
    def _clean_metadata_string(self, value: str) -> str:
        """Clean string metadata values."""
        if not value:
            return value
        
        # Remove extra whitespace
        value = self._extra_whitespace_pattern.sub(' ', value).strip()
        
        # Normalize unicode
        if self.config.normalize_unicode:
            value = unicodedata.normalize('NFKC', value)
        
        return value
    
    def _validate_metadata_type(self, key: str, value: Any) -> Any:
        """Validate and convert metadata types."""
        # Basic type validation and conversion
        if key.endswith('_id') and isinstance(value, (int, float)):
            return str(value)
        elif key.endswith('_at') and isinstance(value, str):
            # Try to parse datetime strings
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return value
            except ValueError:
                return datetime.utcnow().isoformat()
        elif key in ['chunk_sequence_id', 'chunk_size'] and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        
        return value
    
    def _get_default_value_for_type(self, key: str) -> Any:
        """Get default value based on field name patterns."""
        if key.endswith('_id'):
            return ""
        elif key.endswith('_at'):
            return datetime.utcnow().isoformat()
        elif key in ['chunk_sequence_id', 'chunk_size']:
            return 0
        elif key in ['user_id', 'team_id', 'document_title']:
            return ""
        else:
            return ""


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
            self.logger.error(f"Error reducing dimensions: {e}")
            return embeddings
    
    def _apply_pca(self, embeddings: np.ndarray, target_dims: Optional[int]) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        try:
            from sklearn.decomposition import PCA
            
            if target_dims is None:
                # Determine components based on explained variance ratio
                pca_full = PCA()
                pca_full.fit(embeddings)
                cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
                target_dims = np.argmax(cumsum_ratio >= self.config.pca_explained_variance_ratio) + 1
            
            pca = PCA(
                n_components=min(target_dims, embeddings.shape[1], embeddings.shape[0]),
                whiten=self.config.pca_whiten
            )
            
            reduced_embeddings = pca.fit_transform(embeddings)
            
            self.logger.info(
                f"PCA reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]}, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.3f}"
            )
            
            return reduced_embeddings
            
        except ImportError:
            self.logger.error("scikit-learn not available for PCA")
            return embeddings
    
    def _apply_tsne(self, embeddings: np.ndarray, target_dims: Optional[int]) -> np.ndarray:
        """Apply t-SNE dimensionality reduction."""
        try:
            from sklearn.manifold import TSNE
            
            target_dims = target_dims or 2  # t-SNE typically used for 2D/3D visualization
            
            tsne = TSNE(
                n_components=min(target_dims, 3),  # t-SNE typically limited to 2-3 dimensions
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


class DataPreparationManager:
    """
    Main coordinator for data preparation and normalization operations.
    
    This class orchestrates data cleaning, normalization, and dimensionality
    reduction for vector database insertion with collection type awareness.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        collection_type_manager: Optional[CollectionTypeManager] = None,
        cleaning_config: Optional[DataCleaningConfig] = None,
        normalization_config: Optional[NormalizationConfig] = None,
        dimensionality_config: Optional[DimensionalityReductionConfig] = None
    ):
        """
        Initialize data preparation manager.
        
        Args:
            config_manager: Configuration manager instance
            collection_type_manager: Collection type manager for type-aware processing
            cleaning_config: Data cleaning configuration
            normalization_config: Normalization configuration  
            dimensionality_config: Dimensionality reduction configuration
        """
        self.config_manager = config_manager or ConfigManager()
        self.collection_type_manager = collection_type_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize service configurations
        self.cleaning_config = cleaning_config or self._load_cleaning_config()
        self.normalization_config = normalization_config or self._load_normalization_config()
        self.dimensionality_config = dimensionality_config or self._load_dimensionality_config()
        
        # Initialize services
        self.cleaning_service = DataCleaningService(self.cleaning_config)
        self.normalization_service = NormalizationService(self.normalization_config)
        self.dimensionality_service = DimensionalityReductionService(self.dimensionality_config)
        
        # Initialize metadata validator
        self.metadata_validator = MetadataValidator()
    
    def prepare_documents(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        collection_type: Optional[CollectionType] = None,
        batch_size: Optional[int] = None
    ) -> DataPreparationResult:
        """
        Prepare documents for vector database insertion.
        
        Args:
            texts: List of text content to prepare
            embeddings: Optional pre-computed embeddings
            metadata_list: Optional metadata for each text
            collection_type: Target collection type for type-aware processing
            batch_size: Batch size for processing (uses collection default if None)
            
        Returns:
            DataPreparationResult with processed data and statistics
        """
        start_time = datetime.utcnow()
        original_count = len(texts)
        
        # Initialize result
        result = DataPreparationResult(
            cleaned_texts=[],
            original_count=original_count
        )
        
        try:
            # Determine batch size based on collection type
            if batch_size is None and collection_type and self.collection_type_manager:
                batch_size = self.collection_type_manager.get_batch_size(collection_type)
            batch_size = batch_size or 100
            
            # Apply collection type-specific configurations
            if collection_type and self.collection_type_manager:
                self._apply_collection_type_config(collection_type)
            
            # Process in batches
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_metadata = metadata_list[batch_start:batch_end] if metadata_list else None
                batch_embeddings = embeddings[batch_start:batch_end] if embeddings is not None else None
                
                # Process batch
                batch_result = self._process_batch(
                    batch_texts,
                    batch_embeddings,
                    batch_metadata,
                    collection_type
                )
                
                # Accumulate results
                result.cleaned_texts.extend(batch_result.cleaned_texts)
                result.processed_metadata.extend(batch_result.processed_metadata)
                
                if batch_result.normalized_embeddings is not None:
                    if result.normalized_embeddings is None:
                        result.normalized_embeddings = batch_result.normalized_embeddings
                    else:
                        result.normalized_embeddings = np.vstack([
                            result.normalized_embeddings,
                            batch_result.normalized_embeddings
                        ])
                
                # Accumulate statistics
                result.processed_count += batch_result.processed_count
                result.filtered_count += batch_result.filtered_count
                result.error_count += batch_result.error_count
                result.duplicate_removed += batch_result.duplicate_removed
                result.empty_removed += batch_result.empty_removed
                result.invalid_removed += batch_result.invalid_removed
                result.warnings.extend(batch_result.warnings)
                result.errors.extend(batch_result.errors)
            
            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
            
            # Log summary
            self.logger.info(
                f"Data preparation completed: {result.processed_count}/{result.original_count} "
                f"documents processed ({result.success_rate:.1%} success rate) "
                f"in {result.processing_time_seconds:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in document preparation: {e}")
            result.errors.append(str(e))
            result.error_count = original_count
            return result
    
    def prepare_single_document(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_type: Optional[CollectionType] = None
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Prepare a single document for vector database insertion.
        
        Args:
            text: Text content to prepare
            embedding: Optional pre-computed embedding
            metadata: Optional metadata
            collection_type: Target collection type
            
        Returns:
            Tuple of (cleaned_text, normalized_embedding, processed_metadata)
        """
        # Process as single-item batch
        result = self.prepare_documents(
            texts=[text],
            embeddings=embedding.reshape(1, -1) if embedding is not None else None,
            metadata_list=[metadata] if metadata else None,
            collection_type=collection_type,
            batch_size=1
        )
        
        if result.processed_count > 0:
            cleaned_text = result.cleaned_texts[0]
            normalized_embedding = result.normalized_embeddings[0] if result.normalized_embeddings is not None else None
            processed_metadata = result.processed_metadata[0] if result.processed_metadata else None
            return cleaned_text, normalized_embedding, processed_metadata
        else:
            return None, None, None
    
    def _process_batch(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]],
        collection_type: Optional[CollectionType]
    ) -> DataPreparationResult:
        """Process a batch of documents."""
        batch_result = DataPreparationResult(
            cleaned_texts=[],
            original_count=len(texts)
        )
        
        # Step 1: Clean texts
        cleaned_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            cleaned_text = self.cleaning_service.clean_text(text)
            if cleaned_text is not None:
                cleaned_texts.append(cleaned_text)
                valid_indices.append(i)
            else:
                if not text or len(text.strip()) == 0:
                    batch_result.empty_removed += 1
                else:
                    batch_result.filtered_count += 1
        
        batch_result.cleaned_texts = cleaned_texts
        batch_result.processed_count = len(cleaned_texts)
        
        # Step 2: Process metadata
        if metadata_list:
            processed_metadata = []
            for i in valid_indices:
                if i < len(metadata_list):
                    metadata = metadata_list[i]
                    cleaned_metadata = self.cleaning_service.clean_metadata(metadata)
                    
                    # Validate with metadata validator
                    try:
                        validated_metadata = self.metadata_validator.validate_metadata(cleaned_metadata)
                        processed_metadata.append(validated_metadata)
                    except Exception as e:
                        self.logger.warning(f"Metadata validation failed: {e}")
                        processed_metadata.append(cleaned_metadata)
                else:
                    processed_metadata.append({})
            
            batch_result.processed_metadata = processed_metadata
        
        # Step 3: Process embeddings
        if embeddings is not None and len(valid_indices) > 0:
            # Filter embeddings to valid indices
            valid_embeddings = embeddings[valid_indices]
            
            # Normalize embeddings
            normalized_embeddings = self.normalization_service.normalize_embeddings(valid_embeddings)
            
            # Apply dimensionality reduction if configured
            reduced_embeddings = self.dimensionality_service.reduce_dimensions(normalized_embeddings)
            
            batch_result.normalized_embeddings = reduced_embeddings
        
        return batch_result
    
    def _apply_collection_type_config(self, collection_type: CollectionType):
        """Apply collection type-specific configurations."""
        if not self.collection_type_manager:
            return
        
        # Get collection type configuration
        type_config = self.collection_type_manager.get_collection_config(collection_type)
        
        # Adjust cleaning configuration based on collection type
        if collection_type == CollectionType.FUNDAMENTAL:
            # More aggressive cleaning for fundamental knowledge
            self.cleaning_config.min_text_length = 20
            self.cleaning_config.remove_duplicate_content = True
            self.cleaning_config.content_similarity_threshold = 0.90
        elif collection_type == CollectionType.PROJECT_SPECIFIC:
            # Preserve more context for project documents
            self.cleaning_config.min_text_length = 10
            self.cleaning_config.remove_duplicate_content = False
        elif collection_type == CollectionType.TEMPORARY:
            # Minimal cleaning for temporary data
            self.cleaning_config.min_text_length = 5
            self.cleaning_config.validate_metadata_types = False
        
        # Reinitialize cleaning service with updated config
        self.cleaning_service = DataCleaningService(self.cleaning_config)
    
    def _load_cleaning_config(self) -> DataCleaningConfig:
        """Load data cleaning configuration from config manager."""
        try:
            config_data = self.config_manager.get_config()
            cleaning_section = config_data.get('data_preparation', {}).get('cleaning', {})
            
            return DataCleaningConfig(
                remove_extra_whitespace=cleaning_section.get('remove_extra_whitespace', True),
                normalize_unicode=cleaning_section.get('normalize_unicode', True),
                remove_control_chars=cleaning_section.get('remove_control_chars', True),
                fix_encoding_issues=cleaning_section.get('fix_encoding_issues', True),
                min_text_length=cleaning_section.get('min_text_length', 10),
                max_text_length=cleaning_section.get('max_text_length'),
                remove_empty_content=cleaning_section.get('remove_empty_content', True),
                remove_duplicate_content=cleaning_section.get('remove_duplicate_content', True),
                content_similarity_threshold=cleaning_section.get('content_similarity_threshold', 0.95),
                standardize_metadata_fields=cleaning_section.get('standardize_metadata_fields', True),
                validate_metadata_types=cleaning_section.get('validate_metadata_types', True),
                fill_missing_metadata=cleaning_section.get('fill_missing_metadata', True),
                default_values=cleaning_section.get('default_values', {})
            )
        except Exception as e:
            self.logger.warning(f"Failed to load cleaning config: {e}, using defaults")
            return DataCleaningConfig()
    
    def _load_normalization_config(self) -> NormalizationConfig:
        """Load normalization configuration from config manager."""
        try:
            config_data = self.config_manager.get_config()
            norm_section = config_data.get('data_preparation', {}).get('normalization', {})
            
            return NormalizationConfig(
                embedding_method=NormalizationMethod(norm_section.get('embedding_method', 'unit_vector')),
                preserve_magnitude=norm_section.get('preserve_magnitude', False),
                handle_zero_vectors=norm_section.get('handle_zero_vectors', True),
                normalize_numerical_metadata=norm_section.get('normalize_numerical_metadata', True),
                numerical_method=NormalizationMethod(norm_section.get('numerical_method', 'min_max')),
                feature_range=tuple(norm_section.get('feature_range', [0.0, 1.0])),
                with_centering=norm_section.get('with_centering', True),
                with_scaling=norm_section.get('with_scaling', True)
            )
        except Exception as e:
            self.logger.warning(f"Failed to load normalization config: {e}, using defaults")
            return NormalizationConfig()
    
    def _load_dimensionality_config(self) -> DimensionalityReductionConfig:
        """Load dimensionality reduction configuration from config manager."""
        try:
            config_data = self.config_manager.get_config()
            dim_section = config_data.get('data_preparation', {}).get('dimensionality_reduction', {})
            
            return DimensionalityReductionConfig(
                method=DimensionalityReductionMethod(dim_section.get('method', 'none')),
                target_dimensions=dim_section.get('target_dimensions'),
                pca_explained_variance_ratio=dim_section.get('pca_explained_variance_ratio', 0.95),
                pca_whiten=dim_section.get('pca_whiten', False),
                tsne_perplexity=dim_section.get('tsne_perplexity', 30.0),
                tsne_learning_rate=dim_section.get('tsne_learning_rate', 200.0),
                tsne_n_iter=dim_section.get('tsne_n_iter', 1000),
                umap_n_neighbors=dim_section.get('umap_n_neighbors', 15),
                umap_min_dist=dim_section.get('umap_min_dist', 0.1),
                umap_metric=dim_section.get('umap_metric', 'cosine')
            )
        except Exception as e:
            self.logger.warning(f"Failed to load dimensionality config: {e}, using defaults")
            return DimensionalityReductionConfig()


def create_data_preparation_manager(
    config_file: Optional[str] = None,
    collection_type_manager: Optional[CollectionTypeManager] = None,
    **config_overrides
) -> DataPreparationManager:
    """
    Factory function to create DataPreparationManager instance.
    
    Args:
        config_file: Path to configuration file
        collection_type_manager: Optional collection type manager
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized DataPreparationManager instance
    """
    config_manager = ConfigManager(config_file=config_file) if config_file else ConfigManager()
    
    # Apply configuration overrides
    cleaning_config = None
    normalization_config = None
    dimensionality_config = None
    
    if 'cleaning' in config_overrides:
        cleaning_config = DataCleaningConfig(**config_overrides['cleaning'])
    
    if 'normalization' in config_overrides:
        normalization_config = NormalizationConfig(**config_overrides['normalization'])
    
    if 'dimensionality' in config_overrides:
        dimensionality_config = DimensionalityReductionConfig(**config_overrides['dimensionality'])
    
    return DataPreparationManager(
        config_manager=config_manager,
        collection_type_manager=collection_type_manager,
        cleaning_config=cleaning_config,
        normalization_config=normalization_config,
        dimensionality_config=dimensionality_config
    ) 