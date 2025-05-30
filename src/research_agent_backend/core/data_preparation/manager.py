"""
Main data preparation manager for orchestrating all data processing operations.

This module provides the main DataPreparationManager class that coordinates data cleaning,
normalization, and dimensionality reduction services with collection type awareness.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    DataCleaningConfig, 
    NormalizationConfig, 
    DimensionalityReductionConfig,
    DataPreparationResult,
    NormalizationMethod,
    DimensionalityReductionMethod
)
from .cleaning import DataCleaningService
from .normalization import NormalizationService
from .dimensionality import DimensionalityReductionService

from ...models.metadata_schema import CollectionType, MetadataValidator
from ...utils.config import ConfigManager
from ..collection_type_manager import CollectionTypeManager


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
        # Apply collection type configuration
        if collection_type and self.collection_type_manager:
            self._apply_collection_type_config(collection_type)
        
        # Clean text
        cleaned_text = self.cleaning_service.clean_text(text)
        if cleaned_text is None:
            return None, None, None
        
        # Process metadata
        processed_metadata = None
        if metadata:
            processed_metadata = self.cleaning_service.clean_metadata(metadata)
            try:
                processed_metadata = self.metadata_validator.validate_metadata(processed_metadata)
            except Exception as e:
                self.logger.warning(f"Metadata validation failed: {e}")
        
        # Process embedding
        normalized_embedding = None
        if embedding is not None:
            # Ensure embedding is 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Normalize embedding
            normalized_embedding = self.normalization_service.normalize_embeddings(embedding)
            
            # Apply dimensionality reduction
            reduced_embedding = self.dimensionality_service.reduce_dimensions(normalized_embedding)
            normalized_embedding = reduced_embedding[0] if reduced_embedding.ndim > 1 else reduced_embedding
        
        return cleaned_text, normalized_embedding, processed_metadata
    
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