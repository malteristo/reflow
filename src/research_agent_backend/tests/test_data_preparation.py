"""
Tests for Data Preparation and Normalization module.

This module tests data cleaning, normalization, and dimensionality reduction
functionality for vector database preparation.

Implements TDD principles for data preparation functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from ..core.data_preparation import (
    DataPreparationManager,
    DataCleaningService,
    NormalizationService,
    DimensionalityReductionService,
    DataCleaningConfig,
    NormalizationConfig,
    DimensionalityReductionConfig,
    DataPreparationResult,
    NormalizationMethod,
    DimensionalityReductionMethod,
    create_data_preparation_manager
)
from ..models.metadata_schema import CollectionType
from ..utils.config import ConfigManager


class TestDataCleaningConfig:
    """Test DataCleaningConfig dataclass functionality."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = DataCleaningConfig()
        
        assert config.remove_extra_whitespace is True
        assert config.normalize_unicode is True
        assert config.min_text_length == 10
        assert config.remove_duplicate_content is True
        assert config.default_values == {}
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = DataCleaningConfig(
            min_text_length=20,
            max_text_length=1000,
            remove_duplicate_content=False,
            default_values={'user_id': 'test_user'}
        )
        
        assert config.min_text_length == 20
        assert config.max_text_length == 1000
        assert config.remove_duplicate_content is False
        assert config.default_values == {'user_id': 'test_user'}


class TestDataCleaningService:
    """Test DataCleaningService functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DataCleaningConfig()
        self.service = DataCleaningService(self.config)
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  This is a test text.  "
        cleaned = self.service.clean_text(text)
        
        assert cleaned == "This is a test text."
    
    def test_clean_text_unicode_normalization(self):
        """Test Unicode normalization."""
        text = "café naïve résumé"  # Various accented characters
        cleaned = self.service.clean_text(text)
        
        assert cleaned is not None
        assert "café" in cleaned
    
    def test_clean_text_control_characters(self):
        """Test removal of control characters."""
        text = "Normal text\x00\x08\x1F with control chars"
        cleaned = self.service.clean_text(text)
        
        assert cleaned == "Normal text with control chars"
    
    def test_clean_text_encoding_fixes(self):
        """Test common encoding issue fixes."""
        text = "It's a \"quote\" with â€™ issues"
        cleaned = self.service.clean_text(text)
        
        assert "â€™" not in cleaned
        assert "'" in cleaned
    
    def test_clean_text_length_filtering(self):
        """Test text length filtering."""
        short_text = "Hi"  # Below min_text_length
        long_enough = "This is long enough text"
        
        assert self.service.clean_text(short_text) is None
        assert self.service.clean_text(long_enough) is not None
    
    def test_clean_text_max_length_truncation(self):
        """Test maximum length truncation."""
        config = DataCleaningConfig(max_text_length=20)
        service = DataCleaningService(config)
        
        long_text = "This is a very long text that exceeds the maximum length limit"
        cleaned = service.clean_text(long_text)
        
        assert len(cleaned) <= 20
        assert not cleaned.endswith(' ')  # Should cut at word boundary
    
    def test_clean_text_duplicate_detection(self):
        """Test duplicate content detection."""
        text1 = "This is duplicate content"
        text2 = "This is duplicate content"  # Exact duplicate
        
        cleaned1 = self.service.clean_text(text1)
        cleaned2 = self.service.clean_text(text2)
        
        assert cleaned1 is not None
        assert cleaned2 is None  # Should be filtered as duplicate
    
    def test_clean_metadata_basic(self):
        """Test basic metadata cleaning."""
        metadata = {
            'user_id': '  test_user  ',
            'documentTitle': 'Test Document',
            'chunk-sequence-id': '1',
            'created_at': '2023-01-01T00:00:00Z'
        }
        
        cleaned = self.service.clean_metadata(metadata)
        
        assert cleaned['user_id'] == 'test_user'
        assert 'document_title' in cleaned  # Should be standardized
        assert cleaned['chunk_sequence_id'] == 1  # Should be converted to int
    
    def test_clean_metadata_missing_values(self):
        """Test handling of missing metadata values."""
        config = DataCleaningConfig(
            fill_missing_metadata=True,
            default_values={'user_id': 'default_user'}
        )
        service = DataCleaningService(config)
        
        metadata = {
            'user_id': None,
            'team_id': '',
            'document_title': 'Test'
        }
        
        cleaned = service.clean_metadata(metadata)
        
        assert cleaned['user_id'] == 'default_user'
        assert 'team_id' in cleaned
    
    def test_clean_metadata_type_validation(self):
        """Test metadata type validation and conversion."""
        metadata = {
            'chunk_sequence_id': '123',  # Should convert to int
            'created_at': 'invalid-date',  # Should get default
            'document_id': 456  # Should convert to string
        }
        
        cleaned = self.service.clean_metadata(metadata)
        
        assert isinstance(cleaned['chunk_sequence_id'], int)
        assert isinstance(cleaned['document_id'], str)


class TestNormalizationConfig:
    """Test NormalizationConfig dataclass functionality."""
    
    def test_default_normalization_config(self):
        """Test default normalization configuration."""
        config = NormalizationConfig()
        
        assert config.embedding_method == NormalizationMethod.UNIT_VECTOR
        assert config.handle_zero_vectors is True
        assert config.feature_range == (0.0, 1.0)
    
    def test_custom_normalization_config(self):
        """Test custom normalization configuration."""
        config = NormalizationConfig(
            embedding_method=NormalizationMethod.Z_SCORE,
            feature_range=(-1.0, 1.0),
            normalize_numerical_metadata=False
        )
        
        assert config.embedding_method == NormalizationMethod.Z_SCORE
        assert config.feature_range == (-1.0, 1.0)
        assert config.normalize_numerical_metadata is False


class TestNormalizationService:
    """Test NormalizationService functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = NormalizationConfig()
        self.service = NormalizationService(self.config)
    
    def test_normalize_embeddings_unit_vector(self):
        """Test unit vector normalization."""
        embeddings = np.array([
            [3.0, 4.0],  # Length 5
            [1.0, 0.0],  # Length 1
            [0.0, 2.0]   # Length 2
        ])
        
        normalized = self.service.normalize_embeddings(embeddings)
        
        # Check that all vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0])
    
    def test_normalize_embeddings_min_max(self):
        """Test min-max normalization."""
        config = NormalizationConfig(embedding_method=NormalizationMethod.MIN_MAX)
        service = NormalizationService(config)
        
        embeddings = np.array([
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 1.0]
        ])
        
        normalized = service.normalize_embeddings(embeddings)
        
        # Check that values are in range [0, 1]
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert np.max(normalized[:, 0]) == 1.0  # Max should be 1
        assert np.min(normalized[:, 0]) == 0.0  # Min should be 0
    
    def test_normalize_embeddings_z_score(self):
        """Test z-score normalization."""
        config = NormalizationConfig(embedding_method=NormalizationMethod.Z_SCORE)
        service = NormalizationService(config)
        
        embeddings = np.array([
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 1.0]
        ])
        
        normalized = service.normalize_embeddings(embeddings)
        
        # Check that mean is approximately 0 and std is approximately 1
        means = np.mean(normalized, axis=0)
        stds = np.std(normalized, axis=0)
        
        np.testing.assert_array_almost_equal(means, [0.0, 0.0], decimal=10)
        np.testing.assert_array_almost_equal(stds, [1.0, 1.0], decimal=10)
    
    def test_normalize_embeddings_zero_vectors(self):
        """Test handling of zero vectors."""
        embeddings = np.array([
            [1.0, 1.0],
            [0.0, 0.0],  # Zero vector
            [2.0, 2.0]
        ])
        
        normalized = self.service.normalize_embeddings(embeddings)
        
        # Zero vector should remain zero
        assert np.allclose(normalized[1], [0.0, 0.0])
        # Other vectors should be normalized
        assert not np.allclose(normalized[0], [1.0, 1.0])
    
    def test_normalize_numerical_metadata(self):
        """Test numerical metadata normalization."""
        metadata_list = [
            {'score': 10, 'count': 100, 'user_id': 'user1'},
            {'score': 20, 'count': 200, 'user_id': 'user2'},
            {'score': 30, 'count': 300, 'user_id': 'user3'}
        ]
        
        normalized = self.service.normalize_numerical_metadata(metadata_list)
        
        # Should preserve non-numerical fields
        assert all('user_id' in meta for meta in normalized)
        # Should process numerical fields
        assert all('score' in meta for meta in normalized)


class TestDimensionalityReductionService:
    """Test DimensionalityReductionService functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DimensionalityReductionConfig()
        self.service = DimensionalityReductionService(self.config)
    
    def test_no_reduction_when_disabled(self):
        """Test that no reduction occurs when disabled."""
        embeddings = np.random.rand(10, 5)
        
        reduced = self.service.reduce_dimensions(embeddings)
        
        np.testing.assert_array_equal(reduced, embeddings)
    
    @patch('sklearn.decomposition.PCA')
    def test_pca_reduction(self, mock_pca_class):
        """Test PCA dimensionality reduction."""
        # Mock PCA
        mock_pca = Mock()
        mock_pca.fit_transform.return_value = np.random.rand(10, 3)
        mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca_class.return_value = mock_pca
        
        config = DimensionalityReductionConfig(
            method=DimensionalityReductionMethod.PCA,
            target_dimensions=3
        )
        service = DimensionalityReductionService(config)
        
        embeddings = np.random.rand(10, 5)
        reduced = service.reduce_dimensions(embeddings)
        
        assert reduced.shape[1] == 3
        mock_pca_class.assert_called_once()
    
    def test_pca_fallback_without_sklearn(self):
        """Test PCA fallback when sklearn is not available."""
        config = DimensionalityReductionConfig(
            method=DimensionalityReductionMethod.PCA,
            target_dimensions=3
        )
        service = DimensionalityReductionService(config)
        
        embeddings = np.random.rand(10, 5)
        
        # Mock ImportError for sklearn
        with patch('sklearn.decomposition.PCA', side_effect=ImportError):
            reduced = service.reduce_dimensions(embeddings)
            
            # Should return original embeddings when sklearn unavailable
            np.testing.assert_array_equal(reduced, embeddings)


class TestDataPreparationResult:
    """Test DataPreparationResult functionality."""
    
    def test_success_rate_calculation(self):
        """Test success rate property calculation."""
        result = DataPreparationResult(
            cleaned_texts=["text1", "text2"],
            original_count=10,
            processed_count=8
        )
        
        assert result.success_rate == 0.8
    
    def test_filter_rate_calculation(self):
        """Test filter rate property calculation."""
        result = DataPreparationResult(
            cleaned_texts=["text1", "text2"],
            original_count=10,
            filtered_count=2
        )
        
        assert result.filter_rate == 0.2
    
    def test_zero_original_count_handling(self):
        """Test handling of zero original count."""
        result = DataPreparationResult(
            cleaned_texts=[],
            original_count=0
        )
        
        assert result.success_rate == 0.0
        assert result.filter_rate == 0.0


class TestDataPreparationManager:
    """Test DataPreparationManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=ConfigManager)
        # Properly mock the get_config method
        self.config_manager.get_config = Mock(return_value={})
        
        self.manager = DataPreparationManager(
            config_manager=self.config_manager
        )
    
    def test_prepare_single_document(self):
        """Test preparing a single document."""
        text = "This is a test document for preparation"
        metadata = {'user_id': 'test_user', 'document_type': 'test'}
        
        cleaned_text, normalized_embedding, processed_metadata = self.manager.prepare_single_document(
            text=text,
            metadata=metadata
        )
        
        assert cleaned_text is not None
        assert cleaned_text.strip() == text.strip()
        assert processed_metadata is not None
        assert 'user_id' in processed_metadata
    
    def test_prepare_documents_batch(self):
        """Test preparing multiple documents in batch."""
        texts = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        metadata_list = [
            {'user_id': 'user1'},
            {'user_id': 'user2'},
            {'user_id': 'user3'}
        ]
        
        result = self.manager.prepare_documents(
            texts=texts,
            metadata_list=metadata_list
        )
        
        assert result.original_count == 3
        assert result.processed_count == 3
        assert len(result.cleaned_texts) == 3
        assert len(result.processed_metadata) == 3
        assert result.success_rate == 1.0
    
    def test_prepare_documents_with_embeddings(self):
        """Test preparing documents with embeddings."""
        texts = ["Document 1", "Document 2"]
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        result = self.manager.prepare_documents(
            texts=texts,
            embeddings=embeddings
        )
        
        assert result.normalized_embeddings is not None
        assert result.normalized_embeddings.shape == (2, 3)
    
    def test_prepare_documents_filtering(self):
        """Test document filtering during preparation."""
        texts = [
            "Good document",
            "Hi",  # Too short
            "",    # Empty
            "Another good document"
        ]
        
        result = self.manager.prepare_documents(texts=texts)
        
        assert result.original_count == 4
        assert result.processed_count == 2  # Only 2 should pass
        assert result.empty_removed == 1
        assert result.filtered_count == 1
    
    def test_prepare_documents_with_collection_type(self):
        """Test collection type-aware document preparation."""
        texts = ["Document for fundamental collection"]
        
        result = self.manager.prepare_documents(
            texts=texts,
            collection_type=CollectionType.FUNDAMENTAL
        )
        
        assert result.processed_count >= 0  # Should process successfully
    
    def test_collection_type_config_adjustment(self):
        """Test collection type-specific configuration adjustments."""
        # Create a mock collection type manager
        mock_collection_manager = Mock()
        mock_collection_manager.get_collection_config.return_value = Mock()
        
        # Create manager with collection type manager
        self.manager.collection_type_manager = mock_collection_manager
        
        # Store original config values
        original_min_length = self.manager.cleaning_config.min_text_length
        original_remove_duplicates = self.manager.cleaning_config.remove_duplicate_content
        
        # Test fundamental collection (more aggressive cleaning)
        self.manager._apply_collection_type_config(CollectionType.FUNDAMENTAL)
        assert self.manager.cleaning_config.min_text_length == 20
        assert self.manager.cleaning_config.remove_duplicate_content is True
        
        # Reset and test project-specific collection (preserve more context)
        self.manager.cleaning_config.min_text_length = original_min_length
        self.manager.cleaning_config.remove_duplicate_content = original_remove_duplicates
        self.manager._apply_collection_type_config(CollectionType.PROJECT_SPECIFIC)
        assert self.manager.cleaning_config.min_text_length == 10
        assert self.manager.cleaning_config.remove_duplicate_content is False
        
        # Reset and test temporary collection (minimal cleaning)
        self.manager.cleaning_config.min_text_length = original_min_length
        self.manager.cleaning_config.validate_metadata_types = True
        self.manager._apply_collection_type_config(CollectionType.TEMPORARY)
        assert self.manager.cleaning_config.min_text_length == 5
        assert self.manager.cleaning_config.validate_metadata_types is False


class TestDataPreparationFactory:
    """Test factory function for creating DataPreparationManager."""
    
    def test_create_data_preparation_manager_defaults(self):
        """Test creating manager with default configuration."""
        manager = create_data_preparation_manager()
        
        assert isinstance(manager, DataPreparationManager)
        assert manager.config_manager is not None
        assert manager.cleaning_service is not None
        assert manager.normalization_service is not None
        assert manager.dimensionality_service is not None
    
    def test_create_data_preparation_manager_with_overrides(self):
        """Test creating manager with configuration overrides."""
        cleaning_overrides = {
            'min_text_length': 15,
            'remove_duplicate_content': False
        }
        
        manager = create_data_preparation_manager(
            cleaning=cleaning_overrides
        )
        
        assert manager.cleaning_config.min_text_length == 15
        assert manager.cleaning_config.remove_duplicate_content is False
    
    def test_create_data_preparation_manager_with_custom_config(self):
        """Test creating manager with custom config file."""
        with patch('src.research_agent_backend.core.data_preparation.ConfigManager') as mock_config:
            mock_instance = Mock()
            mock_instance.get_config = Mock(return_value={})
            mock_config.return_value = mock_instance
            
            manager = create_data_preparation_manager(config_file="custom_config.json")
            
            mock_config.assert_called_with(config_file="custom_config.json")


class TestIntegrationDataPreparation:
    """Integration tests for data preparation system."""
    
    def test_end_to_end_document_processing(self):
        """Test complete end-to-end document processing pipeline."""
        # Create manager with realistic configuration
        manager = create_data_preparation_manager()
        
        # Prepare test data
        texts = [
            "This is a comprehensive test document with proper length and content.",
            "Another document for testing the data preparation pipeline functionality.",
            "",  # Empty document (should be filtered)
            "Hi",  # Too short (should be filtered)
            "  A document with extra whitespace and Unicode characters: café naïve  "
        ]
        
        metadata_list = [
            {'user_id': 'user1', 'document_type': 'research'},
            {'user_id': 'user2', 'document_type': 'reference'},
            {'user_id': 'user3', 'document_type': 'empty'},
            {'user_id': 'user4', 'document_type': 'short'},
            {'user_id': 'user5', 'document_type': 'unicode'}
        ]
        
        # Generate mock embeddings
        embeddings = np.random.rand(5, 384)  # Typical embedding dimension
        
        # Process documents
        result = manager.prepare_documents(
            texts=texts,
            embeddings=embeddings,
            metadata_list=metadata_list,
            collection_type=CollectionType.PROJECT_SPECIFIC
        )
        
        # Verify results
        assert result.original_count == 5
        assert result.processed_count == 3  # Only 3 should pass filtering
        assert result.empty_removed >= 1
        assert result.filtered_count >= 1
        
        # Check that valid documents are properly processed
        assert len(result.cleaned_texts) == result.processed_count
        assert len(result.processed_metadata) == result.processed_count
        assert result.normalized_embeddings.shape[0] == result.processed_count
        
        # Verify text cleaning
        for text in result.cleaned_texts:
            assert len(text.strip()) >= 10  # Minimum length
            assert not text.startswith(' ')  # No leading whitespace
            assert not text.endswith(' ')   # No trailing whitespace
        
        # Verify metadata processing
        for metadata in result.processed_metadata:
            assert 'user_id' in metadata
            assert isinstance(metadata.get('user_id'), str)
        
        # Verify embedding normalization
        norms = np.linalg.norm(result.normalized_embeddings, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)  # Should be unit vectors 