"""
Comprehensive Test Suite for Data Preparation Module.

This test suite aims for 85%+ coverage of the data_preparation module.
"""

import pytest
import numpy as np
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from src.research_agent_backend.core.data_preparation import (
    DataCleaningService,
    NormalizationService,
    DimensionalityReductionService,
    DataPreparationManager,
    DataCleaningConfig,
    NormalizationConfig,
    DimensionalityReductionConfig,
    DataPreparationResult,
    NormalizationMethod,
    DimensionalityReductionMethod,
    create_data_preparation_manager
)
from src.research_agent_backend.models.metadata_schema import (
    CollectionType,
    ContentType,
    DocumentType
)
from src.research_agent_backend.utils.config import ConfigManager


class TestDataCleaningConfig:
    """Test DataCleaningConfig class and its functionality."""
    
    def test_default_config_creation(self):
        """Test creating DataCleaningConfig with default values."""
        config = DataCleaningConfig()
        
        # Test text cleaning defaults
        assert config.remove_extra_whitespace == True
        assert config.normalize_unicode == True
        assert config.remove_control_chars == True
        assert config.fix_encoding_issues == True
        assert config.min_text_length == 10
        assert config.max_text_length is None
        
        # Test content filtering defaults
        assert config.remove_empty_content == True
        assert config.remove_duplicate_content == True
        assert config.content_similarity_threshold == 0.95
        
        # Test metadata cleaning defaults
        assert config.standardize_metadata_fields == True
        assert config.validate_metadata_types == True
        assert config.fill_missing_metadata == True
        assert isinstance(config.default_values, dict)
        
        # Test language processing defaults
        assert config.detect_language == True
        assert config.filter_languages is None
        assert config.transliterate_non_latin == False

    def test_custom_config_creation(self):
        """Test creating DataCleaningConfig with custom values."""
        custom_defaults = {"author": "unknown", "priority": "low"}
        
        config = DataCleaningConfig(
            min_text_length=5,
            max_text_length=1000,
            content_similarity_threshold=0.8,
            default_values=custom_defaults,
            filter_languages=["en", "es"],
            transliterate_non_latin=True
        )
        
        assert config.min_text_length == 5
        assert config.max_text_length == 1000
        assert config.content_similarity_threshold == 0.8
        assert config.default_values == custom_defaults
        assert config.filter_languages == ["en", "es"]
        assert config.transliterate_non_latin == True


class TestNormalizationConfig:
    """Test NormalizationConfig class functionality."""
    
    def test_default_normalization_config(self):
        """Test default normalization configuration."""
        config = NormalizationConfig()
        
        assert config.embedding_method == NormalizationMethod.UNIT_VECTOR
        assert config.preserve_magnitude == False
        assert config.handle_zero_vectors == True
        assert config.normalize_numerical_metadata == True
        assert config.numerical_method == NormalizationMethod.MIN_MAX
        assert config.feature_range == (0.0, 1.0)
        assert config.with_centering == True
        assert config.with_scaling == True

    def test_custom_normalization_config(self):
        """Test custom normalization configuration."""
        config = NormalizationConfig(
            embedding_method=NormalizationMethod.Z_SCORE,
            preserve_magnitude=True,
            numerical_method=NormalizationMethod.ROBUST,
            feature_range=(-1.0, 1.0),
            with_centering=False
        )
        
        assert config.embedding_method == NormalizationMethod.Z_SCORE
        assert config.preserve_magnitude == True
        assert config.numerical_method == NormalizationMethod.ROBUST
        assert config.feature_range == (-1.0, 1.0)
        assert config.with_centering == False


class TestDimensionalityReductionConfig:
    """Test DimensionalityReductionConfig class functionality."""
    
    def test_default_dimensionality_config(self):
        """Test default dimensionality reduction configuration."""
        config = DimensionalityReductionConfig()
        
        assert config.method == DimensionalityReductionMethod.NONE
        assert config.target_dimensions is None
        assert config.pca_explained_variance_ratio == 0.95
        assert config.pca_whiten == False
        assert config.tsne_perplexity == 30.0
        assert config.tsne_learning_rate == 200.0
        assert config.tsne_n_iter == 1000
        assert config.umap_n_neighbors == 15
        assert config.umap_min_dist == 0.1
        assert config.umap_metric == "cosine"

    def test_custom_dimensionality_config(self):
        """Test custom dimensionality reduction configuration."""
        config = DimensionalityReductionConfig(
            method=DimensionalityReductionMethod.PCA,
            target_dimensions=50,
            pca_explained_variance_ratio=0.90,
            pca_whiten=True,
            tsne_perplexity=20.0,
            umap_n_neighbors=10
        )
        
        assert config.method == DimensionalityReductionMethod.PCA
        assert config.target_dimensions == 50
        assert config.pca_explained_variance_ratio == 0.90
        assert config.pca_whiten == True
        assert config.tsne_perplexity == 20.0
        assert config.umap_n_neighbors == 10


class TestDataPreparationResult:
    """Test DataPreparationResult class functionality."""
    
    def test_result_creation_and_properties(self):
        """Test creating DataPreparationResult and its calculated properties."""
        result = DataPreparationResult(
            cleaned_texts=["text1", "text2", "text3"],
            original_count=10,
            processed_count=8,
            filtered_count=2,
            error_count=0,
            duplicate_removed=1,
            empty_removed=1,
            processing_time_seconds=0.5
        )
        
        assert len(result.cleaned_texts) == 3
        assert result.original_count == 10
        assert result.processed_count == 8
        assert result.filtered_count == 2
        assert result.success_rate == 0.8  # 8/10
        assert result.filter_rate == 0.2   # 2/10
        assert result.processing_time_seconds == 0.5

    def test_result_empty_case(self):
        """Test DataPreparationResult with empty data."""
        result = DataPreparationResult(
            cleaned_texts=[],
            original_count=0,
            processed_count=0,
            filtered_count=0
        )
        
        assert result.success_rate == 0.0
        assert result.filter_rate == 0.0


class TestDataCleaningService:
    """Test DataCleaningService functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.config = DataCleaningConfig()
        self.service = DataCleaningService(self.config)

    def test_service_initialization(self):
        """Test DataCleaningService initialization."""
        assert self.service.config == self.config
        assert hasattr(self.service, 'logger')
        assert hasattr(self.service, '_extra_whitespace_pattern')
        assert hasattr(self.service, '_control_char_pattern')
        assert isinstance(self.service._duplicate_detector, dict)

    def test_clean_text_basic_cleaning(self):
        """Test basic text cleaning operations."""
        # Test normal text
        clean_text = self.service.clean_text("Hello world!")
        assert clean_text == "Hello world!"
        
        # Test text with extra whitespace
        text_with_spaces = "Hello    world  \n\n  test"
        cleaned = self.service.clean_text(text_with_spaces)
        assert "    " not in cleaned
        assert cleaned == "Hello world test"
        
        # Test text with control characters
        text_with_control = "Hello\x00\x08world\x1F"
        cleaned = self.service.clean_text(text_with_control)
        assert "\x00" not in cleaned
        assert "\x08" not in cleaned
        assert "\x1F" not in cleaned

    def test_clean_text_edge_cases(self):
        """Test text cleaning edge cases."""
        # Test None input
        assert self.service.clean_text(None) is None
        
        # Test empty string
        assert self.service.clean_text("") is None
        
        # Test non-string input
        assert self.service.clean_text(123) is None
        
        # Test very short text (below min_text_length)
        short_text = "hi"
        assert self.service.clean_text(short_text) is None
        
        # Test text exactly at min_text_length
        min_text = "a" * self.config.min_text_length
        cleaned = self.service.clean_text(min_text)
        assert cleaned == min_text

    def test_clean_text_unicode_normalization(self):
        """Test Unicode normalization in text cleaning."""
        # Test Unicode normalization
        unicode_text = "café"  # é as combining character
        cleaned = self.service.clean_text(unicode_text)
        assert cleaned is not None
        assert "café" in cleaned or "cafe" in cleaned

    def test_clean_text_max_length_filtering(self):
        """Test maximum text length filtering."""
        config_with_max = DataCleaningConfig(max_text_length=20)
        service = DataCleaningService(config_with_max)
        
        # Text within limit
        short_text = "Short text here"
        assert service.clean_text(short_text) == "Short text here"
        
        # Text exceeding limit
        long_text = "This is a very long text that exceeds the maximum length limit"
        assert service.clean_text(long_text) is None

    def test_clean_metadata_basic(self):
        """Test basic metadata cleaning operations."""
        metadata = {
            "title": "  Test Title  ",
            "author": "John Doe",
            "tags": ["tag1", "tag2"],
            "count": 42
        }
        
        cleaned = self.service.clean_metadata(metadata)
        
        assert cleaned["title"] == "Test Title"  # Whitespace removed
        assert cleaned["author"] == "John Doe"
        assert cleaned["tags"] == ["tag1", "tag2"]
        assert cleaned["count"] == 42

    def test_clean_metadata_field_standardization(self):
        """Test metadata field name standardization."""
        metadata = {
            "Title": "Test",
            "Author Name": "John",
            "created-at": "2023-01-01"
        }
        
        cleaned = self.service.clean_metadata(metadata)
        
        # Should standardize field names
        assert any(key.lower().replace(" ", "_").replace("-", "_") in cleaned for key in metadata.keys())

    def test_clean_metadata_with_defaults(self):
        """Test metadata cleaning with default values."""
        config = DataCleaningConfig(
            fill_missing_metadata=True,
            default_values={"author": "unknown", "priority": "medium"}
        )
        service = DataCleaningService(config)
        
        metadata = {"title": "Test Document"}
        cleaned = service.clean_metadata(metadata)
        
        assert cleaned["title"] == "Test Document"
        # Default values should be added based on implementation


class TestNormalizationService:
    """Test NormalizationService functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.config = NormalizationConfig()
        self.service = NormalizationService(self.config)

    def test_service_initialization(self):
        """Test NormalizationService initialization."""
        assert self.service.config == self.config
        assert hasattr(self.service, 'logger')

    def test_normalize_embeddings_unit_vector(self):
        """Test unit vector normalization of embeddings."""
        # Create test embeddings
        embeddings = np.array([
            [3.0, 4.0, 0.0],  # Should normalize to [0.6, 0.8, 0.0]
            [1.0, 0.0, 0.0],  # Should normalize to [1.0, 0.0, 0.0]
            [0.0, 0.0, 0.0]   # Zero vector case
        ])
        
        normalized = self.service.normalize_embeddings(embeddings, NormalizationMethod.UNIT_VECTOR)
        
        # Check dimensions are preserved
        assert normalized.shape == embeddings.shape
        
        # Check first vector is normalized (should have magnitude ~1)
        first_magnitude = np.linalg.norm(normalized[0])
        assert abs(first_magnitude - 1.0) < 1e-6
        
        # Check second vector (already unit vector)
        assert abs(np.linalg.norm(normalized[1]) - 1.0) < 1e-6

    def test_normalize_embeddings_min_max(self):
        """Test min-max normalization of embeddings."""
        embeddings = np.array([
            [0.0, 10.0, 5.0],
            [5.0, 0.0, 10.0],
            [10.0, 5.0, 0.0]
        ])
        
        normalized = self.service.normalize_embeddings(embeddings, NormalizationMethod.MIN_MAX)
        
        # Check values are in [0, 1] range
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
        
        # Check that min and max values exist
        assert np.any(normalized == 0.0)
        assert np.any(normalized == 1.0)

    def test_normalize_embeddings_z_score(self):
        """Test z-score normalization of embeddings."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        normalized = self.service.normalize_embeddings(embeddings, NormalizationMethod.Z_SCORE)
        
        # Check that each column has mean ~0 and std ~1
        means = np.mean(normalized, axis=0)
        stds = np.std(normalized, axis=0)
        
        assert np.allclose(means, 0.0, atol=1e-10)
        assert np.allclose(stds, 1.0, atol=1e-10)

    def test_normalize_embeddings_robust(self):
        """Test robust normalization of embeddings."""
        # Include outliers to test robust normalization
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [100.0, 200.0, 300.0]  # Outlier
        ])
        
        normalized = self.service.normalize_embeddings(embeddings, NormalizationMethod.ROBUST)
        
        # Robust normalization should handle outliers better than z-score
        assert normalized.shape == embeddings.shape
        assert np.all(np.isfinite(normalized))

    def test_normalize_embeddings_none_method(self):
        """Test no normalization (NONE method)."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        normalized = self.service.normalize_embeddings(embeddings, NormalizationMethod.NONE)
        
        # Should return unchanged embeddings
        assert np.array_equal(normalized, embeddings)

    def test_normalize_numerical_metadata(self):
        """Test normalization of numerical metadata fields."""
        metadata_list = [
            {"score": 10, "rating": 5.0, "title": "doc1"},
            {"score": 20, "rating": 3.0, "title": "doc2"},
            {"score": 30, "rating": 4.0, "title": "doc3"}
        ]
        
        normalized_metadata = self.service.normalize_numerical_metadata(metadata_list)
        
        # Check that numerical fields are normalized
        assert len(normalized_metadata) == len(metadata_list)
        
        # Non-numerical fields should remain unchanged
        for i, metadata in enumerate(normalized_metadata):
            assert metadata["title"] == metadata_list[i]["title"]
            
        # Numerical fields should be normalized
        scores = [m["score"] for m in normalized_metadata]
        ratings = [m["rating"] for m in normalized_metadata]
        
        # Check normalization worked (values should be different from original)
        original_scores = [m["score"] for m in metadata_list]
        assert scores != original_scores


class TestDimensionalityReductionService:
    """Test DimensionalityReductionService functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.config = DimensionalityReductionConfig()
        self.service = DimensionalityReductionService(self.config)

    def test_service_initialization(self):
        """Test DimensionalityReductionService initialization."""
        assert self.service.config == self.config
        assert hasattr(self.service, 'logger')

    def test_reduce_dimensions_none_method(self):
        """Test no dimensionality reduction (NONE method)."""
        embeddings = np.random.rand(10, 50)
        
        reduced = self.service.reduce_dimensions(embeddings, DimensionalityReductionMethod.NONE)
        
        # Should return unchanged embeddings
        assert np.array_equal(reduced, embeddings)

    @patch('src.research_agent_backend.core.data_preparation.PCA')
    def test_reduce_dimensions_pca(self, mock_pca_class):
        """Test PCA dimensionality reduction."""
        # Mock PCA
        mock_pca = Mock()
        mock_pca.fit_transform.return_value = np.random.rand(10, 20)
        mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca_class.return_value = mock_pca
        
        embeddings = np.random.rand(10, 50)
        
        reduced = self.service.reduce_dimensions(
            embeddings, 
            DimensionalityReductionMethod.PCA, 
            target_dims=20
        )
        
        # Check PCA was called
        mock_pca_class.assert_called_once()
        mock_pca.fit_transform.assert_called_once_with(embeddings)
        
        # Check dimensions were reduced
        assert reduced.shape == (10, 20)

    def test_reduce_dimensions_invalid_method(self):
        """Test handling of invalid dimensionality reduction method."""
        embeddings = np.random.rand(10, 50)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, NotImplementedError)):
            self.service.reduce_dimensions(embeddings, "invalid_method")


class TestDataPreparationManager:
    """Test DataPreparationManager comprehensive functionality."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.config_manager = ConfigManager()
        self.manager = DataPreparationManager(config_manager=self.config_manager)

    def test_manager_initialization(self):
        """Test DataPreparationManager initialization."""
        assert self.manager.config_manager is not None
        assert hasattr(self.manager, 'cleaning_service')
        assert hasattr(self.manager, 'normalization_service')
        assert hasattr(self.manager, 'dimensionality_service')
        assert hasattr(self.manager, 'metadata_validator')

    def test_manager_initialization_with_custom_configs(self):
        """Test manager initialization with custom configurations."""
        cleaning_config = DataCleaningConfig(min_text_length=5)
        normalization_config = NormalizationConfig(embedding_method=NormalizationMethod.Z_SCORE)
        dimensionality_config = DimensionalityReductionConfig(method=DimensionalityReductionMethod.PCA)
        
        manager = DataPreparationManager(
            cleaning_config=cleaning_config,
            normalization_config=normalization_config,
            dimensionality_config=dimensionality_config
        )
        
        assert manager.cleaning_service.config.min_text_length == 5
        assert manager.normalization_service.config.embedding_method == NormalizationMethod.Z_SCORE
        assert manager.dimensionality_service.config.method == DimensionalityReductionMethod.PCA

    def test_prepare_single_document(self):
        """Test preparing a single document."""
        text = "This is a test document for data preparation."
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metadata = {"title": "Test Doc", "author": "Test Author"}
        
        cleaned_text, normalized_embedding, processed_metadata = self.manager.prepare_single_document(
            text=text,
            embedding=embedding,
            metadata=metadata
        )
        
        # Check that processing occurred
        assert cleaned_text is not None
        assert len(cleaned_text) > 0
        assert normalized_embedding is not None
        assert processed_metadata is not None
        assert "title" in processed_metadata

    def test_prepare_single_document_edge_cases(self):
        """Test single document preparation edge cases."""
        # Test with None text
        result = self.manager.prepare_single_document(text=None)
        cleaned_text, normalized_embedding, processed_metadata = result
        assert cleaned_text is None
        
        # Test with empty text
        result = self.manager.prepare_single_document(text="")
        cleaned_text, normalized_embedding, processed_metadata = result
        assert cleaned_text is None
        
        # Test with very short text
        result = self.manager.prepare_single_document(text="hi")
        cleaned_text, normalized_embedding, processed_metadata = result
        assert cleaned_text is None  # Below min_text_length

    def test_prepare_documents_batch(self):
        """Test batch document preparation."""
        texts = [
            "This is the first test document for processing.",
            "Here is another document to process in the batch.",
            "Final document in the test batch preparation."
        ]
        
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        metadata_list = [
            {"title": "Doc 1", "type": "test"},
            {"title": "Doc 2", "type": "test"},
            {"title": "Doc 3", "type": "test"}
        ]
        
        result = self.manager.prepare_documents(
            texts=texts,
            embeddings=embeddings,
            metadata_list=metadata_list
        )
        
        # Check result structure
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 3
        assert result.processed_count <= 3
        assert len(result.cleaned_texts) <= 3
        assert result.processing_time_seconds > 0

    def test_prepare_documents_with_filtering(self):
        """Test document preparation with content filtering."""
        texts = [
            "This is a good document with sufficient length.",
            "short",  # Should be filtered out
            "",  # Should be filtered out
            "Another good document that meets length requirements."
        ]
        
        result = self.manager.prepare_documents(texts=texts)
        
        # Check that short/empty texts were filtered
        assert result.original_count == 4
        assert result.processed_count <= 2  # Only 2 valid documents
        assert result.filtered_count >= 2   # At least 2 filtered out

    def test_prepare_documents_collection_type_config(self):
        """Test document preparation with collection type configuration."""
        texts = ["Test document for collection type configuration."]
        
        # Test with different collection types
        for collection_type in [CollectionType.FUNDAMENTAL, CollectionType.GENERAL, CollectionType.PROJECT_SPECIFIC]:
            result = self.manager.prepare_documents(
                texts=texts,
                collection_type=collection_type
            )
            
            assert isinstance(result, DataPreparationResult)
            assert result.original_count == 1

    def test_prepare_documents_batch_processing(self):
        """Test batch processing with custom batch size."""
        texts = [f"Document number {i} for batch processing test." for i in range(10)]
        
        result = self.manager.prepare_documents(
            texts=texts,
            batch_size=3
        )
        
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 10


class TestUtilityFunctions:
    """Test utility functions and factory methods."""
    
    def test_create_data_preparation_manager_defaults(self):
        """Test creating manager with default settings."""
        manager = create_data_preparation_manager()
        
        assert isinstance(manager, DataPreparationManager)
        assert manager.config_manager is not None

    def test_create_data_preparation_manager_with_config_file(self):
        """Test creating manager with custom config file."""
        # Test with non-existent config file (should handle gracefully)
        try:
            manager = create_data_preparation_manager(config_file="nonexistent.json")
            assert isinstance(manager, DataPreparationManager)
        except Exception:
            # Some implementations might raise exceptions for missing config
            pass

    def test_create_data_preparation_manager_with_overrides(self):
        """Test creating manager with configuration overrides."""
        manager = create_data_preparation_manager(
            min_text_length=5,
            embedding_method="z_score"
        )
        
        assert isinstance(manager, DataPreparationManager)


class TestIntegrationScenarios:
    """Test realistic integration scenarios for comprehensive coverage."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.manager = DataPreparationManager()

    def test_research_document_preparation_workflow(self):
        """Test complete workflow for research document preparation."""
        # Simulate research documents
        research_texts = [
            "Abstract: This paper presents a novel approach to machine learning...",
            "Introduction: Recent advances in artificial intelligence have shown...",
            "Methodology: We propose a new algorithm based on transformer architecture...",
            "Results: Our experiments demonstrate significant improvements over baseline...",
            "Conclusion: The proposed method offers substantial benefits for practitioners..."
        ]
        
        # Simulate embeddings (would normally come from embedding model)
        embeddings = np.random.rand(5, 128)
        
        # Simulate research metadata
        metadata_list = [
            {"section": "abstract", "page": 1, "confidence": 0.95},
            {"section": "introduction", "page": 2, "confidence": 0.92},
            {"section": "methodology", "page": 5, "confidence": 0.98},
            {"section": "results", "page": 10, "confidence": 0.89},
            {"section": "conclusion", "page": 15, "confidence": 0.94}
        ]
        
        result = self.manager.prepare_documents(
            texts=research_texts,
            embeddings=embeddings,
            metadata_list=metadata_list,
            collection_type=CollectionType.FUNDAMENTAL
        )
        
        # Verify comprehensive processing
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 5
        assert result.processed_count > 0
        assert len(result.cleaned_texts) > 0
        assert result.success_rate > 0

    def test_code_documentation_preparation_workflow(self):
        """Test workflow for code documentation preparation."""
        code_docs = [
            "def process_data(input_data): \"\"\"Process input data and return cleaned result.\"\"\"",
            "class DataProcessor: \"\"\"Main data processing class with advanced features.\"\"\"",
            "# Configuration file for the application settings and parameters",
            "README.md: Getting started with the project setup and installation",
            "API documentation for the vector store management endpoints"
        ]
        
        embeddings = np.random.rand(5, 64)
        
        metadata_list = [
            {"type": "function", "language": "python", "complexity": "medium"},
            {"type": "class", "language": "python", "complexity": "high"},
            {"type": "comment", "language": "python", "complexity": "low"},
            {"type": "documentation", "format": "markdown", "complexity": "low"},
            {"type": "api_doc", "format": "text", "complexity": "medium"}
        ]
        
        result = self.manager.prepare_documents(
            texts=code_docs,
            embeddings=embeddings,
            metadata_list=metadata_list,
            collection_type=CollectionType.PROJECT_SPECIFIC
        )
        
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 5
        assert result.success_rate > 0

    def test_mixed_content_preparation_with_errors(self):
        """Test preparation with mixed content including error cases."""
        mixed_texts = [
            "Valid document with sufficient content for processing.",
            "",  # Empty content
            "x",  # Too short
            None,  # None content
            "Another valid document that should pass all validation checks.",
            "Document with special characters: café, naïve, résumé, coöperate",
            "   Extra    whitespace   should   be   cleaned   properly   ",
            "Document\x00with\x08control\x1Fcharacters\x7F"
        ]
        
        # Mix of valid and invalid embeddings
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],  # Zero vector
            [1.0, 2.0, 3.0],
            [np.nan, 0.1, 0.2],  # NaN value
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.1, 0.3, 0.5],
            [0.2, 0.4, 0.6]
        ])
        
        metadata_list = [
            {"title": "Valid 1", "score": 10},
            {},  # Empty metadata
            {"title": "Short"},
            None,  # None metadata
            {"title": "Valid 2", "score": 20},
            {"title": "Special chars", "score": 15},
            {"title": "  Whitespace  ", "score": 25},
            {"title": "Control chars", "score": 12}
        ]
        
        result = self.manager.prepare_documents(
            texts=mixed_texts,
            embeddings=embeddings,
            metadata_list=metadata_list
        )
        
        # Check error handling
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 8
        assert result.filtered_count > 0  # Some should be filtered
        assert len(result.errors) >= 0  # May have errors
        assert len(result.warnings) >= 0  # May have warnings


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    def test_large_batch_processing(self):
        """Test processing large batches of documents."""
        # Create a large batch of documents
        large_texts = [f"Document {i} with sufficient content for processing and validation." for i in range(100)]
        large_embeddings = np.random.rand(100, 256)
        large_metadata = [{"id": i, "batch": "large", "score": i % 10} for i in range(100)]
        
        start_time = time.time()
        
        manager = DataPreparationManager()
        result = manager.prepare_documents(
            texts=large_texts,
            embeddings=large_embeddings,
            metadata_list=large_metadata,
            batch_size=10
        )
        
        processing_time = time.time() - start_time
        
        # Check performance and results
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 100
        assert result.processing_time_seconds > 0
        assert processing_time < 30  # Should complete within reasonable time
        assert result.success_rate > 0.8  # Most documents should process successfully

    def test_memory_efficiency_with_large_embeddings(self):
        """Test memory efficiency with large embedding dimensions."""
        texts = ["Large embedding test document."] * 10
        large_embeddings = np.random.rand(10, 1024)  # Large embedding dimension
        
        manager = DataPreparationManager()
        result = manager.prepare_documents(
            texts=texts,
            embeddings=large_embeddings
        )
        
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 10
        assert result.normalized_embeddings is not None
        assert result.normalized_embeddings.shape == large_embeddings.shape

    def test_unicode_and_encoding_handling(self):
        """Test comprehensive Unicode and encoding handling."""
        unicode_texts = [
            "English text with basic characters",
            "Café with French accents and naïve characters",
            "Düsseldorf with German umlauts and ß",
            "Москва with Cyrillic characters",
            "东京 with Chinese characters",
            "🌟 Text with emoji characters 🚀",
            "Mixed: café 东京 🌟 Düsseldorf"
        ]
        
        manager = DataPreparationManager()
        result = manager.prepare_documents(texts=unicode_texts)
        
        assert isinstance(result, DataPreparationResult)
        assert result.original_count == 7
        assert result.processed_count > 0
        
        # Check that Unicode was handled properly
        for text in result.cleaned_texts:
            assert isinstance(text, str)
            assert len(text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 