"""
Test suite for knowledge base augmentation service implementation following TDD approach.

Tests cover:
- External result addition with source attribution
- Research report ingestion and categorization
- Document updating with version tracking
- Duplicate detection and merging
- Quality validation for new content
- Collection assignment logic

This follows TDD Red-Green-Refactor methodology.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import tempfile
from pathlib import Path

from research_agent_backend.core.augmentation_service import (
    AugmentationService,
    ExternalResult,
    ResearchReport,
    DocumentUpdate,
    DuplicateGroup,
    QualityMetrics,
    AugmentationConfig
)
from research_agent_backend.models.metadata_schema import DocumentMetadata, DocumentType
from research_agent_backend.core.integration_pipeline.models import SearchResult
from research_agent_backend.utils.config import ConfigManager


class TestAugmentationService:
    """Test suite for AugmentationService following TDD methodology."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with augmentation settings."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            'augmentation.quality_threshold': 0.7,
            'augmentation.similarity_threshold': 0.85,
            'augmentation.auto_categorize': True,
            'augmentation.enable_versioning': True,
            'augmentation.batch_size': 50,
            'augmentation.cache_size': 1000,
            'augmentation.default_collection': 'research'
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        vector_store = Mock()
        vector_store.add_documents.return_value = {'success': True, 'ids': ['doc_001']}
        vector_store.search.return_value = []
        vector_store.get_document.return_value = None
        vector_store.update_document.return_value = {'success': True}
        vector_store.delete_document.return_value = {'success': True}
        return vector_store
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing."""
        embedding_service = Mock()
        embedding_service.embed_text.return_value = [0.1] * 384
        embedding_service.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
        return embedding_service
    
    @pytest.fixture
    def sample_external_result(self):
        """Sample external result for testing."""
        return ExternalResult(
            source_url="https://example.com/article",
            title="Machine Learning Advances",
            content="Recent advances in machine learning have led to breakthrough results...",
            author="John Doe",
            publication_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tags=["AI", "ML", "research"],
            metadata={"journal": "AI Review", "doi": "10.1000/123456"}
        )

    # RED PHASE: Service Initialization and Configuration Tests

    def test_augmentation_service_initialization_with_dependencies(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test AugmentationService initializes properly with all dependencies."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        assert service.config_manager == mock_config_manager
        assert service.vector_store == mock_vector_store
        assert service.embedding_service == mock_embedding_service
        assert service.quality_threshold == 0.7
        assert service.similarity_threshold == 0.85

    def test_augmentation_config_creation_and_validation(self):
        """Test AugmentationConfig dataclass creation and validation."""
        config = AugmentationConfig(
            quality_threshold=0.8,
            similarity_threshold=0.9,
            auto_categorize=False,
            enable_versioning=True,
            batch_size=100
        )
        
        assert config.quality_threshold == 0.8
        assert config.similarity_threshold == 0.9
        assert config.auto_categorize == False
        assert config.enable_versioning == True
        assert config.batch_size == 100

    def test_augmentation_config_default_values(self):
        """Test AugmentationConfig has appropriate default values."""
        config = AugmentationConfig()
        
        assert config.quality_threshold == 0.7
        assert config.similarity_threshold == 0.85
        assert config.auto_categorize == True
        assert config.enable_versioning == True
        assert config.batch_size == 50
        assert config.default_collection == "research"

    # RED PHASE: External Result Addition Tests

    def test_add_external_result_basic_functionality(self, mock_config_manager, mock_vector_store, mock_embedding_service, sample_external_result):
        """Test adding a basic external result with proper metadata extraction."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        result = service.add_external_result(sample_external_result, collection="research")
        
        assert result['status'] == 'success'
        assert 'document_id' in result
        assert result['collection'] == 'research'
        assert result['source_attribution'] in result
        mock_vector_store.add_documents.assert_called_once()
        mock_embedding_service.embed_text.assert_called()

    def test_add_external_result_with_source_attribution(self, mock_config_manager, mock_vector_store, mock_embedding_service, sample_external_result):
        """Test external result addition includes proper source attribution."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        result = service.add_external_result(sample_external_result)
        
        # Should include source attribution metadata
        assert result['source_attribution']['url'] == "https://example.com/article"
        assert result['source_attribution']['author'] == "John Doe"
        assert result['source_attribution']['added_date'] is not None
        assert result['source_attribution']['type'] == "external"

    def test_add_external_result_quality_validation_pass(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test external result passes quality validation."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        high_quality_result = ExternalResult(
            source_url="https://arxiv.org/abs/2301.12345",
            title="Novel Deep Learning Architecture for Natural Language Processing",
            content="This paper presents a comprehensive analysis of transformer architectures with detailed experimental results...",
            author="Dr. Jane Smith",
            publication_date=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )
        
        with patch.object(service, '_calculate_quality_score', return_value=0.85):
            result = service.add_external_result(high_quality_result)
            
            assert result['status'] == 'success'
            assert result['quality_score'] == 0.85
            assert result['quality_passed'] == True

    def test_add_external_result_quality_validation_fail(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test external result fails quality validation."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        low_quality_result = ExternalResult(
            source_url="https://example.com",
            title="Test",
            content="Short content",
            author="",
            publication_date=None
        )
        
        with patch.object(service, '_calculate_quality_score', return_value=0.4):
            result = service.add_external_result(low_quality_result)
            
            assert result['status'] == 'rejected'
            assert result['quality_score'] == 0.4
            assert result['quality_passed'] == False
            assert 'rejection_reason' in result

    def test_add_external_result_automatic_collection_assignment(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test automatic collection assignment based on content analysis."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        ml_result = ExternalResult(
            source_url="https://example.com/ml-article",
            title="Machine Learning Tutorial",
            content="This tutorial covers neural networks, deep learning, and supervised learning algorithms...",
            tags=["machine-learning", "neural-networks"]
        )
        
        with patch.object(service, '_auto_assign_collection', return_value='machine-learning'):
            result = service.add_external_result(ml_result)
            
            assert result['collection'] == 'machine-learning'
            assert result['auto_assigned'] == True
            assert 'assignment_confidence' in result

    def test_add_external_result_duplicate_detection(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test duplicate detection during external result addition."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        # Mock existing similar document
        mock_vector_store.search.return_value = [
            {'id': 'existing_doc', 'score': 0.92, 'content': 'Very similar content...'}
        ]
        
        duplicate_result = ExternalResult(
            source_url="https://example.com/duplicate",
            title="Similar Article",
            content="Very similar content to existing document...",
        )
        
        result = service.add_external_result(duplicate_result, detect_duplicates=True)
        
        assert result['status'] == 'duplicate_detected'
        assert 'similar_documents' in result
        assert result['similarity_score'] >= 0.85
        assert 'merge_suggestion' in result

    # RED PHASE: Research Report Ingestion Tests

    def test_add_research_report_basic_functionality(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test adding a basic research report with chunking and embedding."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Research Report\n\nThis is a comprehensive research report on artificial intelligence...")
            temp_path = f.name
        
        try:
            research_report = ResearchReport(
                file_path=temp_path,
                category="literature-review",
                metadata={"project": "AI Research"}
            )
            
            with patch.object(service, '_chunk_document', return_value=['chunk1', 'chunk2', 'chunk3']):
                result = service.add_research_report(research_report, collection="research-reports")
                
                assert result['status'] == 'success'
                assert 'document_id' in result
                assert result['chunks_created'] >= 3
                assert result['collection'] == 'research-reports'
                mock_vector_store.add_documents.assert_called()
        finally:
            Path(temp_path).unlink()

    def test_add_research_report_with_auto_categorization(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test research report with automatic categorization."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Machine Learning Research\n\nThis report analyzes neural network architectures...")
            temp_path = f.name
        
        try:
            research_report = ResearchReport(file_path=temp_path)
            
            with patch.object(service, '_auto_categorize_content', return_value=('machine-learning', 0.87)):
                with patch.object(service, '_chunk_document', return_value=['chunk1']):
                    result = service.add_research_report(research_report, auto_categorize=True)
                    
                    assert result['auto_category'] == 'machine-learning'
                    assert result['categorization_confidence'] == 0.87
                    assert result['auto_categorized'] == True
        finally:
            Path(temp_path).unlink()

    def test_add_research_reports_batch_processing(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test batch processing of multiple research reports."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "report1.md").write_text("# Report 1\n\nContent 1...")
            (temp_path / "report2.md").write_text("# Report 2\n\nContent 2...")
            (temp_path / "report3.md").write_text("# Report 3\n\nContent 3...")
            
            with patch.object(service, '_chunk_document', return_value=['chunk1']):
                result = service.add_research_reports_batch(
                    folder_path=str(temp_path),
                    pattern="*.md",
                    collection="batch-reports"
                )
                
                assert result['processed'] == 3
                assert result['successful'] >= 2
                assert 'results' in result
                assert isinstance(result['results'], list)

    # RED PHASE: Document Update Tests

    def test_update_document_content_only(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test updating document content without metadata changes."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        mock_vector_store.get_document.return_value = {
            'id': 'doc_001',
            'content': 'Old content...',
            'metadata': {'title': 'Original Title', 'version': 1}
        }
        
        update = DocumentUpdate(
            document_id='doc_001',
            new_content='Updated content with new information...',
            update_embeddings=True
        )
        
        result = service.update_document(update)
        
        assert result['status'] == 'updated'
        assert result['document_id'] == 'doc_001'
        assert result['version'] == 2
        assert 'content_updated' in result['changes']
        assert 'embeddings_regenerated' in result['changes']
        mock_vector_store.update_document.assert_called_once()

    def test_update_document_metadata_only(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test updating document metadata without content changes."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        mock_vector_store.get_document.return_value = {
            'id': 'doc_002',
            'content': 'Existing content...',
            'metadata': {'title': 'Old Title', 'version': 1}
        }
        
        update = DocumentUpdate(
            document_id='doc_002',
            new_metadata={'title': 'Updated Title', 'tags': ['updated', 'metadata']},
            update_embeddings=False
        )
        
        result = service.update_document(update)
        
        assert result['status'] == 'updated'
        assert 'metadata_updated' in result['changes']
        assert 'embeddings_regenerated' not in result['changes']

    def test_update_document_from_file(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test updating document content from file."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        mock_vector_store.get_document.return_value = {
            'id': 'doc_003',
            'content': 'Old content...',
            'metadata': {'title': 'Document', 'version': 1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Updated Document\n\nThis is the new content from file...")
            temp_path = f.name
        
        try:
            update = DocumentUpdate(
                document_id='doc_003',
                source_file=temp_path
            )
            
            result = service.update_document(update)
            
            assert result['status'] == 'updated'
            assert 'content_updated_from_file' in result['changes']
        finally:
            Path(temp_path).unlink()

    def test_update_nonexistent_document_error(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test error handling when updating nonexistent document."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        mock_vector_store.get_document.return_value = None
        
        update = DocumentUpdate(
            document_id='nonexistent_doc',
            new_content='New content...'
        )
        
        result = service.update_document(update)
        
        assert result['status'] == 'error'
        assert 'not found' in result['error'].lower()

    # RED PHASE: Duplicate Detection and Merging Tests

    def test_detect_duplicates_automatic_detection(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test automatic duplicate detection across collections."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        # Mock similar documents
        mock_vector_store.get_all_documents.return_value = [
            {'id': 'doc_001', 'content': 'Python programming tutorial...'},
            {'id': 'doc_002', 'content': 'Python programming guide...'},
            {'id': 'doc_003', 'content': 'JavaScript tutorial...'}
        ]
        
        with patch.object(service, '_calculate_similarity', side_effect=[0.95, 0.3, 0.2]):
            duplicates = service.detect_duplicates(threshold=0.9)
            
            assert len(duplicates) == 1
            assert duplicates[0]['group_id'] == 1
            assert 'doc_001' in duplicates[0]['documents']
            assert 'doc_002' in duplicates[0]['documents']
            assert duplicates[0]['similarity'] == 0.95

    def test_detect_duplicates_manual_selection(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test duplicate detection for manually selected documents."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        document_ids = ['doc_001', 'doc_002', 'doc_003']
        mock_vector_store.get_documents.return_value = [
            {'id': 'doc_001', 'content': 'Content A...'},
            {'id': 'doc_002', 'content': 'Content A similar...'},
            {'id': 'doc_003', 'content': 'Different content...'}
        ]
        
        with patch.object(service, '_calculate_similarity', side_effect=[0.88, 0.4, 0.35]):
            duplicates = service.detect_duplicates(document_ids=document_ids, threshold=0.85)
            
            assert len(duplicates) == 1
            assert duplicates[0]['similarity'] >= 0.85

    def test_merge_duplicates_content_union_strategy(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test merging duplicates using content union strategy."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        duplicate_group = DuplicateGroup(
            group_id=1,
            documents=['doc_001', 'doc_002'],
            similarity=0.93,
            merge_strategy='content_union'
        )
        
        mock_vector_store.get_documents.return_value = [
            {'id': 'doc_001', 'content': 'Section A content...', 'metadata': {'title': 'Doc 1'}},
            {'id': 'doc_002', 'content': 'Section B content...', 'metadata': {'title': 'Doc 2'}}
        ]
        
        result = service.merge_duplicates([duplicate_group], keep_originals=False)
        
        assert result['merged_groups'] == 1
        assert result['documents_merged'] == 2
        assert 'new_document_id' in result
        mock_vector_store.add_documents.assert_called()  # For merged document
        mock_vector_store.delete_document.assert_called()  # For originals

    def test_merge_duplicates_latest_version_strategy(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test merging duplicates using latest version strategy."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        duplicate_group = DuplicateGroup(
            group_id=1,
            documents=['doc_001', 'doc_002'],
            similarity=0.90,
            merge_strategy='latest_version'
        )
        
        mock_vector_store.get_documents.return_value = [
            {'id': 'doc_001', 'content': 'Older content...', 'metadata': {'created': '2024-01-01'}},
            {'id': 'doc_002', 'content': 'Newer content...', 'metadata': {'created': '2024-01-15'}}
        ]
        
        result = service.merge_duplicates([duplicate_group])
        
        assert result['merged_groups'] == 1
        assert 'latest_version_kept' in result

    def test_merge_duplicates_preview_mode(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test duplicate merging in preview mode without actual changes."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        duplicate_groups = [
            DuplicateGroup(group_id=1, documents=['doc_001', 'doc_002'], similarity=0.93),
            DuplicateGroup(group_id=2, documents=['doc_003', 'doc_004'], similarity=0.88)
        ]
        
        result = service.merge_duplicates(duplicate_groups, preview_only=True)
        
        assert result['preview_mode'] == True
        assert result['would_merge_groups'] == 2
        assert result['would_merge_documents'] == 4
        # Should not call actual merge operations
        mock_vector_store.add_documents.assert_not_called()
        mock_vector_store.delete_document.assert_not_called()

    # RED PHASE: Quality Validation Tests

    def test_calculate_quality_score_high_quality_content(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test quality score calculation for high-quality content."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        high_quality_content = """
        This comprehensive research paper presents a novel approach to natural language processing
        using transformer architectures. The methodology includes detailed experimental design,
        statistical analysis, and peer-reviewed validation. The results demonstrate significant
        improvements over existing baselines with proper citation and reproducible code.
        """
        
        quality_metrics = service._calculate_quality_score(
            content=high_quality_content,
            metadata={'author': 'Dr. Smith', 'journal': 'Nature AI', 'peer_reviewed': True}
        )
        
        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.overall_score >= 0.8
        assert quality_metrics.content_length_score > 0.7
        assert quality_metrics.structure_score > 0.7
        assert quality_metrics.credibility_score > 0.8

    def test_calculate_quality_score_low_quality_content(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test quality score calculation for low-quality content."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        low_quality_content = "Short text. No details."
        
        quality_metrics = service._calculate_quality_score(
            content=low_quality_content,
            metadata={'author': '', 'source': 'unknown'}
        )
        
        assert quality_metrics.overall_score < 0.5
        assert quality_metrics.content_length_score < 0.3
        assert quality_metrics.credibility_score < 0.4

    def test_quality_validation_with_custom_threshold(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test quality validation with custom threshold settings."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        medium_quality_content = "This is a moderate quality article with some useful information but limited depth."
        
        # Test with high threshold
        quality_metrics = service._calculate_quality_score(medium_quality_content)
        assert service._passes_quality_threshold(quality_metrics, threshold=0.9) == False
        
        # Test with low threshold
        assert service._passes_quality_threshold(quality_metrics, threshold=0.4) == True

    # RED PHASE: Collection Assignment Tests

    def test_auto_assign_collection_ml_content(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test automatic collection assignment for machine learning content."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        ml_content = "This paper discusses neural networks, deep learning algorithms, and supervised learning techniques for image classification."
        
        with patch.object(service, '_analyze_content_topics', return_value={'machine-learning': 0.92, 'AI': 0.85}):
            collection, confidence = service._auto_assign_collection(ml_content)
            
            assert collection == 'machine-learning'
            assert confidence == 0.92

    def test_auto_assign_collection_fallback_default(self, mock_config_manager, mock_vector_store, mock_embedding_service):
        """Test collection assignment fallback to default when no clear category."""
        service = AugmentationService(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        ambiguous_content = "This is some general text without clear categorization markers."
        
        with patch.object(service, '_analyze_content_topics', return_value={'general': 0.4, 'misc': 0.3}):
            collection, confidence = service._auto_assign_collection(ambiguous_content)
            
            assert collection == 'research'  # Default collection
            assert confidence < 0.5


class TestQualityMetrics:
    """Test suite for QualityMetrics dataclass."""
    
    def test_quality_metrics_creation(self):
        """Test QualityMetrics dataclass creation."""
        metrics = QualityMetrics(
            overall_score=0.85,
            content_length_score=0.9,
            structure_score=0.8,
            credibility_score=0.85,
            freshness_score=0.7,
            relevance_score=0.9
        )
        
        assert metrics.overall_score == 0.85
        assert metrics.content_length_score == 0.9
        assert metrics.structure_score == 0.8
        assert metrics.credibility_score == 0.85
        assert metrics.freshness_score == 0.7
        assert metrics.relevance_score == 0.9

    def test_quality_metrics_validation(self):
        """Test QualityMetrics score validation."""
        metrics = QualityMetrics(
            overall_score=1.2,  # Invalid: > 1.0
            content_length_score=-0.1,  # Invalid: < 0.0
            structure_score=0.5
        )
        
        # Should normalize or validate scores
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.content_length_score <= 1.0


class TestExternalResult:
    """Test suite for ExternalResult dataclass."""
    
    def test_external_result_creation_basic(self):
        """Test basic ExternalResult creation."""
        result = ExternalResult(
            source_url="https://example.com",
            title="Test Article",
            content="Test content..."
        )
        
        assert result.source_url == "https://example.com"
        assert result.title == "Test Article"
        assert result.content == "Test content..."
        assert result.author is None
        assert result.publication_date is None

    def test_external_result_creation_full_metadata(self):
        """Test ExternalResult creation with full metadata."""
        result = ExternalResult(
            source_url="https://arxiv.org/abs/2301.12345",
            title="Research Paper",
            content="Abstract content...",
            author="Dr. Jane Smith",
            publication_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            tags=["AI", "research"],
            metadata={"doi": "10.1000/123456"}
        )
        
        assert result.author == "Dr. Jane Smith"
        assert result.publication_date.year == 2024
        assert "AI" in result.tags
        assert result.metadata["doi"] == "10.1000/123456"

    def test_external_result_url_validation(self):
        """Test URL validation in ExternalResult."""
        # Valid URL
        result = ExternalResult(
            source_url="https://valid-url.com/article",
            title="Test",
            content="Content..."
        )
        assert result.source_url.startswith("https://")
        
        # Invalid URL should be handled gracefully
        with pytest.raises(ValueError):
            ExternalResult(
                source_url="not-a-valid-url",
                title="Test",
                content="Content..."
            )


class TestResearchReport:
    """Test suite for ResearchReport dataclass."""
    
    def test_research_report_creation(self):
        """Test ResearchReport creation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Research Report\n\nContent...")
            temp_path = f.name
        
        try:
            report = ResearchReport(
                file_path=temp_path,
                category="literature-review",
                metadata={"project": "AI Research"}
            )
            
            assert report.file_path == temp_path
            assert report.category == "literature-review"
            assert report.metadata["project"] == "AI Research"
        finally:
            Path(temp_path).unlink()

    def test_research_report_file_validation(self):
        """Test file path validation in ResearchReport."""
        # Nonexistent file should raise error
        with pytest.raises(FileNotFoundError):
            ResearchReport(file_path="/nonexistent/file.md")


class TestDocumentUpdate:
    """Test suite for DocumentUpdate dataclass."""
    
    def test_document_update_content_only(self):
        """Test DocumentUpdate for content-only changes."""
        update = DocumentUpdate(
            document_id="doc_001",
            new_content="Updated content...",
            update_embeddings=True
        )
        
        assert update.document_id == "doc_001"
        assert update.new_content == "Updated content..."
        assert update.update_embeddings == True
        assert update.new_metadata is None

    def test_document_update_metadata_only(self):
        """Test DocumentUpdate for metadata-only changes."""
        update = DocumentUpdate(
            document_id="doc_002",
            new_metadata={"title": "New Title", "tags": ["updated"]},
            update_embeddings=False
        )
        
        assert update.new_metadata["title"] == "New Title"
        assert update.update_embeddings == False
        assert update.new_content is None

    def test_document_update_from_file(self):
        """Test DocumentUpdate from source file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Updated Content\n\nNew content from file...")
            temp_path = f.name
        
        try:
            update = DocumentUpdate(
                document_id="doc_003",
                source_file=temp_path
            )
            
            assert update.source_file == temp_path
        finally:
            Path(temp_path).unlink()


class TestDuplicateGroup:
    """Test suite for DuplicateGroup dataclass."""
    
    def test_duplicate_group_creation(self):
        """Test DuplicateGroup creation."""
        group = DuplicateGroup(
            group_id=1,
            documents=["doc_001", "doc_002", "doc_003"],
            similarity=0.93,
            merge_strategy="content_union"
        )
        
        assert group.group_id == 1
        assert len(group.documents) == 3
        assert group.similarity == 0.93
        assert group.merge_strategy == "content_union"

    def test_duplicate_group_validation(self):
        """Test DuplicateGroup validation."""
        # Should require at least 2 documents
        with pytest.raises(ValueError):
            DuplicateGroup(
                group_id=1,
                documents=["doc_001"],  # Only one document
                similarity=0.9
            )
        
        # Similarity should be between 0 and 1
        with pytest.raises(ValueError):
            DuplicateGroup(
                group_id=1,
                documents=["doc_001", "doc_002"],
                similarity=1.5  # Invalid similarity > 1.0
            ) 