"""
Test suite for user feedback service implementation following TDD approach.

Tests cover:
- User feedback submission and storage
- Feedback rating and categorization
- Feedback analytics and reporting
- Quality score updates based on feedback
- Feedback export functionality

This follows TDD Red-Green-Refactor methodology.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import tempfile
from pathlib import Path

from research_agent_backend.core.feedback_service import (
    FeedbackService,
    UserFeedback,
    FeedbackAnalytics,
    FeedbackExportOptions,
    FeedbackConfig
)
from research_agent_backend.models.metadata_schema import DocumentMetadata
from research_agent_backend.utils.config import ConfigManager


class TestFeedbackService:
    """Test suite for FeedbackService following TDD methodology."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with feedback settings."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            'feedback.enable_analytics': True,
            'feedback.quality_impact_weight': 0.3,
            'feedback.storage_backend': 'sqlite',
            'feedback.retention_days': 365,
            'feedback.batch_size': 100,
            'feedback.anonymous_allowed': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_storage_backend(self):
        """Mock storage backend for feedback data."""
        storage = Mock()
        storage.store_feedback.return_value = {'feedback_id': 'fb_001', 'status': 'stored'}
        storage.get_feedback.return_value = None
        storage.get_feedback_by_chunk.return_value = []
        storage.get_analytics.return_value = {}
        storage.export_feedback.return_value = {'records': 100, 'file': 'export.csv'}
        return storage
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for chunk operations."""
        vector_store = Mock()
        vector_store.get_chunk.return_value = {
            'id': 'chunk_001',
            'content': 'Sample chunk content...',
            'document_id': 'doc_001',
            'quality_score': 0.75
        }
        vector_store.update_chunk_quality.return_value = {'success': True}
        return vector_store
    
    @pytest.fixture
    def sample_feedback(self):
        """Sample user feedback for testing."""
        return UserFeedback(
            chunk_id="chunk_001",
            rating="positive",
            reason="very-relevant",
            comment="This information was exactly what I needed.",
            user_id="user_123",
            timestamp=datetime.now(timezone.utc)
        )

    # RED PHASE: Service Initialization and Configuration Tests

    def test_feedback_service_initialization_with_dependencies(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test FeedbackService initializes properly with all dependencies."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        assert service.config_manager == mock_config_manager
        assert service.storage_backend == mock_storage_backend
        assert service.vector_store == mock_vector_store
        assert service.enable_analytics == True
        assert service.quality_impact_weight == 0.3

    def test_feedback_config_creation_and_validation(self):
        """Test FeedbackConfig dataclass creation and validation."""
        config = FeedbackConfig(
            enable_analytics=False,
            quality_impact_weight=0.5,
            storage_backend='postgresql',
            retention_days=180,
            anonymous_allowed=False
        )
        
        assert config.enable_analytics == False
        assert config.quality_impact_weight == 0.5
        assert config.storage_backend == 'postgresql'
        assert config.retention_days == 180
        assert config.anonymous_allowed == False

    def test_feedback_config_default_values(self):
        """Test FeedbackConfig has appropriate default values."""
        config = FeedbackConfig()
        
        assert config.enable_analytics == True
        assert config.quality_impact_weight == 0.3
        assert config.storage_backend == 'sqlite'
        assert config.retention_days == 365
        assert config.batch_size == 100
        assert config.anonymous_allowed == True

    # RED PHASE: Feedback Submission Tests

    def test_submit_feedback_positive_basic(self, mock_config_manager, mock_storage_backend, mock_vector_store, sample_feedback):
        """Test submitting basic positive feedback."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        result = service.submit_feedback(sample_feedback)
        
        assert result['feedback_id'] == 'fb_001'
        assert result['status'] == 'recorded'
        assert result['impact'] == 'quality_score_updated'
        mock_storage_backend.store_feedback.assert_called_once()
        mock_vector_store.update_chunk_quality.assert_called()

    def test_submit_feedback_negative_with_reason(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test submitting negative feedback with specific reason."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        negative_feedback = UserFeedback(
            chunk_id="chunk_002",
            rating="negative",
            reason="incorrect-information",
            comment="This information is outdated and misleading.",
            user_id="user_456"
        )
        
        mock_storage_backend.store_feedback.return_value = {
            'feedback_id': 'fb_002',
            'status': 'stored',
            'flagged_for_review': True
        }
        
        result = service.submit_feedback(negative_feedback)
        
        assert result['feedback_id'] == 'fb_002'
        assert result['flagged_for_review'] == True
        assert result['action'] == 'marked_for_review'

    def test_submit_feedback_with_metadata_validation(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback submission with metadata validation."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Mock chunk not found
        mock_vector_store.get_chunk.return_value = None
        
        invalid_feedback = UserFeedback(
            chunk_id="nonexistent_chunk",
            rating="positive",
            reason="relevant"
        )
        
        result = service.submit_feedback(invalid_feedback)
        
        assert result['status'] == 'error'
        assert 'chunk not found' in result['error'].lower()

    def test_submit_feedback_anonymous_user(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test submitting feedback from anonymous user."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        anonymous_feedback = UserFeedback(
            chunk_id="chunk_003",
            rating="positive",
            reason="helpful",
            user_id=None  # Anonymous user
        )
        
        result = service.submit_feedback(anonymous_feedback)
        
        assert result['status'] == 'recorded'
        assert result['anonymous'] == True

    def test_submit_feedback_batch_processing(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test batch submission of multiple feedback items."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        feedback_batch = [
            UserFeedback(chunk_id="chunk_001", rating="positive", reason="relevant"),
            UserFeedback(chunk_id="chunk_002", rating="negative", reason="outdated"),
            UserFeedback(chunk_id="chunk_003", rating="positive", reason="helpful")
        ]
        
        mock_storage_backend.store_feedback_batch.return_value = {
            'processed': 3,
            'successful': 2,
            'failed': 1,
            'feedback_ids': ['fb_001', 'fb_002']
        }
        
        result = service.submit_feedback_batch(feedback_batch)
        
        assert result['processed'] == 3
        assert result['successful'] == 2
        assert result['failed'] == 1

    # RED PHASE: Quality Score Update Tests

    def test_update_chunk_quality_score_positive_feedback(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test quality score update for positive feedback."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Mock existing feedback for the chunk
        mock_storage_backend.get_feedback_by_chunk.return_value = [
            {'rating': 'positive', 'weight': 1.0},
            {'rating': 'positive', 'weight': 1.0},
            {'rating': 'negative', 'weight': 1.0}
        ]
        
        result = service._update_chunk_quality_score("chunk_001", "positive")
        
        assert result['new_quality_score'] > 0.75  # Should increase
        assert result['feedback_impact'] > 0
        mock_vector_store.update_chunk_quality.assert_called()

    def test_update_chunk_quality_score_negative_feedback(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test quality score update for negative feedback."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_feedback_by_chunk.return_value = [
            {'rating': 'negative', 'weight': 1.0},
            {'rating': 'negative', 'weight': 1.0},
            {'rating': 'positive', 'weight': 1.0}
        ]
        
        result = service._update_chunk_quality_score("chunk_001", "negative")
        
        assert result['new_quality_score'] < 0.75  # Should decrease
        assert result['feedback_impact'] < 0

    def test_quality_score_calculation_with_weighted_feedback(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test quality score calculation with weighted feedback."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Mock feedback with different weights (expert vs. regular users)
        mock_storage_backend.get_feedback_by_chunk.return_value = [
            {'rating': 'positive', 'weight': 2.0, 'user_type': 'expert'},
            {'rating': 'negative', 'weight': 1.0, 'user_type': 'regular'},
            {'rating': 'positive', 'weight': 1.0, 'user_type': 'regular'}
        ]
        
        quality_score = service._calculate_weighted_quality_score("chunk_001")
        
        # Expert positive feedback should have more impact
        assert quality_score > 0.75
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

    # RED PHASE: Feedback Analytics Tests

    def test_get_feedback_analytics_basic(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test basic feedback analytics retrieval."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_analytics.return_value = {
            'total_feedback': 150,
            'positive_count': 110,
            'negative_count': 40,
            'positive_ratio': 0.73,
            'negative_ratio': 0.27
        }
        
        analytics = service.get_feedback_analytics(period='30d')
        
        assert analytics['total_feedback'] == 150
        assert analytics['positive_ratio'] == 0.73
        assert analytics['negative_ratio'] == 0.27
        mock_storage_backend.get_analytics.assert_called_once()

    def test_get_feedback_analytics_with_collection_filter(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback analytics with collection filtering."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_analytics.return_value = {
            'collection': 'research',
            'total_feedback': 50,
            'positive_ratio': 0.8
        }
        
        analytics = service.get_feedback_analytics(collection='research')
        
        assert analytics['collection'] == 'research'
        assert analytics['total_feedback'] == 50

    def test_get_feedback_trends_analysis(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback trends analysis over time."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_trends_data.return_value = {
            'quality_trends': [0.65, 0.71, 0.73, 0.75],
            'feedback_volume': [20, 35, 40, 45],
            'satisfaction_trend': 'improving'
        }
        
        trends = service.get_feedback_trends(period='90d')
        
        assert trends['satisfaction_trend'] == 'improving'
        assert len(trends['quality_trends']) == 4
        assert trends['quality_trends'][-1] > trends['quality_trends'][0]  # Improving

    def test_get_top_issues_analysis(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test analysis of top feedback issues."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_issue_analysis.return_value = {
            'top_issues': [
                {'reason': 'outdated-info', 'count': 25, 'percentage': 35.7},
                {'reason': 'irrelevant-content', 'count': 18, 'percentage': 25.7},
                {'reason': 'insufficient-detail', 'count': 12, 'percentage': 17.1}
            ],
            'total_negative_feedback': 70
        }
        
        issues = service.get_top_issues()
        
        assert len(issues['top_issues']) == 3
        assert issues['top_issues'][0]['reason'] == 'outdated-info'
        assert issues['top_issues'][0]['count'] == 25

    # RED PHASE: Feedback Export Tests

    def test_export_feedback_csv_format(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback export in CSV format."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        export_options = FeedbackExportOptions(
            format='csv',
            date_range=('2024-01-01', '2024-01-31'),
            include_comments=True,
            include_metadata=False
        )
        
        result = service.export_feedback('/tmp/feedback_export.csv', export_options)
        
        assert result['export_file'] == '/tmp/feedback_export.csv'
        assert result['records_exported'] == 100
        assert result['format'] == 'csv'
        mock_storage_backend.export_feedback.assert_called_once()

    def test_export_feedback_json_format_with_metadata(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback export in JSON format with metadata."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        export_options = FeedbackExportOptions(
            format='json',
            include_metadata=True,
            include_chunk_content=True
        )
        
        mock_storage_backend.export_feedback.return_value = {
            'records': 75,
            'file': '/tmp/feedback_export.json',
            'metadata_included': True
        }
        
        result = service.export_feedback('/tmp/feedback_export.json', export_options)
        
        assert result['metadata_included'] == True
        assert result['format'] == 'json'

    def test_export_feedback_filtered_by_rating(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback export filtered by rating type."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        export_options = FeedbackExportOptions(
            format='csv',
            rating_filter='negative',
            include_comments=True
        )
        
        result = service.export_feedback('/tmp/negative_feedback.csv', export_options)
        
        assert result['rating_filter_applied'] == 'negative'

    # RED PHASE: User Feedback Validation Tests

    def test_validate_feedback_rating_values(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test validation of feedback rating values."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Valid ratings
        assert service._validate_rating('positive') == True
        assert service._validate_rating('negative') == True
        assert service._validate_rating('neutral') == True
        
        # Invalid ratings
        assert service._validate_rating('invalid_rating') == False
        assert service._validate_rating('') == False
        assert service._validate_rating(None) == False

    def test_validate_feedback_reason_categories(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test validation of feedback reason categories."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Valid reasons
        valid_reasons = [
            'very-relevant', 'somewhat-relevant', 'irrelevant-content',
            'outdated-info', 'incorrect-information', 'insufficient-detail'
        ]
        
        for reason in valid_reasons:
            assert service._validate_reason(reason) == True
        
        # Invalid reasons
        assert service._validate_reason('invalid_reason') == False
        assert service._validate_reason('') == False

    def test_sanitize_feedback_comment(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test feedback comment sanitization."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        # Test various comment sanitization scenarios
        clean_comment = "This is a helpful comment."
        assert service._sanitize_comment(clean_comment) == clean_comment
        
        # Test HTML removal
        html_comment = "This is <script>alert('test')</script> content."
        sanitized = service._sanitize_comment(html_comment)
        assert '<script>' not in sanitized
        assert 'content.' in sanitized
        
        # Test length limits
        long_comment = "A" * 2000  # Very long comment
        sanitized = service._sanitize_comment(long_comment)
        assert len(sanitized) <= 1000  # Should be truncated

    # RED PHASE: Feedback Storage and Retrieval Tests

    def test_get_feedback_by_chunk_id(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test retrieving feedback for specific chunk."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_feedback_by_chunk.return_value = [
            {
                'feedback_id': 'fb_001',
                'rating': 'positive',
                'reason': 'very-relevant',
                'timestamp': '2024-01-15T10:00:00Z'
            },
            {
                'feedback_id': 'fb_002',
                'rating': 'negative', 
                'reason': 'outdated-info',
                'timestamp': '2024-01-16T14:30:00Z'
            }
        ]
        
        feedback_list = service.get_feedback_by_chunk('chunk_001')
        
        assert len(feedback_list) == 2
        assert feedback_list[0]['rating'] == 'positive'
        assert feedback_list[1]['rating'] == 'negative'

    def test_get_feedback_by_user(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test retrieving feedback submitted by specific user."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_feedback_by_user.return_value = [
            {'feedback_id': 'fb_003', 'chunk_id': 'chunk_001', 'rating': 'positive'},
            {'feedback_id': 'fb_004', 'chunk_id': 'chunk_005', 'rating': 'positive'},
            {'feedback_id': 'fb_005', 'chunk_id': 'chunk_010', 'rating': 'negative'}
        ]
        
        user_feedback = service.get_feedback_by_user('user_123')
        
        assert len(user_feedback) == 3
        assert all(fb['feedback_id'].startswith('fb_') for fb in user_feedback)

    def test_delete_feedback_with_quality_adjustment(self, mock_config_manager, mock_storage_backend, mock_vector_store):
        """Test deleting feedback and adjusting quality scores."""
        service = FeedbackService(
            config_manager=mock_config_manager,
            storage_backend=mock_storage_backend,
            vector_store=mock_vector_store
        )
        
        mock_storage_backend.get_feedback.return_value = {
            'feedback_id': 'fb_001',
            'chunk_id': 'chunk_001',
            'rating': 'positive',
            'weight': 1.0
        }
        
        mock_storage_backend.delete_feedback.return_value = {'success': True}
        
        result = service.delete_feedback('fb_001', adjust_quality=True)
        
        assert result['success'] == True
        assert result['quality_adjusted'] == True
        mock_storage_backend.delete_feedback.assert_called_once()
        mock_vector_store.update_chunk_quality.assert_called()


class TestUserFeedback:
    """Test suite for UserFeedback dataclass."""
    
    def test_user_feedback_creation_basic(self):
        """Test basic UserFeedback creation."""
        feedback = UserFeedback(
            chunk_id="chunk_001",
            rating="positive",
            reason="very-relevant"
        )
        
        assert feedback.chunk_id == "chunk_001"
        assert feedback.rating == "positive"
        assert feedback.reason == "very-relevant"
        assert feedback.comment is None
        assert feedback.user_id is None

    def test_user_feedback_creation_full(self):
        """Test UserFeedback creation with all fields."""
        timestamp = datetime.now(timezone.utc)
        feedback = UserFeedback(
            chunk_id="chunk_002",
            rating="negative",
            reason="incorrect-information",
            comment="This information contradicts recent research.",
            user_id="user_456",
            timestamp=timestamp,
            metadata={"session_id": "sess_123"}
        )
        
        assert feedback.comment == "This information contradicts recent research."
        assert feedback.user_id == "user_456"
        assert feedback.timestamp == timestamp
        assert feedback.metadata["session_id"] == "sess_123"

    def test_user_feedback_validation(self):
        """Test UserFeedback field validation."""
        # Valid feedback
        feedback = UserFeedback(
            chunk_id="chunk_001",
            rating="positive",
            reason="helpful"
        )
        assert feedback.rating in ["positive", "negative", "neutral"]
        
        # Invalid rating should raise error
        with pytest.raises(ValueError):
            UserFeedback(
                chunk_id="chunk_001",
                rating="invalid_rating",
                reason="helpful"
            )


class TestFeedbackAnalytics:
    """Test suite for FeedbackAnalytics dataclass."""
    
    def test_feedback_analytics_creation(self):
        """Test FeedbackAnalytics creation."""
        analytics = FeedbackAnalytics(
            total_feedback=150,
            positive_count=110,
            negative_count=40,
            positive_ratio=0.73,
            negative_ratio=0.27,
            top_issues=["outdated-info", "irrelevant-content"],
            quality_trends=[0.65, 0.71, 0.73]
        )
        
        assert analytics.total_feedback == 150
        assert analytics.positive_ratio == 0.73
        assert len(analytics.top_issues) == 2
        assert len(analytics.quality_trends) == 3

    def test_feedback_analytics_calculations(self):
        """Test analytics calculations."""
        analytics = FeedbackAnalytics(
            total_feedback=100,
            positive_count=75,
            negative_count=25
        )
        
        # Should calculate ratios automatically
        assert analytics.positive_ratio == 0.75
        assert analytics.negative_ratio == 0.25
        assert analytics.positive_ratio + analytics.negative_ratio == 1.0


class TestFeedbackExportOptions:
    """Test suite for FeedbackExportOptions dataclass."""
    
    def test_export_options_creation(self):
        """Test FeedbackExportOptions creation."""
        options = FeedbackExportOptions(
            format='csv',
            date_range=('2024-01-01', '2024-01-31'),
            rating_filter='negative',
            include_comments=True,
            include_metadata=False
        )
        
        assert options.format == 'csv'
        assert options.date_range == ('2024-01-01', '2024-01-31')
        assert options.rating_filter == 'negative'
        assert options.include_comments == True
        assert options.include_metadata == False

    def test_export_options_validation(self):
        """Test export options validation."""
        # Valid format
        options = FeedbackExportOptions(format='json')
        assert options.format in ['csv', 'json', 'excel']
        
        # Invalid format should raise error
        with pytest.raises(ValueError):
            FeedbackExportOptions(format='invalid_format')
        
        # Valid rating filter
        options = FeedbackExportOptions(rating_filter='positive')
        assert options.rating_filter in ['positive', 'negative', 'neutral', None]
        
        # Invalid rating filter should raise error
        with pytest.raises(ValueError):
            FeedbackExportOptions(rating_filter='invalid_filter') 