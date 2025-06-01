"""
Test suite for knowledge base augmentation CLI commands.

Tests the augmentation CLI commands including add-external-result, add-research-report,
update-document, merge-duplicates, and user feedback functionality.
Follows TDD approach - RED PHASE.

These tests are designed to fail initially and guide implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from click.testing import Result
import json
import tempfile
from pathlib import Path

from research_agent_backend.cli.cli import app
from research_agent_backend.core.query_manager import QueryManager, QueryResult
from research_agent_backend.exceptions.query_exceptions import QueryError

runner = CliRunner()


class TestAddExternalResultCommand:
    """Test the add-external-result command."""
    
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_external_result_basic(self, mock_augmentation_service):
        """Test adding a basic external search result."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_external_result.return_value = {
            'document_id': 'ext_001',
            'status': 'success',
            'collection': 'research'
        }
        
        result = runner.invoke(app, [
            'kb', 'add-external-result',
            '--source', 'https://example.com/article',
            '--title', 'Machine Learning Advances',
            '--content', 'Recent advances in machine learning...',
            '--collection', 'research'
        ])
        
        assert result.exit_code == 0
        assert 'ext_001' in result.stdout
        assert 'success' in result.stdout
        mock_service_instance.add_external_result.assert_called_once()
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_external_result_with_metadata(self, mock_augmentation_service):
        """Test adding external result with rich metadata."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_external_result.return_value = {
            'document_id': 'ext_002',
            'status': 'success',
            'metadata_extracted': True
        }
        
        result = runner.invoke(app, [
            'kb', 'add-external-result',
            '--source', 'https://arxiv.org/abs/2301.12345',
            '--title', 'Neural Architecture Search',
            '--content', 'We propose a novel approach...',
            '--author', 'John Doe',
            '--publication-date', '2023-01-15',
            '--tags', 'AI,NAS,neural-networks'
        ])
        
        assert result.exit_code == 0
        assert 'ext_002' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_external_result_from_file(self, mock_augmentation_service):
        """Test adding external result from JSON file."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_external_result.return_value = {
            'document_id': 'ext_003',
            'status': 'success'
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'source': 'https://example.com',
                'title': 'Test Article',
                'content': 'Test content...',
                'metadata': {'category': 'research'}
            }, f)
            temp_path = f.name
        
        try:
            result = runner.invoke(app, [
                'kb', 'add-external-result',
                '--from-file', temp_path
            ])
            
            assert result.exit_code == 0
            assert 'ext_003' in result.stdout
        finally:
            Path(temp_path).unlink()
            
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_external_result_validation_error(self, mock_augmentation_service):
        """Test error handling for invalid external result data."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_external_result.side_effect = ValueError("Invalid content format")
        
        result = runner.invoke(app, [
            'kb', 'add-external-result',
            '--source', 'invalid-url',
            '--content', ''  # Empty content should fail validation
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.stdout.lower() or 'invalid' in result.stdout.lower()


class TestAddResearchReportCommand:
    """Test the add-research-report command."""
    
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_research_report_basic(self, mock_augmentation_service):
        """Test adding a basic research report."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_research_report.return_value = {
            'document_id': 'rpt_001',
            'status': 'success',
            'chunks_created': 15
        }
        
        # Create temporary research file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Research Report\n\nThis is a comprehensive research report...")
            temp_path = f.name
        
        try:
            result = runner.invoke(app, [
                'kb', 'add-research-report',
                temp_path,
                '--collection', 'research-reports',
                '--category', 'literature-review'
            ])
            
            assert result.exit_code == 0
            assert 'rpt_001' in result.stdout
            assert '15' in result.stdout  # chunks created
        finally:
            Path(temp_path).unlink()
            
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_research_report_with_auto_categorization(self, mock_augmentation_service):
        """Test research report with automatic categorization."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_research_report.return_value = {
            'document_id': 'rpt_002',
            'status': 'success',
            'auto_category': 'machine-learning',
            'confidence': 0.87
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# ML Research\n\nMachine learning algorithms...")
            temp_path = f.name
        
        try:
            result = runner.invoke(app, [
                'kb', 'add-research-report',
                temp_path,
                '--auto-categorize'
            ])
            
            assert result.exit_code == 0
            assert 'machine-learning' in result.stdout
            assert '0.87' in result.stdout
        finally:
            Path(temp_path).unlink()
            
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_add_research_report_batch_mode(self, mock_augmentation_service):
        """Test adding multiple research reports in batch."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.add_research_reports_batch.return_value = {
            'processed': 3,
            'successful': 2,
            'failed': 1,
            'results': ['rpt_003', 'rpt_004']
        }
        
        # Create temporary directory with research files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "report1.md").write_text("# Report 1")
            (temp_path / "report2.md").write_text("# Report 2")
            (temp_path / "report3.md").write_text("# Report 3")
            
            result = runner.invoke(app, [
                'kb', 'add-research-report',
                str(temp_path),
                '--batch',
                '--pattern', '*.md'
            ])
            
            assert result.exit_code == 0
            assert 'processed: 3' in result.stdout.lower()
            assert 'successful: 2' in result.stdout.lower()


class TestUpdateDocumentCommand:
    """Test the update-document command."""
    
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_update_document_content(self, mock_augmentation_service):
        """Test updating document content."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.update_document.return_value = {
            'document_id': 'doc_001',
            'status': 'updated',
            'version': 2,
            'changes': ['content_updated', 'embeddings_regenerated']
        }
        
        result = runner.invoke(app, [
            'kb', 'update-document',
            'doc_001',
            '--content', 'Updated content for the document...',
            '--update-embeddings'
        ])
        
        assert result.exit_code == 0
        assert 'doc_001' in result.stdout
        assert 'version: 2' in result.stdout.lower()
        assert 'updated' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_update_document_metadata_only(self, mock_augmentation_service):
        """Test updating only document metadata."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.update_document.return_value = {
            'document_id': 'doc_002',
            'status': 'updated',
            'changes': ['metadata_updated']
        }
        
        result = runner.invoke(app, [
            'kb', 'update-document',
            'doc_002',
            '--title', 'New Title',
            '--tags', 'updated,metadata',
            '--no-reembed'
        ])
        
        assert result.exit_code == 0
        assert 'metadata_updated' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_update_document_from_file(self, mock_augmentation_service):
        """Test updating document from file."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.update_document.return_value = {
            'document_id': 'doc_003',
            'status': 'updated'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Updated Document\n\nThis is the updated content...")
            temp_path = f.name
        
        try:
            result = runner.invoke(app, [
                'kb', 'update-document',
                'doc_003',
                '--from-file', temp_path
            ])
            
            assert result.exit_code == 0
        finally:
            Path(temp_path).unlink()


class TestMergeDuplicatesCommand:
    """Test the merge-duplicates command."""
    
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_merge_duplicates_auto_detect(self, mock_augmentation_service):
        """Test automatic duplicate detection and merging."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.detect_duplicates.return_value = [
            {
                'group_id': 1,
                'documents': ['doc_001', 'doc_002'],
                'similarity': 0.95,
                'merge_strategy': 'content_union'
            }
        ]
        mock_service_instance.merge_duplicates.return_value = {
            'merged_groups': 1,
            'documents_merged': 2,
            'new_document_id': 'doc_merged_001'
        }
        
        result = runner.invoke(app, [
            'kb', 'merge-duplicates',
            '--auto-detect',
            '--threshold', '0.9',
            '--strategy', 'union'
        ])
        
        assert result.exit_code == 0
        assert 'merged_groups: 1' in result.stdout.lower()
        assert 'doc_merged_001' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_merge_duplicates_manual_selection(self, mock_augmentation_service):
        """Test manual duplicate selection and merging."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.merge_duplicates.return_value = {
            'merged_groups': 1,
            'documents_merged': 3,
            'new_document_id': 'doc_merged_002'
        }
        
        result = runner.invoke(app, [
            'kb', 'merge-duplicates',
            '--documents', 'doc_001,doc_002,doc_003',
            '--strategy', 'latest-version',
            '--keep-originals'
        ])
        
        assert result.exit_code == 0
        assert 'doc_merged_002' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    def test_merge_duplicates_preview_mode(self, mock_augmentation_service):
        """Test duplicate detection in preview mode."""
        mock_service_instance = Mock()
        mock_augmentation_service.return_value = mock_service_instance
        mock_service_instance.detect_duplicates.return_value = [
            {
                'group_id': 1,
                'documents': ['doc_001', 'doc_002'],
                'similarity': 0.93
            },
            {
                'group_id': 2,
                'documents': ['doc_003', 'doc_004'],
                'similarity': 0.88
            }
        ]
        
        result = runner.invoke(app, [
            'kb', 'merge-duplicates',
            '--preview-only',
            '--threshold', '0.85'
        ])
        
        assert result.exit_code == 0
        assert 'group_id: 1' in result.stdout.lower()
        assert 'group_id: 2' in result.stdout.lower()
        assert 'preview' in result.stdout.lower()


class TestUserFeedbackCommands:
    """Test user feedback commands."""
    
    @patch('research_agent_backend.cli.augmentation.FeedbackService')
    def test_submit_feedback_positive(self, mock_feedback_service):
        """Test submitting positive feedback."""
        mock_service_instance = Mock()
        mock_feedback_service.return_value = mock_service_instance
        mock_service_instance.submit_feedback.return_value = {
            'feedback_id': 'fb_001',
            'status': 'recorded',
            'impact': 'quality_score_updated'
        }
        
        result = runner.invoke(app, [
            'kb', 'feedback',
            '--chunk-id', 'chunk_001',
            '--rating', 'positive',
            '--reason', 'very-relevant',
            '--comment', 'This answer was exactly what I needed.'
        ])
        
        assert result.exit_code == 0
        assert 'fb_001' in result.stdout
        assert 'recorded' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.FeedbackService')
    def test_submit_feedback_negative(self, mock_feedback_service):
        """Test submitting negative feedback."""
        mock_service_instance = Mock()
        mock_feedback_service.return_value = mock_service_instance
        mock_service_instance.submit_feedback.return_value = {
            'feedback_id': 'fb_002',
            'status': 'recorded',
            'action': 'marked_for_review'
        }
        
        result = runner.invoke(app, [
            'kb', 'feedback',
            '--chunk-id', 'chunk_002',
            '--rating', 'negative',
            '--reason', 'incorrect-information',
            '--comment', 'This information is outdated.'
        ])
        
        assert result.exit_code == 0
        assert 'fb_002' in result.stdout
        assert 'marked_for_review' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.FeedbackService')
    def test_feedback_analytics(self, mock_feedback_service):
        """Test feedback analytics command."""
        mock_service_instance = Mock()
        mock_feedback_service.return_value = mock_service_instance
        mock_service_instance.get_feedback_analytics.return_value = {
            'total_feedback': 150,
            'positive_ratio': 0.73,
            'negative_ratio': 0.27,
            'top_issues': ['outdated-info', 'irrelevant-content'],
            'quality_trends': [0.65, 0.71, 0.73]
        }
        
        result = runner.invoke(app, [
            'kb', 'feedback-analytics',
            '--period', '30d',
            '--collection', 'research'
        ])
        
        assert result.exit_code == 0
        assert '150' in result.stdout  # total feedback
        assert '0.73' in result.stdout  # positive ratio
        assert 'outdated-info' in result.stdout
        
    @patch('research_agent_backend.cli.augmentation.FeedbackService')
    def test_feedback_export(self, mock_feedback_service):
        """Test feedback export functionality."""
        mock_service_instance = Mock()
        mock_feedback_service.return_value = mock_service_instance
        mock_service_instance.export_feedback.return_value = {
            'export_file': '/tmp/feedback_export.csv',
            'records_exported': 150,
            'format': 'csv'
        }
        
        result = runner.invoke(app, [
            'kb', 'export-feedback',
            '--output', '/tmp/feedback_export.csv',
            '--format', 'csv',
            '--date-range', '2024-01-01,2024-01-31'
        ])
        
        assert result.exit_code == 0
        assert '150' in result.stdout
        assert 'feedback_export.csv' in result.stdout


class TestAugmentationCommandValidation:
    """Test validation and error handling across augmentation commands."""
    
    def test_add_external_result_missing_required_args(self):
        """Test error when required arguments are missing."""
        result = runner.invoke(app, [
            'kb', 'add-external-result'
            # Missing required source and content
        ])
        
        assert result.exit_code != 0
        assert 'required' in result.stdout.lower() or 'missing' in result.stdout.lower()
        
    def test_update_document_invalid_document_id(self):
        """Test error handling for invalid document ID."""
        result = runner.invoke(app, [
            'kb', 'update-document',
            'invalid_doc_id',
            '--content', 'New content'
        ])
        
        # Should handle gracefully or show appropriate error
        assert result.exit_code != 0 or 'not found' in result.stdout.lower()
        
    def test_merge_duplicates_invalid_threshold(self):
        """Test error handling for invalid similarity threshold."""
        result = runner.invoke(app, [
            'kb', 'merge-duplicates',
            '--threshold', '1.5'  # Invalid threshold > 1.0
        ])
        
        assert result.exit_code != 0
        assert 'threshold' in result.stdout.lower() or 'invalid' in result.stdout.lower()
        
    def test_feedback_invalid_rating(self):
        """Test error handling for invalid feedback rating."""
        result = runner.invoke(app, [
            'kb', 'feedback',
            '--chunk-id', 'chunk_001',
            '--rating', 'invalid_rating'
        ])
        
        assert result.exit_code != 0
        assert 'rating' in result.stdout.lower() or 'invalid' in result.stdout.lower()


class TestAugmentationCommandIntegration:
    """Test integration scenarios across augmentation commands."""
    
    @patch('research_agent_backend.cli.augmentation.AugmentationService')
    @patch('research_agent_backend.cli.augmentation.FeedbackService')
    def test_end_to_end_external_result_workflow(self, mock_feedback_service, mock_augmentation_service):
        """Test complete workflow: add external result -> get feedback -> update."""
        # Setup mocks for complete workflow
        mock_aug_service = Mock()
        mock_fb_service = Mock()
        mock_augmentation_service.return_value = mock_aug_service
        mock_feedback_service.return_value = mock_fb_service
        
        # Mock add external result
        mock_aug_service.add_external_result.return_value = {
            'document_id': 'ext_workflow_001',
            'status': 'success'
        }
        
        # Mock feedback submission
        mock_fb_service.submit_feedback.return_value = {
            'feedback_id': 'fb_workflow_001',
            'status': 'recorded'
        }
        
        # Mock document update
        mock_aug_service.update_document.return_value = {
            'document_id': 'ext_workflow_001',
            'status': 'updated'
        }
        
        # Step 1: Add external result
        result1 = runner.invoke(app, [
            'kb', 'add-external-result',
            '--source', 'https://example.com',
            '--title', 'Test Article',
            '--content', 'Test content for workflow...'
        ])
        
        assert result1.exit_code == 0
        
        # Step 2: Submit feedback
        result2 = runner.invoke(app, [
            'kb', 'feedback',
            '--chunk-id', 'ext_workflow_001_chunk_1',
            '--rating', 'positive',
            '--reason', 'very-relevant'
        ])
        
        assert result2.exit_code == 0
        
        # Step 3: Update document based on feedback
        result3 = runner.invoke(app, [
            'kb', 'update-document',
            'ext_workflow_001',
            '--content', 'Improved content based on feedback...'
        ])
        
        assert result3.exit_code == 0 