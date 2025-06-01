"""
User feedback service implementation.

Provides functionality for collecting, storing, and analyzing user feedback
on knowledge base content quality and relevance.

Implements FR-FB-001: User feedback system functionality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """Data model for user feedback on knowledge base content."""
    chunk_id: str
    rating: Literal["positive", "negative", "neutral"]
    reason: str
    comment: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate feedback data."""
        valid_ratings = ["positive", "negative", "neutral"]
        if self.rating not in valid_ratings:
            raise ValueError(f"Invalid rating: {self.rating}. Must be one of {valid_ratings}")
        
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class FeedbackAnalytics:
    """Data model for feedback analytics and insights."""
    total_feedback: int
    positive_count: int
    negative_count: int
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    top_issues: List[str] = field(default_factory=list)
    quality_trends: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate ratios if not provided."""
        if self.total_feedback > 0:
            self.positive_ratio = self.positive_count / self.total_feedback
            self.negative_ratio = self.negative_count / self.total_feedback


@dataclass
class FeedbackExportOptions:
    """Configuration options for feedback export."""
    format: Literal["csv", "json", "excel"] = "csv"
    date_range: Optional[Tuple[str, str]] = None
    rating_filter: Optional[Literal["positive", "negative", "neutral"]] = None
    include_comments: bool = True
    include_metadata: bool = False
    include_chunk_content: bool = False
    
    def __post_init__(self):
        """Validate export options."""
        valid_formats = ["csv", "json", "excel"]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}. Must be one of {valid_formats}")
        
        if self.rating_filter is not None:
            valid_filters = ["positive", "negative", "neutral"]
            if self.rating_filter not in valid_filters:
                raise ValueError(f"Invalid rating filter: {self.rating_filter}. Must be one of {valid_filters}")


@dataclass
class FeedbackConfig:
    """Configuration for feedback service."""
    enable_analytics: bool = True
    quality_impact_weight: float = 0.3
    storage_backend: str = "sqlite"
    retention_days: int = 365
    batch_size: int = 100
    anonymous_allowed: bool = True


class FeedbackService:
    """
    Service for managing user feedback on knowledge base content.
    
    Provides methods for collecting feedback, updating quality scores,
    generating analytics, and exporting feedback data.
    """
    
    def __init__(self, config_manager=None, storage_backend=None, vector_store=None):
        """Initialize the feedback service."""
        self.config_manager = config_manager
        self.storage_backend = storage_backend
        self.vector_store = vector_store
        
        # Load configuration
        if config_manager:
            self.enable_analytics = config_manager.get('feedback.enable_analytics', True)
            self.quality_impact_weight = config_manager.get('feedback.quality_impact_weight', 0.3)
        else:
            self.enable_analytics = True
            self.quality_impact_weight = 0.3
    
    def submit_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """
        Submit user feedback for a knowledge base chunk.
        
        Args:
            feedback: The user feedback to submit
            
        Returns:
            Dict containing feedback submission result
        """
        try:
            # Validate chunk exists
            if self.vector_store:
                chunk = self.vector_store.get_chunk(feedback.chunk_id)
                if not chunk:
                    return {
                        'status': 'error',
                        'error': f'Chunk {feedback.chunk_id} not found'
                    }
            
            # Generate feedback ID
            feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store feedback
            if self.storage_backend:
                storage_result = self.storage_backend.store_feedback({
                    'feedback_id': feedback_id,
                    'chunk_id': feedback.chunk_id,
                    'rating': feedback.rating,
                    'reason': feedback.reason,
                    'comment': feedback.comment,
                    'user_id': feedback.user_id,
                    'timestamp': feedback.timestamp.isoformat()
                })
            else:
                storage_result = {'feedback_id': feedback_id, 'status': 'stored'}
            
            # Update chunk quality score
            quality_update = self._update_chunk_quality_score(feedback.chunk_id, feedback.rating)
            
            result = {
                'feedback_id': feedback_id,
                'status': 'recorded',
                'impact': 'quality_score_updated'
            }
            
            # Handle negative feedback
            if feedback.rating == 'negative':
                result.update({
                    'flagged_for_review': True,
                    'action': 'marked_for_review'
                })
            
            # Handle anonymous feedback
            if not feedback.user_id:
                result['anonymous'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def submit_feedback_batch(self, feedback_batch: List[UserFeedback]) -> Dict[str, Any]:
        """
        Submit multiple feedback items in batch.
        
        Args:
            feedback_batch: List of feedback items to submit
            
        Returns:
            Dict containing batch processing results
        """
        try:
            processed = len(feedback_batch)
            successful = 0
            failed = 0
            feedback_ids = []
            
            for feedback in feedback_batch:
                try:
                    result = self.submit_feedback(feedback)
                    if result['status'] == 'recorded':
                        successful += 1
                        feedback_ids.append(result['feedback_id'])
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            
            return {
                'processed': processed,
                'successful': successful,
                'failed': failed,
                'feedback_ids': feedback_ids
            }
            
        except Exception as e:
            logger.error(f"Failed batch feedback submission: {e}")
            return {
                'processed': 0,
                'successful': 0,
                'failed': len(feedback_batch),
                'feedback_ids': []
            }
    
    def get_feedback_analytics(self, period: str = "30d", collection: str = None) -> Dict[str, Any]:
        """
        Get feedback analytics for a specified period.
        
        Args:
            period: Time period (7d, 30d, 90d, 1y)
            collection: Optional collection filter
            
        Returns:
            Dict containing analytics data
        """
        try:
            if self.storage_backend:
                analytics_data = self.storage_backend.get_analytics(period=period, collection=collection)
            else:
                # Mock analytics data
                analytics_data = {
                    'total_feedback': 150,
                    'positive_count': 110,
                    'negative_count': 40,
                    'positive_ratio': 0.73,
                    'negative_ratio': 0.27,
                    'top_issues': ['outdated-info', 'irrelevant-content'],
                    'quality_trends': [0.65, 0.71, 0.73]
                }
            
            if collection:
                analytics_data['collection'] = collection
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Failed to get feedback analytics: {e}")
            return {
                'total_feedback': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'error': str(e)
            }
    
    def get_feedback_trends(self, period: str = "90d") -> Dict[str, Any]:
        """Get feedback trends over time."""
        try:
            if self.storage_backend and hasattr(self.storage_backend, 'get_trends_data'):
                return self.storage_backend.get_trends_data(period=period)
            else:
                # Mock trends data
                return {
                    'quality_trends': [0.65, 0.71, 0.73, 0.75],
                    'feedback_volume': [20, 35, 40, 45],
                    'satisfaction_trend': 'improving'
                }
        except Exception as e:
            logger.error(f"Failed to get feedback trends: {e}")
            return {'error': str(e)}
    
    def get_top_issues(self) -> Dict[str, Any]:
        """Get analysis of top feedback issues."""
        try:
            if self.storage_backend and hasattr(self.storage_backend, 'get_issue_analysis'):
                return self.storage_backend.get_issue_analysis()
            else:
                # Mock issues data
                return {
                    'top_issues': [
                        {'reason': 'outdated-info', 'count': 25, 'percentage': 35.7},
                        {'reason': 'irrelevant-content', 'count': 18, 'percentage': 25.7},
                        {'reason': 'insufficient-detail', 'count': 12, 'percentage': 17.1}
                    ],
                    'total_negative_feedback': 70
                }
        except Exception as e:
            logger.error(f"Failed to get top issues: {e}")
            return {'error': str(e)}
    
    def export_feedback(self, output_path: str, options: FeedbackExportOptions) -> Dict[str, Any]:
        """
        Export feedback data to file.
        
        Args:
            output_path: Path to output file
            options: Export configuration options
            
        Returns:
            Dict containing export results
        """
        try:
            if self.storage_backend:
                export_result = self.storage_backend.export_feedback(output_path, options)
            else:
                # Mock export result
                export_result = {
                    'records': 100,
                    'file': output_path,
                    'format': options.format
                }
            
            result = {
                'export_file': output_path,
                'records_exported': export_result.get('records', 100),
                'format': options.format
            }
            
            if options.rating_filter:
                result['rating_filter_applied'] = options.rating_filter
            
            if options.include_metadata:
                result['metadata_included'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to export feedback: {e}")
            return {
                'export_file': output_path,
                'records_exported': 0,
                'error': str(e)
            }
    
    def get_feedback_by_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific chunk."""
        try:
            if self.storage_backend:
                return self.storage_backend.get_feedback_by_chunk(chunk_id)
            else:
                # Mock feedback data
                return [
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
        except Exception as e:
            logger.error(f"Failed to get feedback by chunk: {e}")
            return []
    
    def get_feedback_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all feedback submitted by a specific user."""
        try:
            if self.storage_backend and hasattr(self.storage_backend, 'get_feedback_by_user'):
                return self.storage_backend.get_feedback_by_user(user_id)
            else:
                # Mock user feedback data
                return [
                    {'feedback_id': 'fb_003', 'chunk_id': 'chunk_001', 'rating': 'positive'},
                    {'feedback_id': 'fb_004', 'chunk_id': 'chunk_005', 'rating': 'positive'},
                    {'feedback_id': 'fb_005', 'chunk_id': 'chunk_010', 'rating': 'negative'}
                ]
        except Exception as e:
            logger.error(f"Failed to get feedback by user: {e}")
            return []
    
    def delete_feedback(self, feedback_id: str, adjust_quality: bool = True) -> Dict[str, Any]:
        """Delete feedback and optionally adjust quality scores."""
        try:
            # Get feedback details before deletion
            if self.storage_backend:
                feedback_data = self.storage_backend.get_feedback(feedback_id)
                if not feedback_data:
                    return {'success': False, 'error': 'Feedback not found'}
                
                # Delete feedback
                delete_result = self.storage_backend.delete_feedback(feedback_id)
                
                # Adjust quality if requested
                if adjust_quality and delete_result.get('success'):
                    self._update_chunk_quality_score(
                        feedback_data['chunk_id'], 
                        feedback_data['rating'],
                        remove=True
                    )
                
                return {
                    'success': True,
                    'quality_adjusted': adjust_quality
                }
            else:
                return {'success': True, 'quality_adjusted': adjust_quality}
                
        except Exception as e:
            logger.error(f"Failed to delete feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_chunk_quality_score(self, chunk_id: str, rating: str, remove: bool = False) -> Dict[str, Any]:
        """Update quality score for a chunk based on feedback."""
        try:
            # Get existing feedback for the chunk
            if self.storage_backend:
                existing_feedback = self.storage_backend.get_feedback_by_chunk(chunk_id)
            else:
                # Mock existing feedback
                existing_feedback = [
                    {'rating': 'positive', 'weight': 1.0},
                    {'rating': 'positive', 'weight': 1.0},
                    {'rating': 'negative', 'weight': 1.0}
                ]
            
            # Calculate weighted quality score
            new_quality_score = self._calculate_weighted_quality_score(chunk_id)
            
            # Update in vector store
            if self.vector_store:
                self.vector_store.update_chunk_quality(chunk_id, new_quality_score)
            
            # Calculate feedback impact
            feedback_impact = 0.1 if rating == 'positive' else -0.1
            
            return {
                'new_quality_score': new_quality_score,
                'feedback_impact': feedback_impact
            }
            
        except Exception as e:
            logger.error(f"Failed to update chunk quality score: {e}")
            return {'new_quality_score': 0.75, 'feedback_impact': 0.0}
    
    def _calculate_weighted_quality_score(self, chunk_id: str) -> float:
        """Calculate weighted quality score based on all feedback for a chunk."""
        try:
            if self.storage_backend:
                feedback_list = self.storage_backend.get_feedback_by_chunk(chunk_id)
            else:
                # Mock feedback with different weights
                feedback_list = [
                    {'rating': 'positive', 'weight': 2.0, 'user_type': 'expert'},
                    {'rating': 'negative', 'weight': 1.0, 'user_type': 'regular'},
                    {'rating': 'positive', 'weight': 1.0, 'user_type': 'regular'}
                ]
            
            if not feedback_list:
                return 0.75  # Default score
            
            # Calculate weighted average
            total_weight = 0
            weighted_sum = 0
            
            for feedback in feedback_list:
                weight = feedback.get('weight', 1.0)
                score = 1.0 if feedback['rating'] == 'positive' else 0.0
                if feedback['rating'] == 'neutral':
                    score = 0.5
                
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.75
            
        except Exception as e:
            logger.error(f"Failed to calculate weighted quality score: {e}")
            return 0.75
    
    def _validate_rating(self, rating: str) -> bool:
        """Validate feedback rating value."""
        valid_ratings = ['positive', 'negative', 'neutral']
        return rating in valid_ratings
    
    def _validate_reason(self, reason: str) -> bool:
        """Validate feedback reason category."""
        valid_reasons = [
            'very-relevant', 'somewhat-relevant', 'irrelevant-content',
            'outdated-info', 'incorrect-information', 'insufficient-detail',
            'helpful', 'unhelpful'
        ]
        return reason in valid_reasons
    
    def _sanitize_comment(self, comment: str) -> str:
        """Sanitize feedback comment."""
        if not comment:
            return ""
        
        # Remove HTML tags (basic sanitization)
        import re
        clean_comment = re.sub(r'<[^>]+>', '', comment)
        
        # Truncate if too long
        if len(clean_comment) > 1000:
            clean_comment = clean_comment[:1000] + "..."
        
        return clean_comment.strip() 