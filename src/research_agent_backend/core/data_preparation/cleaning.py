"""
Data cleaning service for text and metadata processing.

This module provides comprehensive data cleaning capabilities including text normalization,
encoding fixes, duplicate detection, and metadata standardization.
"""

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, Optional

from .types import DataCleaningConfig


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