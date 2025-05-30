"""
Metadata Types Module

Core data structures and main extractor for metadata processing.
Provides the primary interface for complete metadata extraction.

Implements FR-KB-003.4: Core metadata types and extraction orchestration.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .frontmatter import FrontmatterParser, FrontmatterResult
from .inline import InlineMetadataExtractor, InlineMetadataResult, InlineTag


@dataclass
class DocumentMetadata:
    """Complete metadata for a document.
    
    Aggregates all metadata types (frontmatter, inline, derived) into
    a single comprehensive data structure.
    
    Attributes:
        document_id: Unique identifier for the document
        title: Document title (from frontmatter or derived)
        author: Document author (from frontmatter)
        tags: List of tags from all sources
        frontmatter: Raw frontmatter metadata dictionary
        inline_metadata: Inline metadata dictionary
        inline_tags: List of inline tag objects with position info
    """
    document_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    inline_metadata: Dict[str, Any] = field(default_factory=dict)
    inline_tags: List[InlineTag] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate document metadata after creation."""
        if not self.document_id or not isinstance(self.document_id, str):
            raise ValueError(f"document_id must be a non-empty string, got: {self.document_id}")
        
        # Ensure tags is always a list
        if not isinstance(self.tags, list):
            if isinstance(self.tags, str):
                self.tags = [self.tags]
            else:
                self.tags = list(self.tags) if self.tags else []
    
    @property
    def has_metadata(self) -> bool:
        """True if document has any metadata."""
        return bool(
            self.title or 
            self.author or 
            self.tags or 
            self.frontmatter or 
            self.inline_metadata or 
            self.inline_tags
        )
    
    @property
    def all_metadata(self) -> Dict[str, Any]:
        """Combined metadata from all sources."""
        combined = {}
        
        # Add frontmatter first (lowest priority)
        combined.update(self.frontmatter)
        
        # Add inline metadata (higher priority)
        combined.update(self.inline_metadata)
        
        # Add document-level metadata (highest priority)
        if self.title:
            combined['title'] = self.title
        if self.author:
            combined['author'] = self.author
        if self.tags:
            combined['tags'] = self.tags
        
        return combined
    
    def get_tag_count(self) -> int:
        """Get total number of tags."""
        return len(self.tags)
    
    def has_tag(self, tag: str) -> bool:
        """Check if document has a specific tag."""
        return tag in self.tags
    
    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag if present."""
        if tag in self.tags:
            self.tags.remove(tag)


@dataclass
class MetadataExtractionResult:
    """Result of complete metadata extraction.
    
    Contains all extracted metadata components and processing results
    from a document.
    
    Attributes:
        document_metadata: Aggregated document metadata
        frontmatter_result: Detailed frontmatter extraction results
        inline_result: Detailed inline metadata extraction results
        content_without_frontmatter: Document content with frontmatter removed
    """
    document_metadata: DocumentMetadata
    frontmatter_result: FrontmatterResult
    inline_result: InlineMetadataResult
    content_without_frontmatter: str
    
    @property
    def has_frontmatter(self) -> bool:
        """True if frontmatter was found."""
        return self.frontmatter_result.has_frontmatter
    
    @property
    def has_inline_metadata(self) -> bool:
        """True if inline metadata was found."""
        return self.inline_result.has_metadata
    
    @property
    def extraction_successful(self) -> bool:
        """True if extraction completed without major errors."""
        return (
            self.document_metadata is not None and
            self.frontmatter_result is not None and
            self.inline_result is not None
        )
    
    @property
    def total_metadata_items(self) -> int:
        """Total number of metadata items found."""
        return (
            len(self.frontmatter_result.metadata) +
            self.inline_result.total_matches
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        return {
            'document_id': self.document_metadata.document_id,
            'has_frontmatter': self.has_frontmatter,
            'frontmatter_type': self.frontmatter_result.frontmatter_type,
            'has_inline_metadata': self.has_inline_metadata,
            'inline_patterns_matched': list(self.inline_result.patterns_matched),
            'total_metadata_items': self.total_metadata_items,
            'total_tags': self.document_metadata.get_tag_count(),
            'extraction_successful': self.extraction_successful,
            'processing_times': {
                'frontmatter_ms': self.frontmatter_result.parse_time_ms,
                'inline_ms': self.inline_result.extraction_time_ms
            }
        }


class MetadataExtractor:
    """Main class for extracting all types of metadata.
    
    Coordinates frontmatter and inline metadata extraction to provide
    a complete metadata extraction service.
    """
    
    def __init__(
        self,
        enable_performance_tracking: bool = False,
        enable_line_tracking: bool = False
    ):
        """Initialize metadata extractor.
        
        Args:
            enable_performance_tracking: Whether to track extraction performance
            enable_line_tracking: Whether to track line numbers for positions
        """
        self.frontmatter_parser = FrontmatterParser()
        self.inline_extractor = InlineMetadataExtractor(
            enable_performance_tracking=enable_performance_tracking,
            enable_line_tracking=enable_line_tracking
        )
    
    def extract_all(self, content: str, document_id: str) -> MetadataExtractionResult:
        """Extract all metadata from document.
        
        Args:
            content: Document content to extract metadata from
            document_id: Unique identifier for the document
            
        Returns:
            MetadataExtractionResult with all extracted metadata
            
        Raises:
            ValueError: If content or document_id is invalid
        """
        if not isinstance(content, str):
            raise ValueError(f"content must be a string, got {type(content)}")
        if not document_id or not isinstance(document_id, str):
            raise ValueError(f"document_id must be a non-empty string, got: {document_id}")
        
        # Parse frontmatter
        frontmatter_result = self.frontmatter_parser.parse(content)
        
        # Extract inline metadata from content without frontmatter
        inline_result = self.inline_extractor.extract(frontmatter_result.content_without_frontmatter)
        
        # Build combined metadata
        combined_tags = frontmatter_result.metadata.get('tags', [])
        if isinstance(combined_tags, str):
            combined_tags = [combined_tags]
        elif not isinstance(combined_tags, list):
            combined_tags = []
        
        # Add tags from inline metadata
        inline_tag_values = [tag.value for tag in inline_result.tags if tag.key == 'tag']
        combined_tags.extend(inline_tag_values)
        
        # Remove duplicates while preserving order
        unique_tags = []
        seen_tags = set()
        for tag in combined_tags:
            if tag not in seen_tags:
                unique_tags.append(tag)
                seen_tags.add(tag)
        
        # Convert inline metadata items to dictionary
        inline_metadata_dict = {item.key: item.value for item in inline_result.metadata}
        
        # Create document metadata
        document_metadata = DocumentMetadata(
            document_id=document_id,
            title=frontmatter_result.metadata.get('title'),
            author=frontmatter_result.metadata.get('author'),
            tags=unique_tags,
            frontmatter=frontmatter_result.metadata,
            inline_metadata=inline_metadata_dict,
            inline_tags=inline_result.tags
        )
        
        return MetadataExtractionResult(
            document_metadata=document_metadata,
            frontmatter_result=frontmatter_result,
            inline_result=inline_result,
            content_without_frontmatter=frontmatter_result.content_without_frontmatter
        )
    
    def extract_frontmatter_only(self, content: str) -> FrontmatterResult:
        """Extract only frontmatter from content.
        
        Args:
            content: Document content
            
        Returns:
            FrontmatterResult with frontmatter data
        """
        return self.frontmatter_parser.parse(content)
    
    def extract_inline_only(self, content: str) -> InlineMetadataResult:
        """Extract only inline metadata from content.
        
        Args:
            content: Document content
            
        Returns:
            InlineMetadataResult with inline metadata
        """
        return self.inline_extractor.extract(content)
    
    def validate_document_metadata(self, content: str) -> Dict[str, Any]:
        """Validate metadata in document content.
        
        Args:
            content: Document content to validate
            
        Returns:
            Dictionary with validation results and any issues found
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'frontmatter_valid': True,
            'inline_metadata_valid': True
        }
        
        try:
            # Validate frontmatter
            frontmatter_valid = self.frontmatter_parser.validate_frontmatter(content)
            validation_results['frontmatter_valid'] = frontmatter_valid
            
            if not frontmatter_valid:
                validation_results['warnings'].append("Frontmatter appears malformed")
        
        except Exception as e:
            validation_results['frontmatter_valid'] = False
            validation_results['errors'].append(f"Frontmatter validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        try:
            # Validate inline metadata syntax
            inline_issues = self.inline_extractor.validate_metadata_syntax(content)
            if inline_issues:
                validation_results['inline_metadata_valid'] = False
                for issue_type, issues in inline_issues.items():
                    validation_results['warnings'].extend(issues)
                    if issue_type == 'json_comments':  # JSON errors are more serious
                        validation_results['is_valid'] = False
                        validation_results['errors'].extend(issues)
        
        except Exception as e:
            validation_results['inline_metadata_valid'] = False
            validation_results['errors'].append(f"Inline metadata validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about extraction performance.
        
        Returns:
            Dictionary with extraction statistics
        """
        inline_stats = self.inline_extractor.get_performance_stats()
        
        return {
            'inline_extraction_stats': inline_stats,
            'extractors_configured': {
                'frontmatter_parser': True,
                'inline_extractor': True,
                'performance_tracking': self.inline_extractor.enable_performance_tracking,
                'line_tracking': self.inline_extractor.enable_line_tracking
            }
        } 