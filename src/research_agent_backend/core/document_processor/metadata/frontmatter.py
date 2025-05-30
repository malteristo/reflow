"""
Frontmatter Parsing Module

Handles YAML and TOML frontmatter extraction from markdown documents.
Provides robust parsing with error handling and validation.

Implements FR-KB-003.1: Frontmatter metadata extraction.
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python versions
    except ImportError:
        tomllib = None

logger = logging.getLogger(__name__)


class FrontmatterParseError(Exception):
    """Exception raised when frontmatter parsing fails.
    
    This exception provides detailed information about frontmatter parsing
    failures, including the line number, parsing error details, and suggested
    fixes for common issues.
    
    Attributes:
        message: Description of the parsing error
        line_number: Line number where parsing failed (if available)
        frontmatter_type: Type of frontmatter that failed (yaml/toml)
        content_preview: Preview of problematic content for debugging
    """
    
    def __init__(
        self, 
        message: str, 
        line_number: Optional[int] = None,
        frontmatter_type: Optional[str] = None,
        content_preview: Optional[str] = None
    ):
        self.message = message
        self.line_number = line_number
        self.frontmatter_type = frontmatter_type
        self.content_preview = content_preview
        
        # Build comprehensive error message
        error_parts = [message]
        if frontmatter_type:
            error_parts.append(f"(Type: {frontmatter_type})")
        if line_number:
            error_parts.append(f"at line {line_number}")
        if content_preview:
            error_parts.append(f"Content: {content_preview[:100]}...")
            
        super().__init__(" ".join(error_parts))


@dataclass
class FrontmatterResult:
    """Result container for frontmatter parsing operations.
    
    Encapsulates the results of frontmatter parsing including the extracted
    metadata, content without frontmatter, and parsing metadata.
    
    Attributes:
        has_frontmatter: True if valid frontmatter was detected and parsed
        metadata: Dictionary containing parsed frontmatter data
        content_without_frontmatter: Document content with frontmatter removed
        frontmatter_type: Type of frontmatter detected ('yaml', 'toml', or None)
        parse_time_ms: Time taken to parse frontmatter in milliseconds
        warnings: List of non-fatal warnings encountered during parsing
    """
    has_frontmatter: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_without_frontmatter: str = ""
    frontmatter_type: Optional[str] = None
    parse_time_ms: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate result data after creation."""
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata)}")
        if not isinstance(self.content_without_frontmatter, str):
            raise ValueError(f"content_without_frontmatter must be a string, got {type(self.content_without_frontmatter)}")


class FrontmatterParser:
    """Parser for YAML and TOML frontmatter.
    
    Provides robust parsing of document frontmatter with comprehensive error
    handling and validation. Supports both YAML (---) and TOML (+++) formats.
    """
    
    def __init__(self):
        """Initialize parser with compiled regex patterns for performance."""
        self.yaml_pattern = re.compile(r'^---\s*\n(.*?\n)---\s*\n', re.DOTALL | re.MULTILINE)
        self.toml_pattern = re.compile(r'^\+\+\+\s*\n(.*?\n)\+\+\+\s*\n', re.DOTALL | re.MULTILINE)
    
    def parse(self, content: str) -> FrontmatterResult:
        """Parse frontmatter from content.
        
        Args:
            content: Document content potentially containing frontmatter
            
        Returns:
            FrontmatterResult with parsed data and metadata
            
        Raises:
            FrontmatterParseError: If frontmatter is malformed
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        start_time = time.time()
        
        # Try YAML first
        yaml_match = self.yaml_pattern.match(content)
        if yaml_match:
            if yaml is None:
                raise FrontmatterParseError("YAML parsing not available - install PyYAML package")
            
            try:
                frontmatter_content = yaml_match.group(1)
                metadata = yaml.safe_load(frontmatter_content) or {}
                content_without = content[yaml_match.end():]
                parse_time = (time.time() - start_time) * 1000
                
                return FrontmatterResult(
                    has_frontmatter=True,
                    metadata=metadata,
                    content_without_frontmatter=content_without,
                    frontmatter_type="yaml",
                    parse_time_ms=parse_time
                )
            except yaml.YAMLError as e:
                raise FrontmatterParseError(
                    f"Invalid YAML frontmatter: {e}",
                    frontmatter_type="yaml",
                    content_preview=frontmatter_content[:200]
                )
        
        # Try TOML 
        toml_match = self.toml_pattern.match(content)
        if toml_match:
            if tomllib is None:
                raise FrontmatterParseError("TOML parsing not available - install tomli package")
                
            try:
                frontmatter_content = toml_match.group(1)
                metadata = tomllib.loads(frontmatter_content)
                content_without = content[toml_match.end():]
                parse_time = (time.time() - start_time) * 1000
                
                return FrontmatterResult(
                    has_frontmatter=True,
                    metadata=metadata,
                    content_without_frontmatter=content_without,
                    frontmatter_type="toml",
                    parse_time_ms=parse_time
                )
            except Exception as e:
                raise FrontmatterParseError(
                    f"Invalid TOML frontmatter: {e}",
                    frontmatter_type="toml",
                    content_preview=frontmatter_content[:200]
                )
        
        # No frontmatter found
        parse_time = (time.time() - start_time) * 1000
        return FrontmatterResult(
            has_frontmatter=False,
            metadata={},
            content_without_frontmatter=content,
            frontmatter_type=None,
            parse_time_ms=parse_time
        )
    
    def validate_frontmatter(self, content: str) -> bool:
        """Check if content has valid frontmatter without full parsing.
        
        Args:
            content: Content to validate
            
        Returns:
            True if valid frontmatter detected, False otherwise
        """
        try:
            result = self.parse(content)
            return result.has_frontmatter
        except FrontmatterParseError:
            return False
    
    def extract_frontmatter_only(self, content: str) -> Optional[str]:
        """Extract raw frontmatter content without parsing.
        
        Args:
            content: Document content
            
        Returns:
            Raw frontmatter string or None if no frontmatter found
        """
        yaml_match = self.yaml_pattern.match(content)
        if yaml_match:
            return yaml_match.group(1)
        
        toml_match = self.toml_pattern.match(content)
        if toml_match:
            return toml_match.group(1)
        
        return None 