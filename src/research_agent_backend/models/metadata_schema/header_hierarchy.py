"""
Header hierarchy management for markdown document structure.

This module provides the HeaderHierarchy class for tracking and managing
hierarchical header structures in markdown documents.

Implements FR-KB-002.1: Header hierarchy tracking for context preservation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


@dataclass
class HeaderHierarchy:
    """
    Represents hierarchical header structure for Markdown documents.
    
    Implements FR-KB-002.1: Header hierarchy tracking for context preservation.
    """
    levels: List[str] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    
    def add_header(self, text: str, depth: int) -> None:
        """Add a header to the hierarchy."""
        self.levels.append(text.strip())
        self.depths.append(depth)
    
    def get_path(self) -> str:
        """Get formatted header path for display."""
        return " > ".join(self.levels) if self.levels else ""
    
    def get_context_at_depth(self, max_depth: int) -> List[str]:
        """Get header context up to specified depth."""
        return [level for level, depth in zip(self.levels, self.depths) if depth <= max_depth]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "levels": self.levels,
            "depths": self.depths,
            "path": self.get_path()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeaderHierarchy':
        """Create from dictionary."""
        hierarchy = cls()
        hierarchy.levels = data.get("levels", [])
        hierarchy.depths = data.get("depths", [])
        return hierarchy
    
    def to_json(self) -> str:
        """Convert to JSON string for ChromaDB storage."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HeaderHierarchy':
        """Create from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse header hierarchy JSON: {e}")
            return cls() 