"""
Atomic Units Types

Core data structures for atomic content unit detection and processing.
Atomic units represent the smallest meaningful content segments that should be preserved
during document chunking operations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


class AtomicUnitType(Enum):
    """Enumeration of atomic content unit types."""
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    BLOCKQUOTE = "blockquote"
    HORIZONTAL_RULE = "horizontal_rule"
    YAML_FRONTMATTER = "yaml_frontmatter"
    MATH_BLOCK = "math_block"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class AtomicUnit:
    """Represents a single atomic content unit with enhanced validation."""
    unit_type: AtomicUnitType
    content: str
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate atomic unit after creation."""
        if not isinstance(self.unit_type, AtomicUnitType):
            raise ValueError(f"unit_type must be an AtomicUnitType, got {type(self.unit_type)}")
        
        if not isinstance(self.content, str):
            raise ValueError(f"content must be a string, got {type(self.content)}")
        
        if not isinstance(self.start_position, int) or self.start_position < 0:
            raise ValueError(f"start_position must be a non-negative integer, got {self.start_position}")
        
        if not isinstance(self.end_position, int) or self.end_position < 0:
            raise ValueError(f"end_position must be a non-negative integer, got {self.end_position}")
        
        if self.start_position >= self.end_position:
            raise ValueError(f"start_position ({self.start_position}) must be less than end_position ({self.end_position})")
        
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata)}")
    
    def get_length(self) -> int:
        """Get the length of the content."""
        return len(self.content)
    
    def contains_position(self, position: int) -> bool:
        """Check if a position falls within this atomic unit."""
        if not isinstance(position, int):
            raise ValueError(f"position must be an integer, got {type(position)}")
        return self.start_position <= position < self.end_position
    
    def get_boundaries(self) -> Tuple[int, int]:
        """Get the start and end boundaries of this unit."""
        return (self.start_position, self.end_position)
    
    def overlaps_with_range(self, start: int, end: int) -> bool:
        """Check if this unit overlaps with a given range."""
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("start and end must be integers")
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        
        return not (self.end_position <= start or self.start_position >= end)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atomic unit to dictionary representation."""
        return {
            "unit_type": self.unit_type.value,
            "content": self.content,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "length": self.get_length(),
            "metadata": self.metadata.copy()
        } 