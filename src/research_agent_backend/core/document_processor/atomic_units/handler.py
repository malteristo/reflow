"""
Atomic Unit Handler

Main class for detecting, extracting, and managing atomic units in markdown text.
Coordinates with registered handlers to provide comprehensive atomic unit processing.
"""

from typing import List, Dict, Any, Optional, Tuple
from .types import AtomicUnit, AtomicUnitType
from .registry import AtomicUnitRegistry


class AtomicUnitHandler:
    """Main handler for atomic unit detection and management."""
    
    def __init__(self, registry: Optional[AtomicUnitRegistry] = None):
        """Initialize with registry."""
        self.registry = registry or AtomicUnitRegistry()
    
    def detect_units(self, text: str) -> List[AtomicUnit]:
        """Detect all atomic units in text."""
        units = []
        # Simplified implementation for now
        # TODO: Full implementation from document_processor.py
        return units
    
    def get_units_in_range(self, text: str, start: int, end: int) -> List[AtomicUnit]:
        """Get atomic units that overlap with a given range."""
        all_units = self.detect_units(text)
        return [unit for unit in all_units if unit.overlaps_with_range(start, end)]
    
    def merge_overlapping_units(self, units: List[AtomicUnit]) -> List[AtomicUnit]:
        """Merge overlapping atomic units."""
        if not units:
            return []
        
        # Sort by start position
        sorted_units = sorted(units, key=lambda u: u.start_position)
        merged = [sorted_units[0]]
        
        for current in sorted_units[1:]:
            last = merged[-1]
            if current.start_position <= last.end_position:
                # Overlapping units - merge them
                merged_content = last.content + current.content
                merged_metadata = {**last.metadata, **current.metadata}
                merged_unit = AtomicUnit(
                    unit_type=last.unit_type,
                    content=merged_content,
                    start_position=last.start_position,
                    end_position=max(last.end_position, current.end_position),
                    metadata=merged_metadata
                )
                merged[-1] = merged_unit
            else:
                merged.append(current)
        
        return merged 