"""
Main Atomic Unit Handler

Main handler for detecting and managing atomic content units with enhanced error handling.
"""

import logging
from typing import List, Dict, Any, Optional
from .types import AtomicUnit, AtomicUnitType
from .registry import AtomicUnitRegistry


class AtomicUnitHandler:
    """Main handler for detecting and managing atomic content units with enhanced error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize handler with configuration and registry."""
        self.config = config or {}
        self.registry = AtomicUnitRegistry()
        self.logger = logging.getLogger(__name__)
    
    def detect_atomic_units(self, text: str) -> List[AtomicUnit]:
        """Detect all atomic units in text with comprehensive error handling."""
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text)}")
        
        if not text.strip():
            self.logger.debug("Empty or whitespace-only text provided")
            return []
        
        units = []
        
        try:
            # Detect each type of atomic unit
            for unit_type in AtomicUnitType:
                handler = self.registry.get_handler(unit_type)
                if handler:
                    try:
                        type_units = handler.detect(text)
                        units.extend(type_units)
                        self.logger.debug(f"Detected {len(type_units)} {unit_type.value} units")
                    except Exception as e:
                        self.logger.warning(f"Error detecting {unit_type.value} units: {e}")
                        # Continue with other unit types
                        continue
            
            # Sort units by start position for consistent ordering
            units.sort(key=lambda u: u.start_position)
            
            self.logger.debug(f"Total atomic units detected: {len(units)}")
            return units
            
        except Exception as e:
            self.logger.error(f"Critical error in atomic unit detection: {e}")
            raise RuntimeError(f"Failed to detect atomic units: {e}") from e
    
    def get_preservation_boundaries(self, text: str, atomic_units: List[AtomicUnit]) -> List[Dict[str, int]]:
        """Get boundaries that should be preserved during chunking.
        
        Returns boundaries as dictionaries with 'start' and 'end' keys.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text)}")
        
        if not isinstance(atomic_units, list):
            raise ValueError(f"atomic_units must be a list, got {type(atomic_units)}")
        
        try:
            boundaries = []
            
            for unit in atomic_units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in atomic_units list: {type(unit)}")
                    continue
                
                # Add boundary as dictionary with start and end keys
                boundaries.append({
                    'start': unit.start_position,
                    'end': unit.end_position
                })
            
            # Sort boundaries by start position
            boundaries.sort(key=lambda x: x['start'])
            
            self.logger.debug(f"Generated {len(boundaries)} preservation boundaries")
            return boundaries
            
        except Exception as e:
            self.logger.error(f"Error generating preservation boundaries: {e}")
            return []
    
    def get_atomic_units_in_range(self, atomic_units: List[AtomicUnit], start: int, end: int) -> List[AtomicUnit]:
        """Get atomic units that overlap with a specific range."""
        if not isinstance(atomic_units, list):
            raise ValueError(f"atomic_units must be a list, got {type(atomic_units)}")
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("start and end must be integers")
        
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        
        overlapping_units = []
        
        try:
            for unit in atomic_units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in atomic_units list: {type(unit)}")
                    continue
                
                if unit.overlaps_with_range(start, end):
                    overlapping_units.append(unit)
            
            return overlapping_units
            
        except Exception as e:
            self.logger.error(f"Error getting units in range {start}-{end}: {e}")
            return []
    
    def merge_overlapping_units(self, units: List[AtomicUnit]) -> List[AtomicUnit]:
        """Merge overlapping atomic units of the same type."""
        if not isinstance(units, list):
            raise ValueError(f"units must be a list, got {type(units)}")
        
        if not units:
            return []
        
        try:
            # Group by unit type
            type_groups = {}
            for unit in units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in units list: {type(unit)}")
                    continue
                
                unit_type = unit.unit_type
                if unit_type not in type_groups:
                    type_groups[unit_type] = []
                type_groups[unit_type].append(unit)
            
            merged_units = []
            
            # Merge within each type group
            for unit_type, type_units in type_groups.items():
                # Sort by start position
                type_units.sort(key=lambda u: u.start_position)
                
                current_merged = type_units[0]
                
                for unit in type_units[1:]:
                    # Check if units overlap
                    if current_merged.end_position >= unit.start_position:
                        # Merge units - reconstruct content from unit boundaries
                        # Use the existing content plus additional content if needed
                        if unit.end_position > current_merged.end_position:
                            # Extend the merged content
                            merged_content = current_merged.content + unit.content[current_merged.end_position - unit.start_position:]
                        else:
                            # Unit is contained within current_merged
                            merged_content = current_merged.content
                        
                        current_merged = AtomicUnit(
                            unit_type=unit_type,
                            content=merged_content,
                            start_position=current_merged.start_position,
                            end_position=max(current_merged.end_position, unit.end_position),
                            metadata={**current_merged.metadata, **unit.metadata}
                        )
                    else:
                        # No overlap, add current and start new
                        merged_units.append(current_merged)
                        current_merged = unit
                
                merged_units.append(current_merged)
            
            # Sort final result by position
            merged_units.sort(key=lambda u: u.start_position)
            return merged_units
            
        except Exception as e:
            self.logger.error(f"Error merging overlapping units: {e}")
            return units  # Return original units if merge fails 