"""
Atomic Unit Registry

Registry system for managing different atomic unit type handlers.
Provides pluggable architecture for atomic unit detection and processing.
"""

from typing import Dict, Any, List, Optional, Type
from .types import AtomicUnitType, AtomicUnit
from .handlers import (
    CodeBlockHandler,
    TableHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler
)


class AtomicUnitRegistry:
    """Registry for atomic unit handlers."""
    
    def __init__(self):
        """Initialize registry with default handlers."""
        self._handlers: Dict[AtomicUnitType, Any] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default handlers for all unit types."""
        self.register(AtomicUnitType.CODE_BLOCK, CodeBlockHandler())
        self.register(AtomicUnitType.TABLE, TableHandler())
        self.register(AtomicUnitType.LIST, ListHandler())
        self.register(AtomicUnitType.BLOCKQUOTE, BlockquoteHandler())
        self.register(AtomicUnitType.PARAGRAPH, ParagraphHandler())
    
    def register(self, unit_type: AtomicUnitType, handler: Any):
        """Register a handler for a unit type."""
        self._handlers[unit_type] = handler
    
    def get(self, unit_type: AtomicUnitType) -> Optional[Any]:
        """Get handler for a unit type."""
        return self._handlers.get(unit_type)
    
    def has(self, unit_type: AtomicUnitType) -> bool:
        """Check if handler exists for unit type."""
        return unit_type in self._handlers
    
    def unregister(self, unit_type: AtomicUnitType):
        """Remove handler for unit type."""
        if unit_type in self._handlers:
            del self._handlers[unit_type]
    
    def get_supported_types(self) -> List[AtomicUnitType]:
        """Get list of supported unit types."""
        return list(self._handlers.keys()) 