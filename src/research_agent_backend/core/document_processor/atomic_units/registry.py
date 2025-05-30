"""
Atomic Unit Registry

Registry for managing atomic unit handlers.
"""

from typing import Dict, List, Any
from .types import AtomicUnitType
from .handlers import (
    CodeBlockHandler,
    TableHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler
)


class AtomicUnitRegistry:
    """Registry for managing atomic unit handlers."""
    
    def __init__(self) -> None:
        self._handlers: Dict[AtomicUnitType, Any] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default handlers for all unit types."""
        self._handlers[AtomicUnitType.CODE_BLOCK] = CodeBlockHandler()
        self._handlers[AtomicUnitType.TABLE] = TableHandler()
        self._handlers[AtomicUnitType.LIST] = ListHandler()
        self._handlers[AtomicUnitType.BLOCKQUOTE] = BlockquoteHandler()
        self._handlers[AtomicUnitType.PARAGRAPH] = ParagraphHandler()
    
    def register_handler(self, unit_type: AtomicUnitType, handler: Any) -> None:
        """Register a handler for a specific unit type."""
        self._handlers[unit_type] = handler
    
    def get_handler(self, unit_type: AtomicUnitType) -> Any:
        """Get handler for a specific unit type."""
        return self._handlers.get(unit_type)
    
    def has_handler(self, unit_type: AtomicUnitType) -> bool:
        """Check if a handler exists for a unit type."""
        return unit_type in self._handlers
    
    def get_supported_types(self) -> List[AtomicUnitType]:
        """Get list of all supported unit types."""
        return list(self._handlers.keys())
    
    def unregister_handler(self, unit_type: AtomicUnitType) -> None:
        """Unregister a handler for a specific unit type."""
        if unit_type in self._handlers:
            del self._handlers[unit_type] 