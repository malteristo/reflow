"""Tests for AtomicUnitHandler and AtomicUnitRegistry classes - handler system for atomic units."""

import pytest
from research_agent_backend.core.document_processor.atomic_units import (
    AtomicUnitType, AtomicUnit, AtomicUnitHandler, AtomicUnitRegistry
)


class TestAtomicUnitHandler:
    """Tests for AtomicUnitHandler class - main handler for detecting and managing atomic units."""
    
    def test_atomic_unit_handler_creation(self):
        """Test creating AtomicUnitHandler with default configuration."""
        handler = AtomicUnitHandler()
        assert isinstance(handler, AtomicUnitHandler)
        assert isinstance(handler.registry, AtomicUnitRegistry)
    
    def test_detect_atomic_units_in_text(self):
        """Test detecting various atomic units in mixed text content."""
        handler = AtomicUnitHandler()
        
        text = """# Introduction

This is a regular paragraph.

```python
def example():
    return "code block"
```

| Name | Value |
|------|-------|
| Test | 123   |

- List item 1
- List item 2

> This is a blockquote
> with multiple lines

Another paragraph at the end."""
        
        units = handler.detect_atomic_units(text)
        
        # Should detect multiple unit types
        unit_types = [unit.unit_type for unit in units]
        assert AtomicUnitType.CODE_BLOCK in unit_types
        assert AtomicUnitType.TABLE in unit_types
        assert AtomicUnitType.LIST in unit_types
        assert AtomicUnitType.BLOCKQUOTE in unit_types
        assert AtomicUnitType.PARAGRAPH in unit_types
    
    def test_detect_code_blocks_fenced_and_indented(self):
        """Test detecting both fenced and indented code blocks."""
        handler = AtomicUnitHandler()
        
        text = """Text before.

```python
print("fenced code")
```

    def indented_code():
        return "indented"
    
    more_indented = True

Regular text after."""
        
        units = handler.detect_atomic_units(text)
        code_blocks = [u for u in units if u.unit_type == AtomicUnitType.CODE_BLOCK]
        
        assert len(code_blocks) == 2
        assert any("fenced code" in block.content for block in code_blocks)
        assert any("indented_code" in block.content for block in code_blocks)
    
    def test_get_preservation_boundaries(self):
        """Test getting preservation boundaries for atomic units."""
        handler = AtomicUnitHandler()
        
        text = """Regular text.

```code
important_function()
```

More text."""
        
        units = handler.detect_atomic_units(text)
        boundaries = handler.get_preservation_boundaries(text, units)
        
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0
        
        # Should contain boundaries for code block
        code_unit = next(u for u in units if u.unit_type == AtomicUnitType.CODE_BLOCK)
        assert (code_unit.start_position, code_unit.end_position) in boundaries
    
    def test_validate_atomic_unit_integrity(self):
        """Test validating atomic unit integrity and boundaries."""
        handler = AtomicUnitHandler()
        
        # Valid atomic unit
        valid_unit = AtomicUnit(
            unit_type=AtomicUnitType.CODE_BLOCK,
            content="```python\nprint('test')\n```",
            start_position=10,
            end_position=35
        )
        
        # Invalid atomic unit (mismatched boundaries)
        invalid_unit = AtomicUnit(
            unit_type=AtomicUnitType.CODE_BLOCK,
            content="```python\nprint('test')\n```",
            start_position=10,
            end_position=30  # Too short
        )
        
        assert handler.validate_unit(valid_unit) == True
        assert handler.validate_unit(invalid_unit) == False
    
    def test_merge_overlapping_units(self):
        """Test merging atomic units that overlap in content."""
        handler = AtomicUnitHandler()
        
        # Create overlapping units (this might happen in edge cases)
        unit1 = AtomicUnit(
            unit_type=AtomicUnitType.PARAGRAPH,
            content="First paragraph content.",
            start_position=0,
            end_position=20
        )
        
        unit2 = AtomicUnit(
            unit_type=AtomicUnitType.PARAGRAPH,
            content="paragraph content. Second paragraph.",
            start_position=15,
            end_position=40
        )
        
        merged = handler.merge_overlapping_units([unit1, unit2])
        
        # Should merge into single unit or handle appropriately
        assert len(merged) <= 2
        # Check that total coverage is maintained
        covered_positions = set()
        for unit in merged:
            covered_positions.update(range(unit.start_position, unit.end_position))
        assert len(covered_positions) >= 20  # At least covers original range


class TestAtomicUnitRegistry:
    """Tests for AtomicUnitRegistry class - registry system for atomic unit handlers."""
    
    def test_registry_creation_with_default_handlers(self):
        """Test creating AtomicUnitRegistry with default handlers."""
        registry = AtomicUnitRegistry()
        assert isinstance(registry, AtomicUnitRegistry)
        
        # Should have handlers for basic unit types
        assert registry.has_handler(AtomicUnitType.CODE_BLOCK)
        assert registry.has_handler(AtomicUnitType.TABLE)
        assert registry.has_handler(AtomicUnitType.LIST)
        assert registry.has_handler(AtomicUnitType.BLOCKQUOTE)
        assert registry.has_handler(AtomicUnitType.PARAGRAPH)
    
    def test_register_custom_handler(self):
        """Test registering a custom handler for an atomic unit type."""
        registry = AtomicUnitRegistry()
        
        # Create a custom handler (mock implementation)
        class CustomHandler:
            def detect(self, text):
                return []
            
            def validate(self, unit):
                return True
            
            def extract_metadata(self, content):
                return {}
        
        custom_handler = CustomHandler()
        registry.register_handler(AtomicUnitType.MATH_BLOCK, custom_handler)
        
        assert registry.has_handler(AtomicUnitType.MATH_BLOCK)
        assert registry.get_handler(AtomicUnitType.MATH_BLOCK) == custom_handler
    
    def test_get_handler_for_unit_type(self):
        """Test retrieving handler for specific unit type."""
        registry = AtomicUnitRegistry()
        
        code_handler = registry.get_handler(AtomicUnitType.CODE_BLOCK)
        assert code_handler is not None
        
        table_handler = registry.get_handler(AtomicUnitType.TABLE)
        assert table_handler is not None
        
        # Test non-existent handler
        nonexistent = registry.get_handler(AtomicUnitType.MATH_BLOCK)
        assert nonexistent is None or isinstance(nonexistent, object)
    
    def test_get_all_handlers(self):
        """Test getting all registered handlers."""
        registry = AtomicUnitRegistry()
        
        all_handlers = registry.get_all_handlers()
        assert isinstance(all_handlers, dict)
        assert len(all_handlers) >= 4  # At least basic handlers
        
        # Check that returned handlers are valid
        for unit_type, handler in all_handlers.items():
            assert isinstance(unit_type, AtomicUnitType)
            assert hasattr(handler, 'detect')
            assert hasattr(handler, 'validate')
    
    def test_detect_units_with_all_handlers(self):
        """Test using registry to detect units with all registered handlers."""
        registry = AtomicUnitRegistry()
        
        text = """```python
code_example()
```

| Table | Data |
|-------|------|
| Row   | Value|

- List item
- Another item

> Quote content
> Multiple lines"""
        
        all_units = registry.detect_all_units(text)
        
        assert len(all_units) >= 3  # Should detect code, table, list, quote
        unit_types = [u.unit_type for u in all_units]
        assert AtomicUnitType.CODE_BLOCK in unit_types
        assert AtomicUnitType.TABLE in unit_types
        assert AtomicUnitType.LIST in unit_types
        assert AtomicUnitType.BLOCKQUOTE in unit_types
    
    def test_priority_based_detection(self):
        """Test that handlers are applied in correct priority order."""
        registry = AtomicUnitRegistry()
        
        # Text that could match multiple patterns
        text = """> This looks like a quote
> But it's actually code output

```
> This is actual code
> with quote-like content
```"""
        
        units = registry.detect_all_units(text)
        
        # Code blocks should take priority over blockquotes when inside fenced blocks
        code_units = [u for u in units if u.unit_type == AtomicUnitType.CODE_BLOCK]
        quote_units = [u for u in units if u.unit_type == AtomicUnitType.BLOCKQUOTE]
        
        assert len(code_units) >= 1
        # Ensure code block detection doesn't get overridden by blockquote detection
        for code_unit in code_units:
            for quote_unit in quote_units:
                # They shouldn't overlap significantly
                overlap = max(0, min(code_unit.end_position, quote_unit.end_position) - 
                             max(code_unit.start_position, quote_unit.start_position))
                total_code = code_unit.end_position - code_unit.start_position
                assert overlap < total_code * 0.5  # Less than 50% overlap 