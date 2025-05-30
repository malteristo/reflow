"""Integration tests for atomic unit system with existing chunking pipeline."""

import pytest
from research_agent_backend.core.document_processor.atomic_units import (
    AtomicUnitType, AtomicUnit, AtomicUnitHandler
)
from research_agent_backend.core.document_processor.chunking import ChunkConfig, RecursiveChunker
from research_agent_backend.core.document_processor import MarkdownParser
from research_agent_backend.core.document_processor.document_structure import HeaderBasedSplitter


class TestAtomicUnitIntegration:
    """Integration tests for atomic unit system with existing chunking pipeline."""
    
    def test_atomic_units_integrate_with_recursive_chunker(self):
        """Test atomic units work with the recursive chunking system."""
        # Create chunker with atomic unit preservation
        config = ChunkConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=40,  # Fixed: must be <= chunk_size/2 (50)
            preserve_code_blocks=True,
            preserve_tables=True
        )
        chunker = RecursiveChunker(config)
        
        text = """Introduction paragraph.

```python
def important_function():
    # This code should not be split
    return process_data()
```

Middle paragraph with some content.

| Important | Table |
|-----------|-------|
| Data      | Here  |

Conclusion paragraph."""
        
        # Detect atomic units first
        handler = AtomicUnitHandler()
        atomic_units = handler.detect_atomic_units(text)
        
        # Chunk text while preserving atomic units
        chunks = chunker.chunk_text(text)
        
        # Verify atomic units are preserved
        code_unit = next(u for u in atomic_units if u.unit_type == AtomicUnitType.CODE_BLOCK)
        table_unit = next(u for u in atomic_units if u.unit_type == AtomicUnitType.TABLE)
        
        # Code block should be in a single chunk
        code_chunks = [c for c in chunks if "important_function" in c.content]
        assert len(code_chunks) == 1
        assert "def important_function" in code_chunks[0].content
        assert "return process_data" in code_chunks[0].content
        
        # Table should be in a single chunk
        table_chunks = [c for c in chunks if "Important" in c.content and "Table" in c.content]
        assert len(table_chunks) == 1
    
    def test_atomic_units_with_document_tree_chunking(self):
        """Test atomic units work with document tree section chunking."""
        # Create document with sections containing atomic units
        markdown_text = """# Main Section

Regular paragraph in main section.

```sql
SELECT * FROM users
WHERE active = true;
```

## Subsection

| Feature | Status |
|---------|--------|
| Auth    | Done   |
| API     | Pending|

Final paragraph in subsection."""
        
        # Process through document tree pipeline
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        tree = splitter.split_and_build_tree(markdown_text)
        
        # Detect atomic units in each section
        handler = AtomicUnitHandler()
        for section in tree._sections:
            if section.content.strip():
                units = handler.detect_atomic_units(section.content)
                
                if "SELECT" in section.content:
                    # Should detect code block in main section
                    code_units = [u for u in units if u.unit_type == AtomicUnitType.CODE_BLOCK]
                    assert len(code_units) == 1
                    assert "SELECT * FROM users" in code_units[0].content
                
                if "Feature" in section.content:
                    # Should detect table in subsection
                    table_units = [u for u in units if u.unit_type == AtomicUnitType.TABLE]
                    assert len(table_units) == 1
                    assert "Feature" in table_units[0].content
                    assert "Status" in table_units[0].content
    
    def test_atomic_unit_preservation_with_overlap_calculation(self):
        """Test atomic unit boundaries are respected when calculating chunk overlaps."""
        config = ChunkConfig(
            chunk_size=100,
            chunk_overlap=30,
            min_chunk_size=40,  # Fixed: must be <= chunk_size/2 (50)
            preserve_code_blocks=True
        )
        chunker = RecursiveChunker(config)
        
        text = """Short intro.

```python
# Important code block
def process():
    return "result"
```

Short outro paragraph."""
        
        handler = AtomicUnitHandler()
        atomic_units = handler.detect_atomic_units(text)
        
        # Get preservation boundaries
        boundaries = handler.get_preservation_boundaries(text, atomic_units)
        
        # Chunks should respect atomic unit boundaries
        chunks = chunker.chunk_text(text)
        
        # Verify no chunk splits through a code block
        code_unit = next(u for u in atomic_units if u.unit_type == AtomicUnitType.CODE_BLOCK)
        
        for chunk in chunks:
            chunk_start = chunk.start_position
            chunk_end = chunk.end_position
            
            # If chunk overlaps with code block, it should contain the entire block
            if (chunk_start <= code_unit.start_position < chunk_end or 
                chunk_start < code_unit.end_position <= chunk_end):
                assert chunk_start <= code_unit.start_position
                assert chunk_end >= code_unit.end_position 