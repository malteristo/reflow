"""
Test Configuration Integration for Document Processing Pipeline

This module tests the integration between the ConfigManager system and 
the document processing components, ensuring configuration can drive
parser behavior and chunking strategies.

Implements tests for:
- ChunkingConfigBridge: Maps ConfigManager → ChunkConfig
- DocumentProcessorFactory: Creates configured processor instances  
- MarkdownParserConfigurator: Applies configuration to parser behavior
- Plugin system for extending parser functionality

Following TDD approach - these tests should FAIL until implementation is created.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# Import existing classes
from src.research_agent_backend.utils.config import ConfigManager
from src.research_agent_backend.core.document_processor import (
    ChunkConfig, BoundaryStrategy, MarkdownParser, 
    HeaderBasedSplitter, RecursiveChunker, DocumentTree
)


class TestChunkingConfigBridge:
    """
    Tests for ChunkingConfigBridge class that maps ConfigManager settings
    to ChunkConfig instances with validation and hot-reloading support.
    """
    
    def test_create_bridge_with_config_manager(self):
        """Test creating ChunkingConfigBridge with ConfigManager instance."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        config_manager = ConfigManager()
        bridge = ChunkingConfigBridge(config_manager)
        
        assert bridge.config_manager == config_manager
        assert hasattr(bridge, 'get_chunk_config')
        assert hasattr(bridge, 'reload_config')
    
    def test_map_config_to_chunk_config_basic(self):
        """Test mapping basic configuration to ChunkConfig instance."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        # Create temporary config file
        config_data = {
            "chunking_strategy": {
                "chunk_size": 800,
                "chunk_overlap": 100,
                "min_chunk_size": 50,
                "preserve_code_blocks": True,
                "preserve_tables": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            bridge = ChunkingConfigBridge(config_manager)
            chunk_config = bridge.get_chunk_config()
            
            assert isinstance(chunk_config, ChunkConfig)
            assert chunk_config.chunk_size == 800
            assert chunk_config.chunk_overlap == 100
            assert chunk_config.min_chunk_size == 50
            assert chunk_config.preserve_code_blocks == True
            assert chunk_config.preserve_tables == True
        finally:
            Path(config_file).unlink()
    
    def test_map_boundary_strategy_from_config(self):
        """Test mapping boundary strategy configuration to enum."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        config_data = {
            "chunking_strategy": {
                "boundary_strategy": "markup_aware",
                "chunk_size": 1000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            bridge = ChunkingConfigBridge(config_manager)
            chunk_config = bridge.get_chunk_config()
            
            assert chunk_config.boundary_strategy == BoundaryStrategy.MARKUP_AWARE
        finally:
            Path(config_file).unlink()
    
    def test_apply_config_overrides(self):
        """Test applying runtime configuration overrides."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        config_manager = ConfigManager()
        bridge = ChunkingConfigBridge(config_manager)
        
        overrides = {
            "chunk_size": 1500,
            "performance_mode": True,
            "debug_mode": True
        }
        
        chunk_config = bridge.get_chunk_config(overrides=overrides)
        
        assert chunk_config.chunk_size == 1500
        assert chunk_config.performance_mode == True
        assert chunk_config.debug_mode == True
    
    def test_validate_config_against_schema(self):
        """Test configuration validation against schema."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        # Invalid configuration
        config_data = {
            "chunking_strategy": {
                "chunk_size": -100,  # Invalid: negative size
                "chunk_overlap": 2000,  # Invalid: overlap > chunk_size
                "boundary_strategy": "invalid_strategy"  # Invalid enum value
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            bridge = ChunkingConfigBridge(config_manager)
            
            with pytest.raises(ValueError, match="Invalid chunking configuration"):
                bridge.get_chunk_config()
        finally:
            Path(config_file).unlink()
    
    def test_hot_reload_configuration_changes(self):
        """Test hot-reloading when configuration file changes."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        # Initial configuration
        config_data = {"chunking_strategy": {"chunk_size": 500}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            bridge = ChunkingConfigBridge(config_manager)
            
            initial_config = bridge.get_chunk_config()
            assert initial_config.chunk_size == 500
            
            # Update configuration
            updated_data = {"chunking_strategy": {"chunk_size": 1200}}
            with open(config_file, 'w') as f:
                json.dump(updated_data, f)
            
            bridge.reload_config()
            updated_config = bridge.get_chunk_config()
            assert updated_config.chunk_size == 1200
        finally:
            Path(config_file).unlink()
    
    def test_fallback_to_defaults_on_missing_config(self):
        """Test fallback to default ChunkConfig when chunking config is missing."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        config_data = {"other_section": {"some_value": "test"}}  # No chunking config
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            bridge = ChunkingConfigBridge(config_manager)
            chunk_config = bridge.get_chunk_config()
            
            # Should return ChunkConfig with default_config.json values merged
            assert isinstance(chunk_config, ChunkConfig)
            assert chunk_config.chunk_size == 512  # From default_config.json
            assert chunk_config.chunk_overlap == 50  # From default_config.json
            assert chunk_config.min_chunk_size == 100  # From default_config.json
        finally:
            Path(config_file).unlink()


class TestDocumentProcessorFactory:
    """
    Tests for DocumentProcessorFactory that creates configured instances
    of document processing components based on configuration settings.
    """
    
    def test_create_factory_with_config_bridge(self):
        """Test creating DocumentProcessorFactory with ChunkingConfigBridge."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_manager = ConfigManager()
        config_bridge = ChunkingConfigBridge(config_manager)
        factory = DocumentProcessorFactory(config_bridge)
        
        assert factory.config_bridge == config_bridge
        assert hasattr(factory, 'create_markdown_parser')
        assert hasattr(factory, 'create_header_splitter')
        assert hasattr(factory, 'create_recursive_chunker')
    
    def test_create_configured_markdown_parser(self):
        """Test creating MarkdownParser with configuration-driven rules."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_data = {
            "chunking_strategy": {
                "markdown_headers_to_split_on": [["##", "H2"], ["###", "H3"], ["####", "H4"]],
                "preserve_code_blocks": True,
                "preserve_tables": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            parser = factory.create_markdown_parser()
            
            assert isinstance(parser, MarkdownParser)
            # Parser should have rules configured based on settings
            assert parser.get_pattern_by_name("header") is not None
        finally:
            Path(config_file).unlink()
    
    def test_create_configured_header_splitter(self):
        """Test creating HeaderBasedSplitter with configuration."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_manager = ConfigManager()
        config_bridge = ChunkingConfigBridge(config_manager)
        factory = DocumentProcessorFactory(config_bridge)
        
        splitter = factory.create_header_splitter()
        
        assert isinstance(splitter, HeaderBasedSplitter)
        assert isinstance(splitter.parser, MarkdownParser)
    
    def test_create_configured_recursive_chunker(self):
        """Test creating RecursiveChunker with ChunkConfig from configuration."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_data = {
            "chunking_strategy": {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "boundary_strategy": "intelligent",
                "performance_mode": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            chunker = factory.create_recursive_chunker()
            
            assert isinstance(chunker, RecursiveChunker)
            assert chunker.config.chunk_size == 800
            assert chunker.config.chunk_overlap == 150
            assert chunker.config.boundary_strategy == BoundaryStrategy.INTELLIGENT
            assert chunker.config.performance_mode == False
        finally:
            Path(config_file).unlink()
    
    def test_create_processing_pipeline_from_config(self):
        """Test creating complete processing pipeline from configuration."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_data = {
            "chunking_strategy": {
                "type": "hybrid",
                "chunk_size": 600,
                "chunk_overlap": 100,
                "preserve_code_blocks": True,
                "preserve_tables": True,
                "boundary_strategy": "markup_aware"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            pipeline = factory.create_processing_pipeline()
            
            assert "parser" in pipeline
            assert "splitter" in pipeline
            assert "chunker" in pipeline
            assert isinstance(pipeline["parser"], MarkdownParser)
            assert isinstance(pipeline["splitter"], HeaderBasedSplitter)
            assert isinstance(pipeline["chunker"], RecursiveChunker)
        finally:
            Path(config_file).unlink()
    
    def test_factory_caches_components(self):
        """Test that factory caches created components for performance."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_manager = ConfigManager()
        config_bridge = ChunkingConfigBridge(config_manager)
        factory = DocumentProcessorFactory(config_bridge)
        
        parser1 = factory.create_markdown_parser()
        parser2 = factory.create_markdown_parser()
        
        # Should return same cached instance
        assert parser1 is parser2
    
    def test_factory_invalidates_cache_on_config_change(self):
        """Test that factory invalidates cache when configuration changes."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_data = {"chunking_strategy": {"chunk_size": 500}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            chunker1 = factory.create_recursive_chunker()
            
            # Change configuration
            updated_data = {"chunking_strategy": {"chunk_size": 1000}}
            with open(config_file, 'w') as f:
                json.dump(updated_data, f)
            
            factory.invalidate_cache()
            chunker2 = factory.create_recursive_chunker()
            
            # Should be different instances with different configs
            assert chunker1 is not chunker2
            assert chunker1.config.chunk_size != chunker2.config.chunk_size
        finally:
            Path(config_file).unlink()


class TestMarkdownParserConfigurator:
    """
    Tests for MarkdownParserConfigurator that applies configuration
    to customize parser behavior and rule generation.
    """
    
    def test_create_configurator(self):
        """Test creating MarkdownParserConfigurator."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        configurator = MarkdownParserConfigurator()
        
        assert hasattr(configurator, 'configure_parser')
        assert hasattr(configurator, 'apply_header_rules')
        assert hasattr(configurator, 'apply_atomic_unit_rules')
    
    def test_configure_header_splitting_rules(self):
        """Test configuring header-based splitting rules."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        config = {
            "markdown_headers_to_split_on": [
                ["#", "H1"],
                ["##", "H2"], 
                ["###", "H3"]
            ]
        }
        
        configurator = MarkdownParserConfigurator()
        parser = MarkdownParser()
        
        configured_parser = configurator.configure_parser(parser, config)
        
        # Should have header rules for H1, H2, H3
        assert configured_parser.get_pattern_by_name("header") is not None
        # Test that it can parse headers at different levels
        result = configured_parser.parse("# Main\n## Sub\n### Detail")
        assert "<h1>" in result and "<h2>" in result and "<h3>" in result
    
    def test_configure_atomic_unit_preservation(self):
        """Test configuring atomic unit preservation rules."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        config = {
            "handle_code_blocks_as_atomic": True,
            "handle_tables_as_atomic": True,
            "preserve_links": True
        }
        
        configurator = MarkdownParserConfigurator()
        parser = MarkdownParser()
        
        configured_parser = configurator.configure_parser(parser, config)
        
        # Test code block preservation
        code_text = "```python\ndef test():\n    pass\n```"
        result = configured_parser.parse(code_text)
        # Code blocks should be preserved as atomic units
        assert "```python" in result or "def test():" in result
    
    def test_configure_boundary_strategy_rules(self):
        """Test configuring boundary strategy-specific rules."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        config = {
            "boundary_strategy": "markup_aware",
            "preserve_sentences": True,
            "preserve_paragraphs": True
        }
        
        configurator = MarkdownParserConfigurator()
        parser = MarkdownParser()
        
        configured_parser = configurator.configure_parser(parser, config)
        
        # Should have rules that respect markup boundaries
        assert configured_parser is not None
        # Test that markup elements are handled appropriately
        markup_text = "This is **bold** and *italic* text."
        result = configured_parser.parse(markup_text)
        assert "<strong>" in result and "<em>" in result
    
    def test_dynamic_rule_generation(self):
        """Test dynamic generation of parsing rules based on configuration."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        config = {
            "custom_patterns": [
                {"name": "highlight", "pattern": r"==(.*?)==", "replacement": r"<mark>\1</mark>"},
                {"name": "strikethrough", "pattern": r"~~(.*?)~~", "replacement": r"<del>\1</del>"}
            ]
        }
        
        configurator = MarkdownParserConfigurator()
        parser = MarkdownParser()
        
        configured_parser = configurator.configure_parser(parser, config)
        
        # Should support custom patterns from configuration
        test_text = "This is ==highlighted== and ~~strikethrough~~ text."
        result = configured_parser.parse(test_text)
        assert "<mark>highlighted</mark>" in result
        assert "<del>strikethrough</del>" in result
    
    def test_performance_mode_optimization(self):
        """Test performance mode applies optimizations."""
        from src.research_agent_backend.utils.markdown_parser_configurator import MarkdownParserConfigurator
        
        config = {
            "performance_mode": True,
            "cache_compiled_patterns": True,
            "enable_fast_parsing": True
        }
        
        configurator = MarkdownParserConfigurator()
        parser = MarkdownParser()
        
        configured_parser = configurator.configure_parser(parser, config)
        
        # Performance mode should optimize parser behavior
        assert configured_parser is not None
        # Should have fewer, optimized rules for faster parsing


class TestConfigurationIntegration:
    """
    Integration tests for complete configuration system integration
    with document processing pipeline.
    """
    
    def test_end_to_end_configuration_workflow(self):
        """Test complete workflow from configuration to document processing."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        config_data = {
            "chunking_strategy": {
                "type": "hybrid",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "markdown_headers_to_split_on": [["##", "H2"], ["###", "H3"]],
                "handle_code_blocks_as_atomic": True,
                "boundary_strategy": "intelligent"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Complete workflow
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            pipeline = factory.create_processing_pipeline()
            
            # Test processing a document
            test_document = """
# Main Title

This is the introduction.

## Section One

Content for section one with some text.

```python
def example():
    return "code"
```

## Section Two

More content here.
"""
            
            parser = pipeline["parser"]
            splitter = pipeline["splitter"]
            chunker = pipeline["chunker"]
            
            # Parse and split by headers
            tree = splitter.split_and_build_tree(test_document)
            assert isinstance(tree, DocumentTree)
            assert tree.get_section_count() > 0
            
            # Chunk with configured settings
            chunked_sections = chunker.chunk_sections(tree)
            assert len(chunked_sections) > 0
            
            # Verify configuration was applied
            for section in chunked_sections:
                for chunk in section["chunks"]:
                    # Chunks should respect configured size
                    assert len(chunk.content) <= config_data["chunking_strategy"]["chunk_size"] + 100  # Some tolerance
        finally:
            Path(config_file).unlink()
    
    def test_plugin_system_registration(self):
        """Test plugin system for extending parser functionality."""
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        # Mock plugin class
        class CustomMarkdownPlugin:
            def apply_rules(self, parser):
                # Add custom rule to parser
                from src.research_agent_backend.core.document_processor import Pattern, Rule
                pattern = Pattern("custom", r"@([^@]+)@")
                rule = Rule(pattern, r"<custom>\1</custom>")
                parser.add_rule(rule)
                return parser
        
        config_manager = ConfigManager()
        factory = DocumentProcessorFactory(config_manager)
        
        # Register plugin
        plugin = CustomMarkdownPlugin()
        factory.register_plugin("custom_markdown", plugin)
        
        # Create parser with plugin
        parser = factory.create_markdown_parser(plugins=["custom_markdown"])
        
        # Test custom functionality
        result = parser.parse("This is @custom text@.")
        assert "<custom>custom text</custom>" in result
    
    def test_configuration_validation_errors(self):
        """Test proper error handling for invalid configurations."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        
        invalid_configs = [
            {"chunking_strategy": {"chunk_size": "invalid"}},  # Wrong type
            {"chunking_strategy": {"chunk_size": -100}},  # Negative value
            {"chunking_strategy": {"boundary_strategy": "nonexistent"}},  # Invalid enum
        ]
        
        for invalid_config in invalid_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(invalid_config, f)
                config_file = f.name
            
            try:
                config_manager = ConfigManager(config_file=config_file)
                bridge = ChunkingConfigBridge(config_manager)
                
                with pytest.raises((ValueError, TypeError)):
                    bridge.get_chunk_config()
            finally:
                Path(config_file).unlink()
    
    def test_hot_reload_processing_pipeline(self):
        """Test hot-reloading configuration affects processing pipeline."""
        from src.research_agent_backend.utils.chunking_config_bridge import ChunkingConfigBridge
        from src.research_agent_backend.utils.document_processor_factory import DocumentProcessorFactory
        
        # Initial configuration
        config_data = {"chunking_strategy": {"chunk_size": 300}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            config_bridge = ChunkingConfigBridge(config_manager)
            factory = DocumentProcessorFactory(config_bridge)
            
            chunker1 = factory.create_recursive_chunker()
            assert chunker1.config.chunk_size == 300
            
            # Update configuration file
            updated_data = {"chunking_strategy": {"chunk_size": 800}}
            with open(config_file, 'w') as f:
                json.dump(updated_data, f)
            
            # Hot reload
            config_bridge.reload_config()
            factory.invalidate_cache()
            
            chunker2 = factory.create_recursive_chunker()
            assert chunker2.config.chunk_size == 800
            
            # Should be different instances
            assert chunker1 is not chunker2
        finally:
            Path(config_file).unlink() 