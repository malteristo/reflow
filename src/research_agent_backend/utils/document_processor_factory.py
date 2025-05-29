"""
Document Processor Factory with Configuration Integration

This module provides the DocumentProcessorFactory class that creates configured
instances of document processing components based on configuration settings,
with plugin system support for extending parser functionality.

Implements configuration-driven component creation for the document processing pipeline.
"""

import logging
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable
from abc import ABC, abstractmethod

from .chunking_config_bridge import ChunkingConfigBridge
from ..core.document_processor import (
    MarkdownParser, HeaderBasedSplitter, RecursiveChunker,
    ChunkConfig, Pattern, Rule
)
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


@runtime_checkable
class DocumentProcessorPlugin(Protocol):
    """
    Protocol for document processor plugins.
    
    Plugins can extend parser functionality by implementing this protocol.
    """
    
    def apply_rules(self, parser: MarkdownParser) -> MarkdownParser:
        """
        Apply plugin rules to a MarkdownParser instance.
        
        Args:
            parser: MarkdownParser to extend
            
        Returns:
            Extended MarkdownParser instance
        """
        ...


class DocumentProcessorFactory:
    """
    Factory for creating configured document processing components.
    
    Creates instances of MarkdownParser, HeaderBasedSplitter, and RecursiveChunker
    based on configuration settings, with caching and plugin support.
    
    Features:
    - Configuration-driven component creation
    - Component caching for performance
    - Plugin system for extending functionality
    - Cache invalidation on configuration changes
    - Support for custom parser configurations
    
    Example:
        >>> config_manager = ConfigManager()
        >>> config_bridge = ChunkingConfigBridge(config_manager)
        >>> factory = DocumentProcessorFactory(config_bridge)
        >>> 
        >>> # Create configured components
        >>> parser = factory.create_markdown_parser()
        >>> chunker = factory.create_recursive_chunker()
        >>> 
        >>> # Create complete processing pipeline
        >>> pipeline = factory.create_processing_pipeline()
    """
    
    def __init__(self, config_bridge: ChunkingConfigBridge) -> None:
        """
        Initialize the document processor factory.
        
        Args:
            config_bridge: ChunkingConfigBridge for configuration access
            
        Raises:
            TypeError: If config_bridge is not a ChunkingConfigBridge instance
        """
        if not isinstance(config_bridge, ChunkingConfigBridge):
            raise TypeError(f"config_bridge must be ChunkingConfigBridge, got: {type(config_bridge)}")
        
        self.config_bridge = config_bridge
        
        # Component caches for performance
        self._parser_cache: Optional[MarkdownParser] = None
        self._splitter_cache: Optional[HeaderBasedSplitter] = None
        self._chunker_cache: Optional[RecursiveChunker] = None
        
        # Plugin registry
        self._plugins: Dict[str, DocumentProcessorPlugin] = {}
        
        logger.debug("DocumentProcessorFactory initialized")
    
    def create_markdown_parser(self, plugins: Optional[List[str]] = None) -> MarkdownParser:
        """
        Create a configured MarkdownParser instance.
        
        Args:
            plugins: Optional list of plugin names to apply
            
        Returns:
            Configured MarkdownParser instance
        """
        # Return cached instance if available and no plugins requested
        if self._parser_cache is not None and not plugins:
            logger.debug("Returning cached MarkdownParser")
            return self._parser_cache
        
        logger.debug("Creating new MarkdownParser instance")
        
        # Create base parser
        parser = MarkdownParser()
        
        # Apply configuration-based customizations
        parser = self._configure_parser(parser)
        
        # Apply plugins if requested
        if plugins:
            parser = self._apply_plugins(parser, plugins)
            # Don't cache if plugins are applied (may be unique configuration)
            logger.debug(f"Applied plugins: {plugins}")
            return parser
        
        # Cache for future use
        self._parser_cache = parser
        logger.debug("MarkdownParser created and cached")
        return parser
    
    def create_header_splitter(self) -> HeaderBasedSplitter:
        """
        Create a configured HeaderBasedSplitter instance.
        
        Returns:
            Configured HeaderBasedSplitter instance
        """
        # Return cached instance if available
        if self._splitter_cache is not None:
            logger.debug("Returning cached HeaderBasedSplitter")
            return self._splitter_cache
        
        logger.debug("Creating new HeaderBasedSplitter instance")
        
        # Create parser and splitter
        parser = self.create_markdown_parser()
        splitter = HeaderBasedSplitter(parser)
        
        # Cache for future use
        self._splitter_cache = splitter
        logger.debug("HeaderBasedSplitter created and cached")
        return splitter
    
    def create_recursive_chunker(self) -> RecursiveChunker:
        """
        Create a configured RecursiveChunker instance.
        
        Returns:
            Configured RecursiveChunker instance with ChunkConfig from configuration
        """
        # Return cached instance if available
        if self._chunker_cache is not None:
            logger.debug("Returning cached RecursiveChunker")
            return self._chunker_cache
        
        logger.debug("Creating new RecursiveChunker instance")
        
        # Get chunk configuration
        chunk_config = self.config_bridge.get_chunk_config()
        
        # Create chunker with configuration
        chunker = RecursiveChunker(chunk_config)
        
        # Cache for future use
        self._chunker_cache = chunker
        logger.debug(f"RecursiveChunker created and cached with config: {chunk_config}")
        return chunker
    
    def create_processing_pipeline(self) -> Dict[str, Any]:
        """
        Create a complete document processing pipeline.
        
        Returns:
            Dictionary containing configured parser, splitter, and chunker
        """
        logger.debug("Creating complete processing pipeline")
        
        pipeline = {
            "parser": self.create_markdown_parser(),
            "splitter": self.create_header_splitter(),
            "chunker": self.create_recursive_chunker()
        }
        
        logger.debug("Processing pipeline created successfully")
        return pipeline
    
    def register_plugin(self, name: str, plugin: DocumentProcessorPlugin) -> None:
        """
        Register a plugin for extending parser functionality.
        
        Args:
            name: Unique name for the plugin
            plugin: Plugin instance implementing DocumentProcessorPlugin protocol
            
        Raises:
            TypeError: If plugin doesn't implement the required protocol
            ValueError: If plugin name is already registered
        """
        if not isinstance(plugin, DocumentProcessorPlugin):
            raise TypeError(f"Plugin must implement DocumentProcessorPlugin protocol")
        
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        
        self._plugins[name] = plugin
        logger.debug(f"Plugin '{name}' registered successfully")
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Name of plugin to unregister
            
        Raises:
            KeyError: If plugin name is not registered
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not registered")
        
        del self._plugins[name]
        logger.debug(f"Plugin '{name}' unregistered")
    
    def get_registered_plugins(self) -> List[str]:
        """
        Get list of registered plugin names.
        
        Returns:
            List of registered plugin names
        """
        return list(self._plugins.keys())
    
    def invalidate_cache(self) -> None:
        """
        Invalidate all cached components.
        
        Forces recreation of components on next access. Useful when
        configuration changes and cached components need to be updated.
        """
        logger.debug("Invalidating component caches")
        self._parser_cache = None
        self._splitter_cache = None
        self._chunker_cache = None
        logger.debug("All caches invalidated")
    
    def _configure_parser(self, parser: MarkdownParser) -> MarkdownParser:
        """
        Apply configuration-based customizations to parser.
        
        Args:
            parser: MarkdownParser instance to configure
            
        Returns:
            Configured MarkdownParser instance
        """
        try:
            # Get configuration for parser customization
            chunking_config = self.config_bridge.config_manager.get("chunking_strategy", {})
            
            # Apply header configuration if specified
            markdown_headers = chunking_config.get("markdown_headers_to_split_on", [])
            if markdown_headers:
                self._apply_header_configuration(parser, markdown_headers)
            
            # Apply atomic unit handling configuration
            self._apply_atomic_unit_configuration(parser, chunking_config)
            
            # Apply performance optimizations if configured
            if chunking_config.get("performance_mode", False):
                self._apply_performance_optimizations(parser)
            
            logger.debug("Parser configuration applied successfully")
            return parser
            
        except Exception as e:
            logger.warning(f"Failed to apply parser configuration: {e}")
            return parser
    
    def _apply_header_configuration(self, parser: MarkdownParser, headers_config: List[List[str]]) -> None:
        """
        Apply header splitting configuration to parser.
        
        Args:
            parser: MarkdownParser to configure
            headers_config: List of [pattern, replacement] pairs for headers
        """
        logger.debug(f"Applying header configuration: {headers_config}")
        
        for header_spec in headers_config:
            if len(header_spec) >= 2:
                header_pattern = header_spec[0]
                header_tag = header_spec[1]
                
                # Create header pattern and rule
                pattern = Pattern(
                    f"header_{header_tag.lower()}", 
                    f"^{re.escape(header_pattern)}\\s+(.+)$"
                )
                
                # Convert H1, H2, etc. to h1, h2, etc.
                tag_name = header_tag.lower() if header_tag.startswith('H') else header_tag
                rule = Rule(pattern, f"<{tag_name}>\\1</{tag_name}>")
                
                parser.add_rule(rule)
    
    def _apply_atomic_unit_configuration(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply atomic unit handling configuration.
        
        Args:
            parser: MarkdownParser to configure
            config: Configuration dictionary
        """
        # Code block preservation
        if config.get("handle_code_blocks_as_atomic", True) or config.get("preserve_code_blocks", True):
            logger.debug("Enabling code block preservation")
            # Code block patterns are already in default rules
        
        # Table preservation
        if config.get("handle_tables_as_atomic", True) or config.get("preserve_tables", True):
            logger.debug("Enabling table preservation")
            # Table patterns are already in default rules
        
        # Link preservation
        if config.get("preserve_links", True):
            logger.debug("Enabling link preservation")
            # Link patterns are already in default rules
    
    def _apply_performance_optimizations(self, parser: MarkdownParser) -> None:
        """
        Apply performance optimizations to parser.
        
        Args:
            parser: MarkdownParser to optimize
        """
        logger.debug("Applying performance optimizations to parser")
        # Performance optimizations could include:
        # - Reducing number of rules
        # - Optimizing regex patterns
        # - Caching compiled patterns
        # Current implementation relies on existing optimizations
    
    def _apply_plugins(self, parser: MarkdownParser, plugin_names: List[str]) -> MarkdownParser:
        """
        Apply specified plugins to parser.
        
        Args:
            parser: MarkdownParser to extend
            plugin_names: List of plugin names to apply
            
        Returns:
            Extended MarkdownParser instance
        """
        for plugin_name in plugin_names:
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                parser = plugin.apply_rules(parser)
                logger.debug(f"Applied plugin: {plugin_name}")
            else:
                logger.warning(f"Plugin '{plugin_name}' not found, skipping")
        
        return parser
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """
        Get factory usage statistics.
        
        Returns:
            Dictionary with factory statistics
        """
        return {
            "cached_components": {
                "parser": self._parser_cache is not None,
                "splitter": self._splitter_cache is not None,
                "chunker": self._chunker_cache is not None
            },
            "registered_plugins": len(self._plugins),
            "plugin_names": list(self._plugins.keys())
        }


# Import re module for header configuration
import re 