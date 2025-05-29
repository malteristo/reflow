"""
Markdown Parser Configuration Module

This module provides the MarkdownParserConfigurator class that applies configuration
to customize parser behavior, including rule generation, header splitting, and
performance optimizations.

Supports dynamic configuration of parsing behavior based on settings.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..core.document_processor import (
    MarkdownParser, Pattern, Rule, MarkdownParseError
)

logger = logging.getLogger(__name__)


@dataclass
class CustomPattern:
    """Configuration for custom markdown patterns."""
    name: str
    pattern: str
    replacement: str


class MarkdownParserConfigurator:
    """
    Configurator for customizing MarkdownParser behavior.
    
    Applies configuration-driven customizations to MarkdownParser instances,
    including header rules, atomic unit preservation, boundary strategies,
    and custom pattern definitions.
    
    Features:
    - Dynamic header rule generation based on configuration
    - Atomic unit preservation configuration
    - Custom pattern support
    - Performance mode optimizations
    - Boundary strategy-specific rule sets
    
    Example:
        >>> configurator = MarkdownParserConfigurator()
        >>> parser = MarkdownParser()
        >>> 
        >>> config = {
        ...     "markdown_headers_to_split_on": [["##", "H2"], ["###", "H3"]],
        ...     "handle_code_blocks_as_atomic": True,
        ...     "performance_mode": False
        ... }
        >>> 
        >>> configured_parser = configurator.configure_parser(parser, config)
    """
    
    def __init__(self) -> None:
        """Initialize the parser configurator."""
        self._pattern_cache: Dict[str, Pattern] = {}
        logger.debug("MarkdownParserConfigurator initialized")
    
    def configure_parser(self, parser: MarkdownParser, config: Dict[str, Any]) -> MarkdownParser:
        """
        Apply configuration to customize parser behavior.
        
        Args:
            parser: MarkdownParser instance to configure
            config: Configuration dictionary with parser settings
            
        Returns:
            Configured MarkdownParser instance
        """
        logger.debug(f"Configuring parser with {len(config)} settings")
        
        try:
            # Apply header rules configuration
            self.apply_header_rules(parser, config)
            
            # Apply atomic unit preservation rules
            self.apply_atomic_unit_rules(parser, config)
            
            # Apply boundary strategy configuration
            self._apply_boundary_strategy_rules(parser, config)
            
            # Apply custom patterns if defined
            self._apply_custom_patterns(parser, config)
            
            # Apply performance optimizations if enabled
            if config.get("performance_mode", False):
                self._apply_performance_optimizations(parser, config)
            
            logger.debug("Parser configuration applied successfully")
            return parser
            
        except Exception as e:
            logger.error(f"Failed to configure parser: {e}")
            raise MarkdownParseError(f"Parser configuration failed: {e}") from e
    
    def apply_header_rules(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply header-based splitting rules to parser.
        
        Args:
            parser: MarkdownParser to configure
            config: Configuration containing header settings
        """
        headers_config = config.get("markdown_headers_to_split_on", [])
        
        if not headers_config:
            logger.debug("No header configuration found, using defaults")
            return
        
        logger.debug(f"Applying header rules for {len(headers_config)} header levels")
        
        for header_spec in headers_config:
            if len(header_spec) >= 2:
                header_pattern = header_spec[0]
                header_tag = header_spec[1]
                
                try:
                    # Create header pattern - escape markdown characters properly
                    escaped_pattern = re.escape(header_pattern)
                    pattern_name = f"header_{header_tag.lower()}"
                    regex_pattern = f"^{escaped_pattern}\\s+(.+)$"
                    
                    # Create pattern instance
                    pattern = Pattern(pattern_name, regex_pattern)
                    
                    # Determine HTML tag name
                    if header_tag.startswith('H') and len(header_tag) == 2:
                        tag_name = header_tag.lower()  # H1 -> h1
                    else:
                        tag_name = header_tag.lower()
                    
                    # Create replacement rule
                    replacement = f"<{tag_name}>\\1</{tag_name}>"
                    rule = Rule(pattern, replacement)
                    
                    # Add rule to parser
                    parser.add_rule(rule)
                    logger.debug(f"Added header rule: {header_pattern} -> {tag_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to add header rule for {header_pattern}: {e}")
    
    def apply_atomic_unit_rules(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply atomic unit preservation rules to parser.
        
        Args:
            parser: MarkdownParser to configure
            config: Configuration containing atomic unit settings
        """
        logger.debug("Applying atomic unit preservation rules")
        
        # Code block preservation
        if config.get("handle_code_blocks_as_atomic", True) or config.get("preserve_code_blocks", True):
            self._add_code_block_rules(parser)
        
        # Table preservation
        if config.get("handle_tables_as_atomic", True) or config.get("preserve_tables", True):
            self._add_table_rules(parser)
        
        # Link preservation
        if config.get("preserve_links", True):
            self._add_link_rules(parser)
        
        # List preservation
        if config.get("preserve_lists", True):
            self._add_list_rules(parser)
        
        # Blockquote preservation
        if config.get("preserve_blockquotes", True):
            self._add_blockquote_rules(parser)
    
    def _apply_boundary_strategy_rules(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply boundary strategy-specific rules.
        
        Args:
            parser: MarkdownParser to configure
            config: Configuration containing boundary strategy settings
        """
        boundary_strategy = config.get("boundary_strategy", "intelligent")
        
        logger.debug(f"Applying rules for boundary strategy: {boundary_strategy}")
        
        if boundary_strategy == "markup_aware":
            self._add_markup_aware_rules(parser)
        elif boundary_strategy == "sentence_only":
            self._add_sentence_boundary_rules(parser)
        elif boundary_strategy == "paragraph_only":
            self._add_paragraph_boundary_rules(parser)
        
        # Apply sentence/paragraph preservation settings
        if config.get("preserve_sentences", True):
            self._add_sentence_preservation_rules(parser)
        
        if config.get("preserve_paragraphs", True):
            self._add_paragraph_preservation_rules(parser)
    
    def _apply_custom_patterns(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply custom patterns from configuration.
        
        Args:
            parser: MarkdownParser to configure
            config: Configuration containing custom patterns
        """
        custom_patterns = config.get("custom_patterns", [])
        
        if not custom_patterns:
            return
        
        logger.debug(f"Applying {len(custom_patterns)} custom patterns")
        
        for pattern_config in custom_patterns:
            try:
                if isinstance(pattern_config, dict):
                    name = pattern_config.get("name")
                    pattern_str = pattern_config.get("pattern")
                    replacement = pattern_config.get("replacement")
                    
                    if name and pattern_str and replacement is not None:
                        pattern = Pattern(name, pattern_str)
                        rule = Rule(pattern, replacement)
                        parser.add_rule(rule)
                        logger.debug(f"Added custom pattern: {name}")
                    else:
                        logger.warning(f"Invalid custom pattern config: {pattern_config}")
                        
            except Exception as e:
                logger.warning(f"Failed to add custom pattern: {e}")
    
    def _apply_performance_optimizations(self, parser: MarkdownParser, config: Dict[str, Any]) -> None:
        """
        Apply performance optimizations to parser.
        
        Args:
            parser: MarkdownParser to optimize
            config: Configuration containing performance settings
        """
        logger.debug("Applying performance optimizations")
        
        # Cache compiled patterns if enabled
        if config.get("cache_compiled_patterns", True):
            # Patterns are already cached in Pattern class
            pass
        
        # Enable fast parsing mode
        if config.get("enable_fast_parsing", False):
            # Could implement simplified rule sets for faster parsing
            logger.debug("Fast parsing mode enabled")
    
    def _add_code_block_rules(self, parser: MarkdownParser) -> None:
        """Add code block preservation rules."""
        try:
            # Fenced code blocks
            fenced_pattern = Pattern(
                "code_block_fenced",
                r"```(\w+)?\n(.*?)\n```"
            )
            fenced_rule = Rule(
                fenced_pattern,
                lambda match: f"<pre><code class=\"{match.group(1) or 'text'}\">{match.group(2)}</code></pre>"
            )
            parser.add_rule(fenced_rule)
            
            # Inline code
            inline_pattern = Pattern("code_inline", r"`([^`]+)`")
            inline_rule = Rule(inline_pattern, r"<code>\1</code>")
            parser.add_rule(inline_rule)
            
            logger.debug("Added code block preservation rules")
            
        except Exception as e:
            logger.warning(f"Failed to add code block rules: {e}")
    
    def _add_table_rules(self, parser: MarkdownParser) -> None:
        """Add table preservation rules."""
        try:
            # Simple table pattern
            table_pattern = Pattern(
                "table_simple",
                r"(\|.*\|.*\n)+"
            )
            table_rule = Rule(
                table_pattern,
                lambda match: f"<table>{self._process_table_content(match.group(0))}</table>"
            )
            parser.add_rule(table_rule)
            
            logger.debug("Added table preservation rules")
            
        except Exception as e:
            logger.warning(f"Failed to add table rules: {e}")
    
    def _add_link_rules(self, parser: MarkdownParser) -> None:
        """Add link preservation rules."""
        try:
            # Markdown links
            link_pattern = Pattern(
                "link_markdown",
                r"\[([^\]]+)\]\(([^)]+)\)"
            )
            link_rule = Rule(link_pattern, r'<a href="\2">\1</a>')
            parser.add_rule(link_rule)
            
            logger.debug("Added link preservation rules")
            
        except Exception as e:
            logger.warning(f"Failed to add link rules: {e}")
    
    def _add_list_rules(self, parser: MarkdownParser) -> None:
        """Add list preservation rules."""
        try:
            # Bullet lists
            bullet_pattern = Pattern(
                "list_bullet",
                r"^[-*+]\s+(.+)$"
            )
            bullet_rule = Rule(bullet_pattern, r"<li>\1</li>")
            parser.add_rule(bullet_rule)
            
            # Numbered lists
            numbered_pattern = Pattern(
                "list_numbered",
                r"^\d+\.\s+(.+)$"
            )
            numbered_rule = Rule(numbered_pattern, r"<li>\1</li>")
            parser.add_rule(numbered_rule)
            
            logger.debug("Added list preservation rules")
            
        except Exception as e:
            logger.warning(f"Failed to add list rules: {e}")
    
    def _add_blockquote_rules(self, parser: MarkdownParser) -> None:
        """Add blockquote preservation rules."""
        try:
            blockquote_pattern = Pattern(
                "blockquote",
                r"^>\s+(.+)$"
            )
            blockquote_rule = Rule(blockquote_pattern, r"<blockquote>\1</blockquote>")
            parser.add_rule(blockquote_rule)
            
            logger.debug("Added blockquote preservation rules")
            
        except Exception as e:
            logger.warning(f"Failed to add blockquote rules: {e}")
    
    def _add_markup_aware_rules(self, parser: MarkdownParser) -> None:
        """Add markup-aware boundary rules."""
        logger.debug("Adding markup-aware boundary rules")
        # These rules help with boundary detection but don't transform content
        # Implementation would depend on specific markup awareness requirements
    
    def _add_sentence_boundary_rules(self, parser: MarkdownParser) -> None:
        """Add sentence boundary detection rules."""
        logger.debug("Adding sentence boundary rules")
        # Implementation for sentence boundary detection
    
    def _add_paragraph_boundary_rules(self, parser: MarkdownParser) -> None:
        """Add paragraph boundary detection rules."""
        logger.debug("Adding paragraph boundary rules")
        # Implementation for paragraph boundary detection
    
    def _add_sentence_preservation_rules(self, parser: MarkdownParser) -> None:
        """Add sentence preservation rules."""
        logger.debug("Adding sentence preservation rules")
        # Implementation for sentence preservation
    
    def _add_paragraph_preservation_rules(self, parser: MarkdownParser) -> None:
        """Add paragraph preservation rules."""
        logger.debug("Adding paragraph preservation rules")
        # Implementation for paragraph preservation
    
    def _process_table_content(self, table_text: str) -> str:
        """
        Process table content to generate HTML.
        
        Args:
            table_text: Raw table text
            
        Returns:
            Processed HTML table content
        """
        # Simple table processing
        lines = table_text.strip().split('\n')
        html_rows = []
        
        for line in lines:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                row_html = ''.join(f"<td>{cell}</td>" for cell in cells)
                html_rows.append(f"<tr>{row_html}</tr>")
        
        return ''.join(html_rows)
    
    def get_supported_config_keys(self) -> List[str]:
        """
        Get list of supported configuration keys.
        
        Returns:
            List of configuration keys supported by this configurator
        """
        return [
            "markdown_headers_to_split_on",
            "handle_code_blocks_as_atomic",
            "handle_tables_as_atomic", 
            "preserve_code_blocks",
            "preserve_tables",
            "preserve_links",
            "preserve_lists",
            "preserve_blockquotes",
            "preserve_sentences",
            "preserve_paragraphs",
            "boundary_strategy",
            "custom_patterns",
            "performance_mode",
            "cache_compiled_patterns",
            "enable_fast_parsing"
        ] 