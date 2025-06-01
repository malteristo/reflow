"""
Research Agent CLI Application.

Main entry point for the Research Agent command-line interface.
Implements command groups for knowledge base management, collection management,
project operations, and RAG querying.

Implements FR-SI-001: Local CLI interface for core operations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

from ..utils.config import ConfigManager
from ..exceptions.config_exceptions import ConfigurationError
from .knowledge_base import kb_app
from .collections import collections_app
from .projects import projects_app
from .query import query_app
from .model_management import model_app

# Initialize console for rich output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="research-agent",
    help="AI-powered research agent with local-first RAG capabilities",
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Add command groups
app.add_typer(kb_app, name="kb", help="Knowledge base management commands")
app.add_typer(collections_app, name="collections", help="Collection management commands")
app.add_typer(projects_app, name="projects", help="Project-specific operations")
app.add_typer(query_app, name="query", help="RAG querying commands")
app.add_typer(model_app, name="model", help="Model change detection and management")

# Global state
_config_manager: Optional[ConfigManager] = None
_logger: Optional[logging.Logger] = None
_global_config: dict = {}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create rich handler for beautiful console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(log_level)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )
    
    # Get our logger
    logger = logging.getLogger("research_agent")
    logger.setLevel(log_level)
    
    return logger


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create the global configuration manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ConfigManager instance
        
    Raises:
        typer.Exit: If configuration loading fails
    """
    global _config_manager
    
    if _config_manager is None or config_path:
        try:
            _config_manager = ConfigManager(
                config_file=config_path,
                load_env=True
            )
        except ConfigurationError as e:
            rprint(f"[red]Configuration Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            rprint(f"[red]Unexpected error loading configuration:[/red] {e}")
            raise typer.Exit(1)
    
    return _config_manager


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def get_global_config() -> dict:
    """Get the global configuration dictionary."""
    return _global_config


# Global callback for common options
@app.callback()
def main(
    ctx: typer.Context,
    config_path: Optional[str] = typer.Option(
        None,
        "--config-path",
        "-c",
        help="Path to configuration file (default: researchagent.config.json)",
        metavar="PATH",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing",
    ),
) -> None:
    """
    Research Agent CLI - AI-powered research with local-first RAG capabilities.
    
    The Research Agent provides intelligent document processing, embedding generation,
    and semantic search capabilities for your research and knowledge management needs.
    
    Common workflows:
    • Ingest documents: research-agent kb add-document path/to/doc.md
    • Create collections: research-agent collections create my-collection
    • Query knowledge: research-agent query "What is machine learning?"
    
    For detailed help on any command, use: research-agent <command> --help
    """
    # Set up logging first
    global _logger, _global_config
    _logger = setup_logging(verbose)
    
    # Store global options in app context and global state
    _global_config = {
        "config_path": config_path,
        "verbose": verbose,
        "dry_run": dry_run,
        "config_manager": None,  # Lazy-loaded when needed
        "logger": _logger,
    }
    
    # Store in context
    ctx.obj = _global_config.copy()
    
    # Initialize configuration manager (lazy-loaded)
    if config_path:
        try:
            get_config_manager(config_path)
        except typer.Exit:
            # Re-raise typer exits
            raise
        except Exception as e:
            _logger.error(f"Failed to load configuration: {e}")
            raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show system information and configuration status."""
    try:
        config_manager = get_config_manager()
        config = config_manager.config
        
        # Create info panel
        info_text = Text()
        info_text.append("Research Agent Information\n\n", style="bold blue")
        info_text.append(f"Version: {config.get('version', 'Unknown')}\n")
        info_text.append(f"Config file: {config_manager.config_file}\n")
        info_text.append(f"Project root: {config_manager.project_root}\n")
        info_text.append(f"Loaded: {'✓' if config_manager.is_loaded else '✗'}\n\n")
        
        info_text.append("Configuration:\n", style="bold")
        info_text.append(f"• Embedding model: {config.get('embedding_model', {}).get('name', 'Unknown')}\n")
        info_text.append(f"• Vector store: {config.get('vector_store', {}).get('type', 'Unknown')}\n")
        info_text.append(f"• Chunk size: {config.get('chunking_strategy', {}).get('chunk_size', 'Unknown')}\n")
        
        panel = Panel(info_text, title="System Information", border_style="blue")
        console.print(panel)
        
    except ConfigurationError as e:
        rprint(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    try:
        config_manager = get_config_manager()
        config = config_manager.config
        version = config.get('version', '0.1.0')
        rprint(f"Research Agent [blue]v{version}[/blue]")
    except Exception:
        # Fallback if config fails
        rprint("Research Agent [blue]v0.1.0[/blue]")


def handle_cli_error(error: Exception) -> None:
    """
    Handle CLI errors with user-friendly messages.
    
    Args:
        error: The exception that occurred
    """
    logger = get_logger()
    
    if isinstance(error, ConfigurationError):
        rprint(f"[red]Configuration Error:[/red] {error}")
        logger.debug("Configuration error details", exc_info=True)
    elif isinstance(error, FileNotFoundError):
        rprint(f"[red]File Not Found:[/red] {error}")
        logger.debug("File not found details", exc_info=True)
    elif isinstance(error, PermissionError):
        rprint(f"[red]Permission Denied:[/red] {error}")
        logger.debug("Permission error details", exc_info=True)
    else:
        rprint(f"[red]Error:[/red] {error}")
        logger.debug("Unexpected error details", exc_info=True)


def cli_main() -> None:
    """
    Main CLI entry point with error handling.
    
    This function is called by the console script entry point.
    """
    try:
        app()
    except typer.Exit as e:
        # Typer exits are expected, re-raise them
        raise
    except KeyboardInterrupt:
        rprint("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        handle_cli_error(e)
        raise typer.Exit(1)


if __name__ == "__main__":
    cli_main() 