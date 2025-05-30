"""
Research Agent CLI Package.

This package contains the command-line interface for the Research Agent,
providing access to knowledge base management, collection operations,
project management, and RAG querying capabilities.

Implements FR-SI-001: Local CLI interface for core operations.
"""

from .cli import app, cli_main

__all__ = ["app", "cli_main"]
