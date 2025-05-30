"""
Tests for the Research Agent CLI framework.

This module tests the core CLI application structure, command routing,
error handling, and global options functionality.

Implements TDD testing for FR-SI-001: Local CLI interface.
"""

import pytest
import logging
import sys
import typer
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from pathlib import Path
import json

from research_agent_backend.cli.cli import (
    app,
    setup_logging,
    get_config_manager,
    get_logger,
    handle_cli_error,
    cli_main,
)
from research_agent_backend.exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
)


class TestCLIFramework:
    """Test suite for CLI framework functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.mock_config_data = {
            "version": "0.1.0",
            "embedding_model": {"name": "test-model"},
            "vector_store": {"type": "chromadb"},
            "chunking_strategy": {"chunk_size": 512},
        }
    
    def test_cli_app_creation(self):
        """Test that the main CLI app is properly created."""
        assert app.info.name == "research-agent"
        assert "AI-powered research agent" in app.info.help
        
        # Test that command groups are registered by checking help output
        result = self.runner.invoke(app, ["--help"])
        expected_groups = ["kb", "collections", "projects", "query"]
        
        for group in expected_groups:
            assert group in result.stdout, f"Command group '{group}' not found in help output"
    
    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            logger = setup_logging(verbose=True)
            
            # Should set DEBUG level for verbose mode
            mock_logger.setLevel.assert_called_with(logging.DEBUG)
            assert logger == mock_logger
    
    def test_setup_logging_normal(self):
        """Test logging setup with normal mode."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            logger = setup_logging(verbose=False)
            
            # Should set INFO level for normal mode
            mock_logger.setLevel.assert_called_with(logging.INFO)
            assert logger == mock_logger
    
    @patch('research_agent_backend.cli.cli.ConfigManager')
    def test_get_config_manager_success(self, mock_config_manager_class):
        """Test successful configuration manager creation."""
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        
        # Reset global state
        import research_agent_backend.cli.cli as cli_module
        cli_module._config_manager = None
        
        result = get_config_manager("test.json")
        
        assert result == mock_config_manager
        mock_config_manager_class.assert_called_once_with(
            config_file="test.json",
            load_env=True
        )
    
    @patch('research_agent_backend.cli.cli.ConfigManager')
    def test_get_config_manager_configuration_error(self, mock_config_manager_class):
        """Test configuration manager creation with configuration error."""
        mock_config_manager_class.side_effect = ConfigurationError("Test error")
        
        # Reset global state
        import research_agent_backend.cli.cli as cli_module
        cli_module._config_manager = None
        
        with pytest.raises(typer.Exit):
            get_config_manager("test.json")
    
    def test_get_logger(self):
        """Test logger retrieval."""
        # Reset global state
        import research_agent_backend.cli.cli as cli_module
        cli_module._logger = None
        
        with patch('research_agent_backend.cli.cli.setup_logging') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            result = get_logger()
            
            assert result == mock_logger
            mock_setup.assert_called_once_with()
    
    def test_handle_cli_error_configuration_error(self):
        """Test CLI error handling for configuration errors."""
        with patch('research_agent_backend.cli.cli.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('research_agent_backend.cli.cli.rprint') as mock_rprint:
                error = ConfigurationError("Test config error")
                handle_cli_error(error)
                
                mock_rprint.assert_called_once()
                args = mock_rprint.call_args[0][0]
                assert "Configuration Error" in args
                assert "Test config error" in args
                
                mock_logger.debug.assert_called_once()
    
    def test_handle_cli_error_file_not_found(self):
        """Test CLI error handling for file not found errors."""
        with patch('research_agent_backend.cli.cli.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('research_agent_backend.cli.cli.rprint') as mock_rprint:
                error = FileNotFoundError("Test file not found")
                handle_cli_error(error)
                
                mock_rprint.assert_called_once()
                args = mock_rprint.call_args[0][0]
                assert "File Not Found" in args
                assert "Test file not found" in args
    
    def test_handle_cli_error_permission_error(self):
        """Test CLI error handling for permission errors."""
        with patch('research_agent_backend.cli.cli.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('research_agent_backend.cli.cli.rprint') as mock_rprint:
                error = PermissionError("Test permission denied")
                handle_cli_error(error)
                
                mock_rprint.assert_called_once()
                args = mock_rprint.call_args[0][0]
                assert "Permission Denied" in args
                assert "Test permission denied" in args
    
    def test_handle_cli_error_generic(self):
        """Test CLI error handling for generic errors."""
        with patch('research_agent_backend.cli.cli.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('research_agent_backend.cli.cli.rprint') as mock_rprint:
                error = ValueError("Test generic error")
                handle_cli_error(error)
                
                mock_rprint.assert_called_once()
                args = mock_rprint.call_args[0][0]
                assert "Error" in args
                assert "Test generic error" in args
    
    @patch('research_agent_backend.cli.cli.app')
    def test_cli_main_success(self, mock_app):
        """Test successful CLI main execution."""
        mock_app.return_value = None
        
        # Should not raise any exceptions
        cli_main()
        mock_app.assert_called_once()
    
    @patch('research_agent_backend.cli.cli.app')
    def test_cli_main_keyboard_interrupt(self, mock_app):
        """Test CLI main with keyboard interrupt."""
        mock_app.side_effect = KeyboardInterrupt()
        
        with patch('research_agent_backend.cli.cli.rprint') as mock_rprint:
            with pytest.raises(typer.Exit):
                cli_main()
            
            mock_rprint.assert_called_once()
            args = mock_rprint.call_args[0][0]
            assert "Operation cancelled by user" in args
    
    @patch('research_agent_backend.cli.cli.app')
    def test_cli_main_generic_exception(self, mock_app):
        """Test CLI main with generic exception."""
        mock_app.side_effect = ValueError("Test error")
        
        with patch('research_agent_backend.cli.cli.handle_cli_error') as mock_handle:
            with pytest.raises(typer.Exit):
                cli_main()
            
            mock_handle.assert_called_once()
            assert isinstance(mock_handle.call_args[0][0], ValueError)


class TestCLICommands:
    """Test suite for built-in CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.mock_config_data = {
            "version": "0.1.0",
            "embedding_model": {"name": "test-model"},
            "vector_store": {"type": "chromadb"},
            "chunking_strategy": {"chunk_size": 512},
        }
    
    @patch('research_agent_backend.cli.cli.get_config_manager')
    def test_info_command_success(self, mock_get_config_manager):
        """Test the info command with successful configuration loading."""
        mock_config_manager = Mock()
        mock_config_manager.config = self.mock_config_data
        mock_config_manager.config_file = "test.json"
        mock_config_manager.project_root = Path("/test")
        mock_config_manager.is_loaded = True
        mock_get_config_manager.return_value = mock_config_manager
        
        result = self.runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "Research Agent Information" in result.stdout
        assert "0.1.0" in result.stdout
        assert "test-model" in result.stdout
    
    @patch('research_agent_backend.cli.cli.get_config_manager')
    def test_info_command_config_error(self, mock_get_config_manager):
        """Test the info command with configuration error."""
        mock_get_config_manager.side_effect = typer.Exit(1)
        
        result = self.runner.invoke(app, ["info"])
        
        assert result.exit_code == 1
    
    @patch('research_agent_backend.cli.cli.get_config_manager')
    def test_version_command_success(self, mock_get_config_manager):
        """Test the version command with successful configuration loading."""
        mock_config_manager = Mock()
        mock_config_manager.config = self.mock_config_data
        mock_get_config_manager.return_value = mock_config_manager
        
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "Research Agent" in result.stdout
        assert "0.1.0" in result.stdout
    
    @patch('research_agent_backend.cli.cli.get_config_manager')
    def test_version_command_fallback(self, mock_get_config_manager):
        """Test the version command with configuration error fallback."""
        mock_get_config_manager.side_effect = ConfigurationError("Test error")
        
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "Research Agent" in result.stdout
        assert "0.1.0" in result.stdout  # Fallback version


class TestCLIGlobalOptions:
    """Test suite for CLI global options."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_verbose_option(self):
        """Test the --verbose global option."""
        with patch('research_agent_backend.cli.cli.setup_logging') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            result = self.runner.invoke(app, ["--verbose", "version"])
            
            # Should call setup_logging with verbose=True
            mock_setup.assert_called_with(True)
    
    def test_config_path_option(self):
        """Test the --config-path global option."""
        # The config path is handled in the callback, not immediately during invocation
        result = self.runner.invoke(app, ["--config-path", "custom.json", "version"])
        
        # Should execute without error (basic smoke test)
        # Detailed config loading is tested elsewhere
        assert result.exit_code == 0
    
    def test_dry_run_option(self):
        """Test the --dry-run global option."""
        # Test with a command that checks dry-run (kb commands have this check)
        result = self.runner.invoke(app, ["--dry-run", "kb", "add-document", "test.md"])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
    
    def test_help_option(self):
        """Test the help option."""
        result = self.runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        # Check for key help text elements
        assert "AI-powered research agent" in result.stdout
        assert "Commands" in result.stdout
        assert "kb" in result.stdout
        assert "collections" in result.stdout
        assert "projects" in result.stdout
        assert "query" in result.stdout


class TestCLICommandGroups:
    """Test suite for CLI command group registration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_kb_command_group(self):
        """Test knowledge base command group."""
        result = self.runner.invoke(app, ["kb", "--help"])
        
        assert result.exit_code == 0
        assert "Knowledge base management commands" in result.stdout
        assert "add-document" in result.stdout
        assert "ingest-folder" in result.stdout
        assert "list-documents" in result.stdout
    
    def test_collections_command_group(self):
        """Test collections command group."""
        result = self.runner.invoke(app, ["collections", "--help"])
        
        assert result.exit_code == 0
        assert "Collection management commands" in result.stdout
        assert "create" in result.stdout
        assert "list" in result.stdout
        assert "delete" in result.stdout
    
    def test_projects_command_group(self):
        """Test projects command group."""
        result = self.runner.invoke(app, ["projects", "--help"])
        
        assert result.exit_code == 0
        assert "Project-specific operations" in result.stdout
        assert "init" in result.stdout
        assert "list" in result.stdout
        assert "activate" in result.stdout
    
    def test_query_command_group(self):
        """Test query command group."""
        result = self.runner.invoke(app, ["query", "--help"])
        
        assert result.exit_code == 0
        assert "RAG querying commands" in result.stdout
        assert "search" in result.stdout
        assert "ask" in result.stdout
        assert "interactive" in result.stdout
    
    def test_placeholder_commands_return_todo(self):
        """Test that placeholder commands return TODO messages."""
        # Test a sample command from each group
        commands_to_test = [
            ["kb", "add-document", "test.md"],
            ["collections", "create", "test-collection"],
            ["projects", "init", "test-project"],
            ["query", "search", "test query"],
        ]
        
        for cmd in commands_to_test:
            result = self.runner.invoke(app, cmd)
            assert result.exit_code == 0
            assert "TODO" in result.stdout
            assert "Not implemented yet" in result.stdout 