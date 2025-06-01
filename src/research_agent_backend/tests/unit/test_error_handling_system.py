"""
Test suite for comprehensive error handling and logging system.

This module tests:
- Custom exception classes for different error types  
- Comprehensive logging with configurable levels
- User-friendly error messages with suggested actions
- Error recovery mechanisms

Following TDD approach - these tests will initially fail.
"""

import pytest
import logging
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock

# Import the modules we'll be testing (these don't exist yet, so tests will fail)
from research_agent_backend.exceptions.system_exceptions import (
    ResearchAgentError,
    ConfigurationSystemError,
    DatabaseSystemError,
    ModelSystemError,
    FileSystemError,
    NetworkSystemError,
    ErrorSeverity,
    ErrorContext,
    ErrorRecoveryAction
)

from research_agent_backend.utils.logging_config import (
    LoggingManager,
    StructuredLogger,
    LogLevel,
    LogFormat,
    PerformanceLogger,
    AuditLogger
)

from research_agent_backend.utils.error_handler import (
    ErrorHandler,
    ErrorRecoveryManager,
    UserFriendlyErrorFormatter
)


class TestSystemExceptions:
    """Test custom exception classes for different error types."""
    
    def test_research_agent_error_base_class(self):
        """Test base ResearchAgentError class functionality."""
        # Base error should support severity levels
        error = ResearchAgentError(
            message="Test error",
            severity=ErrorSeverity.HIGH,
            context={"operation": "test"},
            suggested_actions=[ErrorRecoveryAction.RETRY_OPERATION, ErrorRecoveryAction.CHECK_CONFIGURATION]
        )
        
        assert str(error) == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "test"
        assert len(error.suggested_actions) == 2
        assert error.suggested_actions[0] == ErrorRecoveryAction.RETRY_OPERATION
        assert error.suggested_actions[1] == ErrorRecoveryAction.CHECK_CONFIGURATION
        assert error.timestamp is not None
        assert error.error_id is not None
    
    def test_configuration_system_error(self):
        """Test configuration-specific error handling."""
        error = ConfigurationSystemError(
            message="Invalid configuration file",
            config_file="/path/to/config.json",
            validation_errors=["Missing required field: api_key"],
            severity=ErrorSeverity.CRITICAL
        )
        
        assert isinstance(error, ResearchAgentError)
        assert error.config_file == "/path/to/config.json"
        assert error.validation_errors == ["Missing required field: api_key"]
        assert error.severity == ErrorSeverity.CRITICAL
        
        # Should suggest recovery actions
        action_strings = [str(action) for action in error.suggested_actions]
        assert any("Check configuration file" in action for action in action_strings)
    
    def test_database_system_error(self):
        """Test database-specific error handling."""
        error = DatabaseSystemError(
            message="Connection failed",
            database_type="chromadb",
            connection_string="memory://test",
            operation="connect",
            severity=ErrorSeverity.HIGH
        )
        
        assert isinstance(error, ResearchAgentError)
        assert error.database_type == "chromadb"
        assert error.connection_string == "memory://test"
        assert error.operation == "connect"
        
        # Should include recovery suggestions - check for the actual default actions
        action_strings = [str(action) for action in error.suggested_actions]
        assert any("retry" in action.lower() or "network" in action.lower() or "service" in action.lower() 
                  for action in action_strings)
    
    def test_model_system_error(self):
        """Test model-specific error handling."""
        error = ModelSystemError(
            message="Model loading failed",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            error_type="loading_failure",
            severity=ErrorSeverity.HIGH
        )
        
        assert isinstance(error, ResearchAgentError)
        assert error.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert error.model_type == "embedding"
        assert error.error_type == "loading_failure"
        
        # Should suggest model-specific recovery - check for actual default actions
        action_strings = [str(action) for action in error.suggested_actions]
        assert any("dependencies" in action.lower() or "retry" in action.lower() or "resources" in action.lower() 
                  for action in action_strings)
    
    def test_file_system_error(self):
        """Test file system error handling."""
        error = FileSystemError(
            message="Permission denied",
            file_path="/protected/file.txt",
            operation="read",
            permissions_required="r",
            severity=ErrorSeverity.MEDIUM
        )
        
        assert isinstance(error, ResearchAgentError)
        assert error.file_path == "/protected/file.txt"
        assert error.operation == "read"
        assert error.permissions_required == "r"
        
        # Should suggest permission fixes
        action_strings = [str(action) for action in error.suggested_actions]
        assert any("permission" in action.lower() for action in action_strings)
    
    def test_network_system_error(self):
        """Test network-specific error handling."""
        error = NetworkSystemError(
            message="API timeout",
            endpoint="https://api.example.com/embeddings",
            status_code=408,
            timeout_duration=30.0,
            retry_count=3,
            severity=ErrorSeverity.MEDIUM
        )
        
        assert isinstance(error, ResearchAgentError)
        assert error.endpoint == "https://api.example.com/embeddings"
        assert error.status_code == 408
        assert error.timeout_duration == 30.0
        assert error.retry_count == 3
        
        # Should suggest retry/network troubleshooting
        action_strings = [str(action) for action in error.suggested_actions]
        assert any("retry" in action.lower() or "network" in action.lower() 
                  for action in action_strings)
    
    def test_error_severity_enum(self):
        """Test error severity levels."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
        
        # Test ordering
        assert ErrorSeverity.LOW < ErrorSeverity.MEDIUM
        assert ErrorSeverity.MEDIUM < ErrorSeverity.HIGH
        assert ErrorSeverity.HIGH < ErrorSeverity.CRITICAL
    
    def test_error_context_handling(self):
        """Test error context management."""
        context = ErrorContext(
            operation="document_ingestion",
            user_id="user123",
            session_id="session456",
            request_id="req789",
            _additional_data={"document_id": "doc001", "collection": "test"}
        )
        
        assert context.operation == "document_ingestion"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.request_id == "req789"
        assert context.additional_data["document_id"] == "doc001"
        
        # Should be serializable
        context_dict = context.to_dict()
        assert context_dict["operation"] == "document_ingestion"
        assert context_dict["additional_data"]["collection"] == "test"


class TestLoggingManager:
    """Test comprehensive logging management system."""
    
    def test_logging_manager_initialization(self):
        """Test LoggingManager creation and configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            manager = LoggingManager(
                log_level=LogLevel.DEBUG,
                log_format=LogFormat.JSON,
                log_file=log_file,
                enable_console=True,
                enable_rotation=True,
                max_file_size="10MB",
                backup_count=5
            )
            
            assert manager.log_level == LogLevel.DEBUG
            assert manager.log_format == LogFormat.JSON
            assert manager.log_file == log_file
            assert manager.enable_console is True
            assert manager.enable_rotation is True
            assert manager.max_file_size == "10MB"
            assert manager.backup_count == 5
    
    def test_structured_logger_json_output(self):
        """Test structured logging with JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "structured.log"
            
            logger = StructuredLogger(
                name="test_logger",
                log_file=log_file,
                format_type=LogFormat.JSON
            )
            
            # Log structured data
            logger.info(
                "Document processed",
                document_id="doc123",
                processing_time=1.5,
                chunk_count=10,
                metadata={"author": "test", "size": 1024}
            )
            
            # Read and verify JSON log entry
            assert log_file.exists()
            with open(log_file) as f:
                log_line = f.readline().strip()
                log_data = json.loads(log_line)
                
                assert log_data["message"] == "Document processed"
                assert log_data["document_id"] == "doc123"
                assert log_data["processing_time"] == 1.5
                assert log_data["chunk_count"] == 10
                assert log_data["metadata"]["author"] == "test"
                assert "timestamp" in log_data
                assert "level" in log_data
    
    def test_performance_logger(self):
        """Test performance metrics logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "performance.log"
            
            perf_logger = PerformanceLogger(log_file=log_file)
            
            # Test operation timing
            with perf_logger.time_operation("embedding_generation", force_log=True):
                import time
                time.sleep(0.1)  # Simulate work
            
            # Test metric logging
            perf_logger.log_metric("memory_usage", 512.5, unit="MB")
            perf_logger.log_metric("documents_processed", 100, unit="count")
            
            # Force flush to ensure all buffered entries are written
            perf_logger.force_flush()
            
            # Verify performance log entries
            assert log_file.exists()
            with open(log_file) as f:
                lines = f.readlines()
                assert len(lines) >= 3  # At least 3 log entries
                
                # Check that we have log entries for our operations
                content = ''.join(lines)
                assert "embedding_generation" in content
                assert "memory_usage" in content
                assert "512.5" in content
                assert "MB" in content
    
    def test_audit_logger(self):
        """Test user action audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "audit.log"
            
            audit_logger = AuditLogger(log_file=log_file)
            
            # Log user actions
            audit_logger.log_action(
                user_id="user123",
                action="document_upload",
                resource="document_456",
                details={"file_size": 1024, "collection": "research"},
                success=True
            )
            
            audit_logger.log_action(
                user_id="user456",
                action="query_execution",
                resource="collection_789",
                details={"query": "machine learning", "results": 15},
                success=True
            )
            
            # Verify audit log entries
            assert log_file.exists()
            with open(log_file) as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                # Parse first audit entry
                audit_data = json.loads(lines[0])
                assert audit_data["user_id"] == "user123"
                assert audit_data["action"] == "document_upload"
                assert audit_data["resource"] == "document_456"
                assert audit_data["success"] is True
                assert audit_data["details"]["file_size"] == 1024
                assert "timestamp" in audit_data
    
    def test_log_rotation(self):
        """Test log file rotation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "rotation.log"
            
            manager = LoggingManager(
                log_file=log_file,
                enable_rotation=True,
                max_file_size="1KB",  # Very small for testing
                backup_count=3
            )
            
            logger = manager.get_logger("test")
            
            # Generate enough logs to trigger rotation
            for i in range(100):
                logger.info(f"Test log message {i} with sufficient content to reach size limit")
            
            # Check for rotated files
            log_files = list(Path(temp_dir).glob("rotation.log*"))
            assert len(log_files) > 1  # Should have main + rotated files
    
    def test_log_level_filtering(self):
        """Test log level filtering functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "filtered.log"
            
            # Create logger with INFO level
            manager = LoggingManager(
                log_level=LogLevel.INFO,
                log_file=log_file
            )
            logger = manager.get_logger("test")
            
            # Log at different levels
            logger.debug("Debug message")  # Should be filtered out
            logger.info("Info message")    # Should be logged
            logger.warning("Warning message")  # Should be logged
            logger.error("Error message")  # Should be logged
            
            # Check log file contents
            with open(log_file) as f:
                content = f.read()
                assert "Debug message" not in content
                assert "Info message" in content
                assert "Warning message" in content
                assert "Error message" in content


class TestErrorHandler:
    """Test error handling and recovery system."""
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler setup and configuration."""
        handler = ErrorHandler(
            enable_recovery=True,
            max_recovery_attempts=3,
            recovery_timeout=30.0
        )
        
        assert handler.enable_recovery is True
        assert handler.max_recovery_attempts == 3
        assert handler.recovery_timeout == 30.0
    
    def test_error_handling_with_context(self):
        """Test error handling with context preservation."""
        handler = ErrorHandler()
        
        context = ErrorContext(
            operation="test_operation",
            user_id="user123"
        )
        
        # Handle an error with context
        try:
            raise ValueError("Test error")
        except ValueError as e:
            handled_error = handler.handle_error(e, context=context)
            
            assert isinstance(handled_error, ResearchAgentError)
            assert handled_error.context.operation == "test_operation"
            assert handled_error.context.user_id == "user123"
            assert handled_error.original_exception == e
    
    def test_user_friendly_error_formatter(self):
        """Test user-friendly error message formatting."""
        formatter = UserFriendlyErrorFormatter()
        
        # Test configuration error formatting
        config_error = ConfigurationSystemError(
            message="Missing API key",
            config_file="config.json",
            validation_errors=["api_key is required"]
        )
        
        formatted = formatter.format_error(config_error)
        
        assert "Configuration Problem" in formatted
        assert "Missing API key" in formatted
        assert "config.json" in formatted
        assert len(formatted.split('\n')) > 1  # Multi-line format
        
        # Should include suggested actions
        assert "Try these solutions:" in formatted or "Suggested actions:" in formatted
    
    def test_error_recovery_manager(self):
        """Test automatic error recovery mechanisms."""
        recovery_manager = ErrorRecoveryManager()
        
        # Test network error recovery
        network_error = NetworkSystemError(
            message="Connection timeout",
            endpoint="https://api.example.com",
            timeout_duration=30.0
        )
        
        # Mock a recovery action
        def mock_retry_connection():
            return True
        
        recovery_manager.register_recovery_action(
            error_type=NetworkSystemError,
            action=mock_retry_connection
        )
        
        # Test recovery execution
        recovery_result = recovery_manager.attempt_recovery(network_error)
        assert recovery_result.success is True
        assert recovery_result.attempts == 1
    
    def test_error_recovery_with_retries(self):
        """Test error recovery with retry logic."""
        recovery_manager = ErrorRecoveryManager(max_attempts=3)
        
        # Create a recovery action that fails twice then succeeds
        attempt_count = 0
        def flaky_recovery_action():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Recovery failed")
            return True
        
        network_error = NetworkSystemError(
            message="Temporary failure",
            endpoint="https://api.example.com"
        )
        
        recovery_manager.register_recovery_action(
            error_type=NetworkSystemError,
            action=flaky_recovery_action
        )
        
        # Test recovery with retries
        recovery_result = recovery_manager.attempt_recovery(network_error)
        assert recovery_result.success is True
        assert recovery_result.attempts == 3
    
    def test_error_aggregation_and_reporting(self):
        """Test error aggregation and reporting functionality."""
        handler = ErrorHandler()
        
        # Generate multiple errors
        errors = [
            ConfigurationSystemError("Config error 1"),
            ConfigurationSystemError("Config error 2"),
            DatabaseSystemError("DB error 1"),
            NetworkSystemError("Network error 1"),
        ]
        
        for error in errors:
            handler.handle_error(error)
        
        # Test error aggregation
        error_summary = handler.get_error_summary()
        
        assert error_summary.total_errors == 4
        assert error_summary.error_counts[ConfigurationSystemError.__name__] == 2
        assert error_summary.error_counts[DatabaseSystemError.__name__] == 1
        assert error_summary.error_counts[NetworkSystemError.__name__] == 1
        
        # Test recent errors - deque keeps insertion order, most recent is last
        recent_errors = handler.get_recent_errors(limit=2)
        assert len(recent_errors) == 2
        assert isinstance(recent_errors[-1], NetworkSystemError)  # Most recent (last added)
        assert isinstance(recent_errors[-2], DatabaseSystemError)  # Second most recent
    
    def test_error_notification_system(self):
        """Test error notification and alerting."""
        handler = ErrorHandler(enable_notifications=True)
        
        # Mock notification handler
        notifications_sent = []
        def mock_notification_handler(error, severity):
            notifications_sent.append((error, severity))
        
        handler.register_notification_handler(mock_notification_handler)
        
        # Test critical error notification
        critical_error = DatabaseSystemError(
            message="Database corruption detected",
            severity=ErrorSeverity.CRITICAL
        )
        
        handler.handle_error(critical_error)
        
        # Verify notification was sent
        assert len(notifications_sent) == 1
        assert notifications_sent[0][0] == critical_error
        assert notifications_sent[0][1] == ErrorSeverity.CRITICAL


class TestIntegratedErrorHandlingAndLogging:
    """Test integration between error handling and logging systems."""
    
    def test_error_logging_integration(self):
        """Test that errors are properly logged when handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "integrated.log"
            
            # Setup integrated logging and error handling
            logging_manager = LoggingManager(
                log_file=log_file,
                log_format=LogFormat.JSON
            )
            error_handler = ErrorHandler(logger=logging_manager.get_logger("errors"))
            
            # Handle an error
            test_error = ConfigurationSystemError(
                message="Test configuration error",
                config_file="test.json"
            )
            
            handled_error = error_handler.handle_error(test_error)
            
            # Verify error was logged
            assert log_file.exists()
            with open(log_file) as f:
                log_content = f.read()
                assert "Test configuration error" in log_content
                assert "test.json" in log_content
                assert handled_error.error_id in log_content
    
    def test_performance_impact_monitoring(self):
        """Test performance impact monitoring during error handling."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            # Setup performance monitoring logger directly
            performance_logger = PerformanceLogger(log_file=f.name)
            
            # Create error handler with performance monitoring
            handler = ErrorHandler(performance_logger=performance_logger)
            
            # Process multiple errors
            for i in range(10):
                error = DatabaseSystemError(f"Test error {i}")
                handler.handle_error(error)
            
            # Force flush to ensure all metrics are written
            performance_logger.force_flush()
            
            # Read performance logs
            f.seek(0)
            content = f.read()
            
            # Verify performance monitoring data is logged
            assert "error_handling" in content
            assert "value" in content  # Changed from "duration" to "value"
            assert "seconds" in content
            assert "DatabaseSystemError" in content
            assert "high" in content
            
            # Clean up
            os.unlink(f.name)
    
    def test_comprehensive_system_health_monitoring(self):
        """Test comprehensive system health monitoring through errors and logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup comprehensive monitoring
            error_log = Path(temp_dir) / "errors.log"
            audit_log = Path(temp_dir) / "audit.log"
            perf_log = Path(temp_dir) / "performance.log"
            
            logging_manager = LoggingManager(log_file=error_log)
            audit_logger = AuditLogger(log_file=audit_log)
            perf_logger = PerformanceLogger(log_file=perf_log)
            
            error_handler = ErrorHandler(
                logger=logging_manager.get_logger("system"),
                audit_logger=audit_logger,
                performance_logger=perf_logger
            )
            
            # Simulate system operations with errors
            context = ErrorContext(
                operation="document_processing",
                user_id="user123"
            )
            
            # Process successful operation
            audit_logger.log_action(
                user_id="user123",
                action="document_upload",
                resource="doc123",
                success=True
            )
            
            # Process failed operation - create error with context instead of setting it
            processing_error = FileSystemError(
                message="File not found",
                file_path="/missing/file.txt",
                operation="read",
                context=context  # Pass context during creation
            )
            
            error_handler.handle_error(processing_error)
            
            # Force flush performance metrics
            perf_logger.force_flush()
            
            # Verify all logs were created
            assert error_log.exists()
            assert audit_log.exists()
            assert perf_log.exists()
            
            # Verify cross-referencing capability
            with open(audit_log) as f:
                audit_content = f.read()
                assert "user123" in audit_content
            
            with open(error_log) as f:
                error_content = f.read()
                assert "File not found" in error_content
                assert "user123" in error_content  # Context preserved 