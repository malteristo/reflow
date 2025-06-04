#!/usr/bin/env python3
"""
Error Handling and Logging System Verification Script.

This script tests the comprehensive error handling and logging infrastructure
to validate it meets production requirements.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_agent_backend.exceptions import (
    ResearchAgentError,
    ConfigurationSystemError,
    DatabaseSystemError,
    ModelSystemError,
    FileSystemError,
    NetworkSystemError,
    ErrorSeverity,
    ErrorContext,
    ErrorRecoveryAction,
)
from research_agent_backend.utils.logging_config import LoggingManager, LogFormat
from research_agent_backend.utils.error_handler import ErrorHandler


def test_exception_hierarchy() -> List[str]:
    """Test the exception hierarchy and error messages."""
    print("\n🔍 Testing Exception Hierarchy...")
    
    results = []
    
    # Test base ResearchAgentError
    try:
        context = ErrorContext(
            operation="test_operation",
            user_id="test_user",
            session_id="test_session"
        )
        
        error = ResearchAgentError(
            message="Test error message",
            severity=ErrorSeverity.HIGH,
            context=context,
            suggested_actions=[ErrorRecoveryAction.CHECK_CONFIGURATION]
        )
        
        # Verify properties
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "test_operation"
        assert len(error.suggested_actions) == 1
        
        results.append("✅ ResearchAgentError base class works correctly")
        
    except Exception as e:
        results.append(f"❌ ResearchAgentError test failed: {e}")
    
    # Test specialized exceptions
    specialized_tests = [
        (ConfigurationSystemError, "config.json", ["validation error"]),
        (DatabaseSystemError, "chromadb", "insert_document"),
        (ModelSystemError, "sentence-transformers", "embedding"),
        (FileSystemError, "/path/to/file", "read"),
        (NetworkSystemError, "http://localhost:8000", 404)
    ]
    
    for error_class, *args in specialized_tests:
        try:
            if error_class == ConfigurationSystemError:
                error = error_class(
                    "Configuration error",
                    config_file=args[0],
                    validation_errors=args[1]
                )
            elif error_class == DatabaseSystemError:
                error = error_class(
                    "Database error",
                    database_type=args[0],
                    operation=args[1]
                )
            elif error_class == ModelSystemError:
                error = error_class(
                    "Model error",
                    model_name=args[0],
                    model_type=args[1]
                )
            elif error_class == FileSystemError:
                error = error_class(
                    "File system error",
                    file_path=args[0],
                    operation=args[1]
                )
            elif error_class == NetworkSystemError:
                error = error_class(
                    "Network error",
                    endpoint=args[0],
                    status_code=args[1]
                )
            
            # Verify error is properly constructed
            assert isinstance(error, ResearchAgentError)
            assert error.message
            assert error.severity
            
            results.append(f"✅ {error_class.__name__} works correctly")
            
        except Exception as e:
            results.append(f"❌ {error_class.__name__} test failed: {e}")
    
    return results


def test_logging_system() -> List[str]:
    """Test the logging system implementation."""
    print("\n📝 Testing Logging System...")
    
    results = []
    
    # Test LoggingManager
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Initialize logging manager
            manager = LoggingManager(
                log_file=log_file,
                log_format=LogFormat.JSON,
                enable_rotation=True
            )
            
            # Get a logger and test it
            logger = manager.get_logger("test_logger")
            
            # Test different log levels
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            # Verify log file was created
            if log_file.exists():
                results.append("✅ LoggingManager creates log files correctly")
                
                # Check log content
                content = log_file.read_text()
                if "Test info message" in content and "Test warning message" in content:
                    results.append("✅ Log messages are written correctly")
                else:
                    results.append("❌ Log messages not found in file")
            else:
                results.append("❌ Log file was not created")
                
    except Exception as e:
        results.append(f"❌ LoggingManager test failed: {e}")
    
    return results


def test_error_handler() -> List[str]:
    """Test the error handler system."""
    print("\n🔧 Testing Error Handler...")
    
    results = []
    
    try:
        # Initialize error handler
        handler = ErrorHandler()
        
        # Test error handling
        test_error = ConfigurationSystemError(
            "Test configuration error",
            config_file="test.json"
        )
        
        handled_error = handler.handle_error(test_error)
        
        # Verify error was handled
        assert isinstance(handled_error, ResearchAgentError)
        assert handled_error.message == "Test configuration error"
        
        results.append("✅ ErrorHandler processes errors correctly")
        
    except Exception as e:
        results.append(f"❌ ErrorHandler test failed: {e}")
    
    return results


def test_cli_error_integration() -> List[str]:
    """Test CLI error handling integration."""
    print("\n💻 Testing CLI Error Integration...")
    
    results = []
    
    try:
        # Import CLI error handling functions
        from research_agent_backend.cli.cli import handle_cli_error
        
        # Test different error types with CLI handler
        test_errors = [
            ConfigurationSystemError("Test config error"),
            FileNotFoundError("Test file not found"),
            PermissionError("Test permission denied"),
            ValueError("Test generic error")
        ]
        
        for error in test_errors:
            try:
                # This would normally print to console and exit
                # We'll catch the typer.Exit exception
                handle_cli_error(error)
            except SystemExit:
                # Expected behavior - CLI handler exits after displaying error
                pass
            except Exception as e:
                results.append(f"❌ CLI error handling failed for {type(error).__name__}: {e}")
                continue
        
        results.append("✅ CLI error handling functions work correctly")
        
    except ImportError:
        results.append("❌ Cannot import CLI error handling functions")
    except Exception as e:
        results.append(f"❌ CLI error integration test failed: {e}")
    
    return results


def test_error_context_and_recovery() -> List[str]:
    """Test error context and recovery actions."""
    print("\n🔄 Testing Error Context and Recovery...")
    
    results = []
    
    try:
        # Test ErrorContext with LOW threshold to allow data addition
        context = ErrorContext(
            operation="document_processing",
            user_id="user123",
            session_id="session456",
            _severity_threshold=ErrorSeverity.LOW  # Allow all data to be added
        )
        
        context.add_data("file_path", "/test/path.md")
        context.add_data("chunk_count", 5)
        
        # Verify context works
        context_dict = context.to_dict()
        assert "document_processing" in str(context_dict)
        
        # Check additional_data exists and has expected content
        additional_data = context_dict.get("additional_data", {})
        if "file_path" not in additional_data:
            results.append(f"❌ Error context test failed: file_path not found in additional_data: {additional_data}")
            return results
        
        results.append("✅ ErrorContext works correctly")
        
    except Exception as e:
        results.append(f"❌ ErrorContext test failed: {e}")
        return results
    
    try:
        # Test error with recovery actions
        error = ResearchAgentError(
            message="Test error with recovery",
            context=context,
            suggested_actions=[
                ErrorRecoveryAction.CHECK_CONFIGURATION,
                ErrorRecoveryAction.RETRY_OPERATION
            ]
        )
        
        # Verify recovery actions
        if len(error.suggested_actions) != 2:
            results.append(f"❌ Recovery actions test failed: expected 2 actions, got {len(error.suggested_actions)}")
            return results
        
        if ErrorRecoveryAction.CHECK_CONFIGURATION not in error.suggested_actions:
            results.append(f"❌ Recovery actions test failed: CHECK_CONFIGURATION not found in {error.suggested_actions}")
            return results
        
        results.append("✅ Recovery actions work correctly")
        
    except Exception as e:
        results.append(f"❌ Recovery actions test failed: {e}")
    
    return results


def main():
    """Run all verification tests."""
    print("🚀 Starting Error Handling and Logging System Verification")
    print("=" * 60)
    
    all_results = []
    
    # Run all tests
    test_functions = [
        test_exception_hierarchy,
        test_logging_system,
        test_error_handler,
        test_cli_error_integration,
        test_error_context_and_recovery,
    ]
    
    for test_func in test_functions:
        try:
            results = test_func()
            all_results.extend(results)
        except Exception as e:
            all_results.append(f"❌ {test_func.__name__} failed completely: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for result in all_results:
        print(result)
        if result.startswith("✅"):
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All error handling and logging system tests PASSED!")
        print("💡 The system is ready for production use.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the system before production use.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 