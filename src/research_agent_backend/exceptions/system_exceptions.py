"""
System-wide exception classes for Research Agent.

This module defines comprehensive error handling with severity levels,
context preservation, and recovery action suggestions.

Implements FR-EH-001 through FR-EH-005: Comprehensive error handling system.
Optimized for memory efficiency and performance.
"""

import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels with ordering support."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        """Enable ordering of severity levels."""
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        
        order = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        return order[self] < order[other]


class ErrorRecoveryAction(Enum):
    """Recovery action suggestions for different error types."""
    CHECK_CONFIGURATION = "Check configuration file for missing or invalid settings"
    RETRY_OPERATION = "Retry the operation after a brief delay"
    CHECK_PERMISSIONS = "Check file/directory permissions"
    CHECK_NETWORK = "Check network connectivity and endpoint availability"
    RESTART_SERVICE = "Restart the affected service or component"
    CHECK_DEPENDENCIES = "Verify all required dependencies are installed and accessible"
    CONTACT_SUPPORT = "Contact system administrator or support team"
    CHECK_DISK_SPACE = "Check available disk space"
    UPDATE_CREDENTIALS = "Update or refresh authentication credentials"
    SCALE_RESOURCES = "Consider scaling system resources (memory, CPU, storage)"
    
    def __str__(self):
        return self.value


class ErrorContext:
    """
    Error context information with selective metadata capture.
    Optimized for memory efficiency with lazy evaluation.
    """
    __slots__ = ('operation', 'user_id', 'session_id', 'request_id', '_additional_data', '_severity_threshold')
    
    def __init__(
        self,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        _additional_data: Optional[Dict[str, Any]] = None,
        _severity_threshold: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self._additional_data = _additional_data
        self._severity_threshold = _severity_threshold
        
        # Initialize with empty additional data to avoid mutable default
        if self._additional_data is None:
            self._additional_data = {}
    
    @property
    def additional_data(self) -> Dict[str, Any]:
        """Get additional data dictionary."""
        if self._additional_data is None:
            self._additional_data = {}
        return self._additional_data
    
    @additional_data.setter
    def additional_data(self, value: Dict[str, Any]):
        """Set additional data dictionary."""
        self._additional_data = value or {}
    
    def add_data(self, key: str, value: Any, min_severity: ErrorSeverity = ErrorSeverity.LOW) -> None:
        """
        Add data to context only if severity threshold is met.
        This reduces memory usage for low-severity errors.
        """
        if min_severity >= self._severity_threshold:
            if self._additional_data is None:
                self._additional_data = {}
            self._additional_data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation."""
        result = {
            "operation": self.operation,
        }
        
        # Only include non-None values to reduce memory footprint
        if self.user_id is not None:
            result["user_id"] = self.user_id
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self._additional_data:
            result["additional_data"] = self._additional_data
            
        return result


class ResearchAgentError(Exception):
    """
    Base exception class for Research Agent with optimized memory usage.
    
    Uses __slots__ for memory efficiency and lazy evaluation for expensive operations.
    """
    __slots__ = (
        '_message', '_severity', '_context', '_suggested_actions', 
        '_timestamp', '_error_id', '_original_exception', '_cached_dict'
    )
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        suggested_actions: Optional[List[Union[ErrorRecoveryAction, str]]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self._message = message
        self._severity = severity
        self._original_exception = original_exception
        self._cached_dict = None  # Cache for expensive operations
        
        # Convert context to ErrorContext if needed
        if isinstance(context, dict):
            self._context = ErrorContext(
                operation=context.get('operation', 'unknown'),
                user_id=context.get('user_id'),
                session_id=context.get('session_id'),
                request_id=context.get('request_id'),
                _additional_data=context.get('additional_data', {})
            )
        else:
            self._context = context or ErrorContext(operation="unknown")
        
        # Set severity threshold for context
        self._context._severity_threshold = severity
        
        # Process suggested actions with lazy evaluation
        self._suggested_actions = self._process_suggested_actions(suggested_actions)
        
        # Generate timestamp and ID only when needed for high-severity errors
        self._timestamp = None
        self._error_id = None
    
    @property
    def message(self) -> str:
        """Get error message."""
        return self._message
    
    @property
    def severity(self) -> ErrorSeverity:
        """Get error severity."""
        return self._severity
    
    @property
    def context(self) -> ErrorContext:
        """Get error context."""
        return self._context
    
    @context.setter
    def context(self, value: Optional[Union[ErrorContext, Dict[str, Any]]]):
        """Set error context."""
        if isinstance(value, dict):
            self._context = ErrorContext(
                operation=value.get('operation', 'unknown'),
                user_id=value.get('user_id'),
                session_id=value.get('session_id'),
                request_id=value.get('request_id'),
                _additional_data=value.get('additional_data', {})
            )
        elif isinstance(value, ErrorContext):
            self._context = value
        elif value is None:
            self._context = ErrorContext(operation="unknown")
        else:
            raise TypeError(f"Context must be ErrorContext or dict, got {type(value)}")
    
    @property
    def suggested_actions(self) -> List[ErrorRecoveryAction]:
        """Get suggested recovery actions."""
        return self._suggested_actions
    
    @property
    def original_exception(self) -> Optional[Exception]:
        """Get original exception if available."""
        return self._original_exception
    
    @property
    def timestamp(self) -> datetime:
        """Get error timestamp (lazy evaluation)."""
        if self._timestamp is None:
            self._timestamp = datetime.now()
        return self._timestamp
    
    @property
    def error_id(self) -> str:
        """Get unique error ID (lazy evaluation)."""
        if self._error_id is None:
            self._error_id = str(uuid.uuid4())
        return self._error_id
    
    def _process_suggested_actions(
        self, 
        actions: Optional[List[Union[ErrorRecoveryAction, str]]]
    ) -> List[ErrorRecoveryAction]:
        """Process and convert suggested actions."""
        if not actions:
            return []
        
        processed_actions = []
        for action in actions:
            if isinstance(action, ErrorRecoveryAction):
                processed_actions.append(action)
            elif isinstance(action, str):
                # Try to find matching enum value, otherwise create a generic one
                for recovery_action in ErrorRecoveryAction:
                    if action.lower() in recovery_action.value.lower():
                        processed_actions.append(recovery_action)
                        break
                else:
                    # If no match found, still store as string but convert when accessed
                    processed_actions.append(ErrorRecoveryAction.CONTACT_SUPPORT)
            
        return processed_actions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary representation with caching.
        Uses lazy evaluation to avoid expensive operations for low-severity errors.
        """
        if self._cached_dict is None or self._severity >= ErrorSeverity.HIGH:
            self._cached_dict = {
                "error_id": self.error_id,
                "message": self._message,
                "severity": self._severity.value,
                "timestamp": self.timestamp.isoformat(),
                "context": self._context.to_dict(),
                "suggested_actions": [str(action) for action in self._suggested_actions]
            }
            
            if self._original_exception:
                self._cached_dict["original_exception"] = str(self._original_exception)
        
        return self._cached_dict
    
    def __str__(self) -> str:
        """String representation of the error."""
        return self._message


class ConfigurationSystemError(ResearchAgentError):
    """Configuration-related system error with optimized memory usage."""
    __slots__ = ('_config_file', '_validation_errors')
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        **kwargs
    ):
        # Default suggested actions for configuration errors
        default_actions = [
            ErrorRecoveryAction.CHECK_CONFIGURATION,
            ErrorRecoveryAction.CHECK_PERMISSIONS,
            ErrorRecoveryAction.CONTACT_SUPPORT
        ]
        
        suggested_actions = kwargs.pop('suggested_actions', default_actions)
        super().__init__(message, severity, suggested_actions=suggested_actions, **kwargs)
        
        self._config_file = config_file
        self._validation_errors = validation_errors or []
    
    @property
    def config_file(self) -> Optional[str]:
        """Get configuration file path."""
        return self._config_file
    
    @property
    def validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors


class DatabaseSystemError(ResearchAgentError):
    """Database-related system error with optimized memory usage."""
    __slots__ = ('_database_type', '_connection_string', '_operation')
    
    def __init__(
        self,
        message: str,
        database_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        operation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        **kwargs
    ):
        # Default suggested actions for database errors
        default_actions = [
            ErrorRecoveryAction.RETRY_OPERATION,
            ErrorRecoveryAction.CHECK_NETWORK,
            ErrorRecoveryAction.RESTART_SERVICE,
            ErrorRecoveryAction.CHECK_DEPENDENCIES
        ]
        
        suggested_actions = kwargs.pop('suggested_actions', default_actions)
        super().__init__(message, severity, suggested_actions=suggested_actions, **kwargs)
        
        self._database_type = database_type
        self._connection_string = connection_string
        self._operation = operation
    
    @property
    def database_type(self) -> Optional[str]:
        """Get database type."""
        return self._database_type
    
    @property
    def connection_string(self) -> Optional[str]:
        """Get connection string."""
        return self._connection_string
    
    @property
    def operation(self) -> Optional[str]:
        """Get database operation."""
        return self._operation


class ModelSystemError(ResearchAgentError):
    """AI model-related system error with optimized memory usage."""
    __slots__ = ('_model_name', '_model_type', '_error_type')
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        error_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        **kwargs
    ):
        # Default suggested actions for model errors
        default_actions = [
            ErrorRecoveryAction.CHECK_DEPENDENCIES,
            ErrorRecoveryAction.RETRY_OPERATION,
            ErrorRecoveryAction.SCALE_RESOURCES,
            ErrorRecoveryAction.CONTACT_SUPPORT
        ]
        
        suggested_actions = kwargs.pop('suggested_actions', default_actions)
        super().__init__(message, severity, suggested_actions=suggested_actions, **kwargs)
        
        self._model_name = model_name
        self._model_type = model_type
        self._error_type = error_type
    
    @property
    def model_name(self) -> Optional[str]:
        """Get model name."""
        return self._model_name
    
    @property
    def model_type(self) -> Optional[str]:
        """Get model type."""
        return self._model_type
    
    @property
    def error_type(self) -> Optional[str]:
        """Get error type."""
        return self._error_type


class FileSystemError(ResearchAgentError):
    """File system-related error with optimized memory usage."""
    __slots__ = ('_file_path', '_operation', '_permissions_required')
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        permissions_required: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs
    ):
        # Default suggested actions for file system errors
        default_actions = [
            ErrorRecoveryAction.CHECK_PERMISSIONS,
            ErrorRecoveryAction.CHECK_DISK_SPACE,
            ErrorRecoveryAction.RETRY_OPERATION
        ]
        
        suggested_actions = kwargs.pop('suggested_actions', default_actions)
        super().__init__(message, severity, suggested_actions=suggested_actions, **kwargs)
        
        self._file_path = file_path
        self._operation = operation
        self._permissions_required = permissions_required
    
    @property
    def file_path(self) -> Optional[str]:
        """Get file path."""
        return self._file_path
    
    @property
    def operation(self) -> Optional[str]:
        """Get file operation."""
        return self._operation
    
    @property
    def permissions_required(self) -> Optional[str]:
        """Get required permissions."""
        return self._permissions_required


class NetworkSystemError(ResearchAgentError):
    """Network-related system error with optimized memory usage."""
    __slots__ = ('_endpoint', '_status_code', '_timeout_duration', '_retry_count')
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        timeout_duration: Optional[float] = None,
        retry_count: Optional[int] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs
    ):
        # Default suggested actions for network errors
        default_actions = [
            ErrorRecoveryAction.CHECK_NETWORK,
            ErrorRecoveryAction.RETRY_OPERATION,
            ErrorRecoveryAction.UPDATE_CREDENTIALS,
            ErrorRecoveryAction.CONTACT_SUPPORT
        ]
        
        suggested_actions = kwargs.pop('suggested_actions', default_actions)
        super().__init__(message, severity, suggested_actions=suggested_actions, **kwargs)
        
        self._endpoint = endpoint
        self._status_code = status_code
        self._timeout_duration = timeout_duration
        self._retry_count = retry_count
    
    @property
    def endpoint(self) -> Optional[str]:
        """Get endpoint URL."""
        return self._endpoint
    
    @property
    def status_code(self) -> Optional[int]:
        """Get HTTP status code."""
        return self._status_code
    
    @property
    def timeout_duration(self) -> Optional[float]:
        """Get timeout duration."""
        return self._timeout_duration
    
    @property
    def retry_count(self) -> Optional[int]:
        """Get retry count."""
        return self._retry_count 