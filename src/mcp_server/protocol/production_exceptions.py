"""
Production MCP Exception Hierarchy for Research Agent.

Comprehensive exception classes for typed error handling, context preservation,
and production-ready error management.

Implements subtask 15.6: Develop Production-Ready Error Handling.
"""

import logging
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for production error classification."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    SECURITY = "security"
    COMMUNICATION = "communication"
    TOOL_EXECUTION = "tool_execution"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    DEPENDENCY = "dependency"
    BUSINESS_LOGIC = "business_logic"


@dataclass
class ErrorContext:
    """Context information for error tracking and debugging."""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


class MCPException(Exception):
    """
    Base exception class for all MCP server operations.
    
    Provides comprehensive error information including context, correlation IDs,
    severity levels, and production-ready error handling capabilities.
    """
    
    def __init__(
        self,
        message: str,
        error_code: int = -32603,  # Internal error by default
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        correlation_id: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        guidance: Optional[str] = None,
        caused_by: Optional[Exception] = None
    ):
        """
        Initialize MCP exception with comprehensive error information.
        
        Args:
            message: Human-readable error message
            error_code: MCP protocol error code
            severity: Error severity level
            category: Error category for classification
            correlation_id: Unique ID for request tracing
            context: Error context information
            recoverable: Whether the error is recoverable
            retry_after: Seconds to wait before retry (if applicable)
            guidance: Helpful guidance for error resolution
            caused_by: Underlying exception that caused this error
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.guidance = guidance
        self.caused_by = caused_by
        self.timestamp = datetime.utcnow()
        self.stack_trace = traceback.format_exc()
        
        # Log the exception creation
        self._log_exception()
    
    def _log_exception(self) -> None:
        """Log the exception with appropriate severity level."""
        log_data = {
            "correlation_id": self.correlation_id,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context.__dict__ if self.context else {},
            "caused_by": str(self.caused_by) if self.caused_by else None
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(f"ERROR: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(f"WARNING: {self.message}", extra=log_data)
        else:
            logger.info(f"INFO: {self.message}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "correlation_id": self.correlation_id,
            "context": self.context.__dict__ if self.context else {},
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "guidance": self.guidance,
            "timestamp": self.timestamp.isoformat(),
            "caused_by": str(self.caused_by) if self.caused_by else None
        }
    
    def get_mcp_error_response(self) -> Dict[str, Any]:
        """Get MCP protocol-compliant error response."""
        error_data = {
            "correlation_id": self.correlation_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.retry_after:
            error_data["retry_after"] = self.retry_after
        
        if self.guidance:
            error_data["guidance"] = self.guidance
        
        if self.context and self.context.operation:
            error_data["operation"] = self.context.operation
        
        return {
            "code": self.error_code,
            "message": self.message,
            "data": error_data
        }


class ValidationError(MCPException):
    """Exception for parameter validation failures."""
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Any = None,
        expected_type: Optional[str] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="parameter_validation",
            additional_context={
                "parameter": parameter,
                "value": str(value) if value is not None else None,
                "expected_type": expected_type,
                "validation_rule": validation_rule
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32602,  # Invalid params
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.VALIDATION,
            context=context,
            recoverable=True,
            guidance="Check parameter values and types according to the API specification",
            **kwargs
        )
        
        self.parameter = parameter
        self.value = value
        self.expected_type = expected_type
        self.validation_rule = validation_rule


class SecurityError(MCPException):
    """Exception for security-related errors."""
    
    def __init__(
        self,
        message: str,
        security_event: Optional[str] = None,
        threat_level: str = "medium",
        **kwargs
    ):
        context = ErrorContext(
            operation="security_validation",
            additional_context={
                "security_event": security_event,
                "threat_level": threat_level
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32001,  # Configuration error (security config)
            severity=ErrorSeverity.CRITICAL if threat_level == "high" else ErrorSeverity.ERROR,
            category=ErrorCategory.SECURITY,
            context=context,
            recoverable=False,
            guidance="Review security settings and request parameters for potential threats",
            **kwargs
        )
        
        self.security_event = security_event
        self.threat_level = threat_level


class CommunicationError(MCPException):
    """Exception for STDIO and communication failures."""
    
    def __init__(
        self,
        message: str,
        communication_type: str = "stdio",
        transport_error: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="communication",
            additional_context={
                "communication_type": communication_type,
                "transport_error": transport_error
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32700,  # Parse error
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.COMMUNICATION,
            context=context,
            recoverable=True,
            retry_after=5,
            guidance="Check network connection and message format",
            **kwargs
        )
        
        self.communication_type = communication_type
        self.transport_error = transport_error


class ToolExecutionError(MCPException):
    """Exception for MCP tool execution failures."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        cli_command: Optional[str] = None,
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="tool_execution",
            tool_name=tool_name,
            additional_context={
                "cli_command": cli_command,
                "exit_code": exit_code,
                "stdout": stdout[:500] if stdout else None,  # Truncate for logging
                "stderr": stderr[:500] if stderr else None
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32000,  # System error
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.TOOL_EXECUTION,
            context=context,
            recoverable=True,
            guidance="Check tool parameters and backend CLI availability",
            **kwargs
        )
        
        self.tool_name = tool_name
        self.cli_command = cli_command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class TimeoutError(MCPException):
    """Exception for operation timeouts."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation_type: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation=operation_type or "timeout",
            execution_time_ms=timeout_duration * 1000 if timeout_duration else None,
            additional_context={
                "timeout_duration": timeout_duration,
                "operation_type": operation_type
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32000,  # System error
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.TIMEOUT,
            context=context,
            recoverable=True,
            retry_after=30,
            guidance="Operation timed out. Consider breaking down the request or increasing timeout limits",
            **kwargs
        )
        
        self.timeout_duration = timeout_duration
        self.operation_type = operation_type


class RateLimitError(MCPException):
    """Exception for rate limiting violations."""
    
    def __init__(
        self,
        message: str,
        rate_limit_type: str = "requests",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="rate_limiting",
            additional_context={
                "rate_limit_type": rate_limit_type,
                "limit": limit,
                "window_seconds": window_seconds
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32000,  # System error
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.RATE_LIMIT,
            context=context,
            recoverable=True,
            retry_after=retry_after or 60,
            guidance=f"Rate limit exceeded for {rate_limit_type}. Please wait before retrying",
            **kwargs
        )
        
        self.rate_limit_type = rate_limit_type
        self.limit = limit
        self.window_seconds = window_seconds


class ConfigurationError(MCPException):
    """Exception for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        config_source: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="configuration",
            additional_context={
                "config_key": config_key,
                "config_value": str(config_value) if config_value is not None else None,
                "config_source": config_source
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32001,  # Configuration error
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recoverable=False,
            guidance="Review configuration settings and ensure all required values are provided",
            **kwargs
        )
        
        self.config_key = config_key
        self.config_value = config_value
        self.config_source = config_source


class DependencyError(MCPException):
    """Exception for external dependency failures."""
    
    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        dependency_type: str = "external_service",
        last_success: Optional[datetime] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="dependency_check",
            additional_context={
                "dependency_name": dependency_name,
                "dependency_type": dependency_type,
                "last_success": last_success.isoformat() if last_success else None
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32000,  # System error
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.DEPENDENCY,
            context=context,
            recoverable=True,
            retry_after=120,
            guidance="External dependency is unavailable. The system will retry automatically",
            **kwargs
        )
        
        self.dependency_name = dependency_name
        self.dependency_type = dependency_type
        self.last_success = last_success


class BusinessLogicError(MCPException):
    """Exception for business logic violations."""
    
    def __init__(
        self,
        message: str,
        business_rule: Optional[str] = None,
        violated_constraint: Optional[str] = None,
        **kwargs
    ):
        context = ErrorContext(
            operation="business_logic_validation",
            additional_context={
                "business_rule": business_rule,
                "violated_constraint": violated_constraint
            }
        )
        
        # Remove context from kwargs if present to avoid conflict
        kwargs.pop('context', None)
        
        super().__init__(
            message=message,
            error_code=-32602,  # Invalid params (business logic)
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.BUSINESS_LOGIC,
            context=context,
            recoverable=True,
            guidance="Request violates business rules. Please adjust parameters or contact support",
            **kwargs
        )
        
        self.business_rule = business_rule
        self.violated_constraint = violated_constraint


# Exception registry for easy access
EXCEPTION_REGISTRY = {
    "validation": ValidationError,
    "security": SecurityError,
    "communication": CommunicationError,
    "tool_execution": ToolExecutionError,
    "timeout": TimeoutError,
    "rate_limit": RateLimitError,
    "configuration": ConfigurationError,
    "dependency": DependencyError,
    "business_logic": BusinessLogicError,
    "base": MCPException
}


def create_exception(
    exception_type: str,
    message: str,
    **kwargs
) -> MCPException:
    """
    Factory function to create typed exceptions.
    
    Args:
        exception_type: Type of exception to create
        message: Error message
        **kwargs: Additional exception-specific parameters
        
    Returns:
        Appropriate exception instance
    """
    exception_class = EXCEPTION_REGISTRY.get(exception_type, MCPException)
    return exception_class(message, **kwargs) 