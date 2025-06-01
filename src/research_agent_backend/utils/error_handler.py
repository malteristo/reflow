"""
Comprehensive error handling and recovery system.

This module provides error recovery mechanisms, user-friendly formatting,
notification systems, and error aggregation capabilities.

Implements FR-EH-006 through FR-EH-010: Error recovery and management.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Type, Union
from datetime import datetime, timedelta

from ..exceptions.system_exceptions import (
    ResearchAgentError,
    ErrorSeverity,
    ErrorContext,
    ConfigurationSystemError,
    DatabaseSystemError,
    ModelSystemError,
    FileSystemError,
    NetworkSystemError
)


@dataclass
class RecoveryResult:
    """Result of an error recovery attempt."""
    success: bool
    attempts: int
    recovery_time: float
    details: Optional[str] = None


@dataclass
class ErrorSummary:
    """Summary of error statistics."""
    total_errors: int
    error_counts: Dict[str, int]
    recent_errors: List[ResearchAgentError]
    error_rate: float  # errors per minute


class ErrorRecoveryManager:
    """Manages automatic error recovery attempts."""
    
    def __init__(self, max_attempts: int = 3, timeout: float = 30.0):
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.recovery_actions: Dict[Type, List[Callable]] = defaultdict(list)
    
    def register_recovery_action(
        self,
        error_type: Type[ResearchAgentError],
        action: Callable
    ) -> None:
        """Register a recovery action for a specific error type."""
        self.recovery_actions[error_type].append(action)
    
    def attempt_recovery(self, error: ResearchAgentError) -> RecoveryResult:
        """Attempt to recover from an error."""
        start_time = time.time()
        error_type = type(error)
        
        # Find recovery actions for this error type
        actions = self.recovery_actions.get(error_type, [])
        if not actions:
            return RecoveryResult(
                success=False,
                attempts=0,
                recovery_time=0.0,
                details="No recovery actions available"
            )
        
        # Attempt recovery with retries
        for attempt in range(1, self.max_attempts + 1):
            try:
                for action in actions:
                    result = action()
                    if result:  # Recovery successful
                        recovery_time = time.time() - start_time
                        return RecoveryResult(
                            success=True,
                            attempts=attempt,
                            recovery_time=recovery_time,
                            details="Recovery successful"
                        )
            except Exception as e:
                if attempt == self.max_attempts:
                    recovery_time = time.time() - start_time
                    return RecoveryResult(
                        success=False,
                        attempts=attempt,
                        recovery_time=recovery_time,
                        details=f"Recovery failed: {e}"
                    )
                # Wait before retrying
                time.sleep(min(2 ** (attempt - 1), 10))  # Exponential backoff
        
        recovery_time = time.time() - start_time
        return RecoveryResult(
            success=False,
            attempts=self.max_attempts,
            recovery_time=recovery_time,
            details="Max recovery attempts exceeded"
        )


class UserFriendlyErrorFormatter:
    """Formats errors in a user-friendly way."""
    
    def format_error(self, error: ResearchAgentError) -> str:
        """Format error for user display."""
        lines = []
        
        # Header based on error type
        if isinstance(error, ConfigurationSystemError):
            lines.append("⚙️ Configuration Problem")
        elif isinstance(error, DatabaseSystemError):
            lines.append("🗄️ Database Issue")
        elif isinstance(error, ModelSystemError):
            lines.append("🤖 AI Model Error")
        elif isinstance(error, FileSystemError):
            lines.append("📁 File System Problem")
        elif isinstance(error, NetworkSystemError):
            lines.append("🌐 Network Connection Issue")
        else:
            lines.append("❌ System Error")
        
        lines.append("=" * 50)
        
        # Error message
        lines.append(f"Problem: {error.message}")
        
        # Context information
        if error.context and error.context.operation:
            lines.append(f"During: {error.context.operation}")
        
        # Specific error details
        if hasattr(error, 'config_file') and error.config_file:
            lines.append(f"Config file: {error.config_file}")
        elif hasattr(error, 'file_path') and error.file_path:
            lines.append(f"File: {error.file_path}")
        elif hasattr(error, 'endpoint') and error.endpoint:
            lines.append(f"Endpoint: {error.endpoint}")
        
        # Suggested actions
        if error.suggested_actions:
            lines.append("")
            lines.append("💡 Try these solutions:")
            for action in error.suggested_actions:
                lines.append(f"  • {action}")
        
        # Error ID for support
        lines.append("")
        lines.append(f"Error ID: {error.error_id}")
        
        return "\n".join(lines)


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(
        self,
        enable_recovery: bool = True,
        max_recovery_attempts: int = 3,
        recovery_timeout: float = 30.0,
        enable_notifications: bool = False,
        logger: Optional[logging.Logger] = None,
        audit_logger: Optional = None,
        performance_logger: Optional = None
    ):
        self.enable_recovery = enable_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout = recovery_timeout
        self.enable_notifications = enable_notifications
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        self.audit_logger = audit_logger
        self.performance_logger = performance_logger
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.notification_handlers: List[Callable] = []
        
        # Recovery system
        self.recovery_manager = ErrorRecoveryManager(
            max_attempts=max_recovery_attempts,
            timeout=recovery_timeout
        )
    
    def handle_error(
        self,
        error: Union[Exception, ResearchAgentError],
        context: Optional[ErrorContext] = None
    ) -> ResearchAgentError:
        """Handle an error with comprehensive processing."""
        start_time = time.time()
        
        # Convert to ResearchAgentError if needed
        if not isinstance(error, ResearchAgentError):
            handled_error = ResearchAgentError(
                message=str(error),
                context=context or ErrorContext(operation="unknown"),
                original_exception=error
            )
        else:
            handled_error = error
            if context:
                handled_error.context = context
        
        # Track error
        self._track_error(handled_error)
        
        # Log error
        self._log_error(handled_error)
        
        # Attempt recovery if enabled
        if self.enable_recovery:
            recovery_result = self.recovery_manager.attempt_recovery(handled_error)
            if recovery_result.success:
                self.logger.info(f"Successfully recovered from error: {handled_error.error_id}")
        
        # Send notifications if enabled
        if self.enable_notifications and handled_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_notifications(handled_error)
        
        # Log performance metrics if performance logger is available
        if self.performance_logger:
            duration = time.time() - start_time
            self.performance_logger.log_metric(
                "error_handling",
                duration,
                unit="seconds",
                error_type=type(handled_error).__name__,
                severity=handled_error.severity.value
            )
        
        return handled_error
    
    def _track_error(self, error: ResearchAgentError) -> None:
        """Track error for statistics and reporting."""
        self.error_history.append(error)
        self.error_counts[type(error).__name__] += 1
    
    def _log_error(self, error: ResearchAgentError) -> None:
        """Log error with appropriate level and context."""
        # Prepare log data (avoid conflicting with built-in LogRecord fields)
        log_data = {
            "error_id": error.error_id,
            "error_type": type(error).__name__,
            "severity": error.severity.value,
            "error_message": error.message,  # Changed from 'message' to 'error_message'
            "context": error.context.to_dict() if error.context else {},
            "suggested_actions": [str(action) for action in error.suggested_actions],
            "timestamp": error.timestamp.isoformat()
        }
        
        # Add specific error attributes to log data
        if hasattr(error, 'config_file') and error.config_file:
            log_data["config_file"] = error.config_file
        if hasattr(error, 'file_path') and error.file_path:
            log_data["file_path"] = error.file_path
        if hasattr(error, 'endpoint') and error.endpoint:
            log_data["endpoint"] = error.endpoint
        if hasattr(error, 'database_type') and error.database_type:
            log_data["database_type"] = error.database_type
        if hasattr(error, 'model_name') and error.model_name:
            log_data["model_name"] = error.model_name
        
        # Create message that includes key details for standard formatters
        message_parts = [error.message]
        if hasattr(error, 'config_file') and error.config_file:
            message_parts.append(f"Config: {error.config_file}")
        if error.context and error.context.user_id:
            message_parts.append(f"User: {error.context.user_id}")
        
        detailed_message = " | ".join(message_parts)
        
        # Log based on severity with extra data
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {detailed_message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {detailed_message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {detailed_message}", extra=log_data)
        else:
            self.logger.info(f"Low severity error: {detailed_message}", extra=log_data)
    
    def _send_notifications(self, error: ResearchAgentError) -> None:
        """Send error notifications to registered handlers."""
        for handler in self.notification_handlers:
            try:
                handler(error, error.severity)
            except Exception as e:
                self.logger.warning(f"Notification handler failed: {e}")
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification handler."""
        self.notification_handlers.append(handler)
    
    def get_error_summary(self) -> ErrorSummary:
        """Get summary of error statistics."""
        # Calculate error rate (errors per minute)
        now = datetime.now()
        recent_errors = [
            error for error in self.error_history
            if (now - error.timestamp).total_seconds() < 3600  # Last hour
        ]
        error_rate = len(recent_errors) / 60.0 if recent_errors else 0.0
        
        return ErrorSummary(
            total_errors=len(self.error_history),
            error_counts=dict(self.error_counts),
            recent_errors=list(self.error_history),
            error_rate=error_rate
        )
    
    def get_recent_errors(self, limit: int = 10) -> List[ResearchAgentError]:
        """Get most recent errors."""
        return list(self.error_history)[-limit:] 