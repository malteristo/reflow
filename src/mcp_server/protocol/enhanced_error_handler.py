"""
Enhanced Error Handler for Production MCP Server.

Comprehensive error handling with classification, recovery, monitoring,
security, and production-ready features.

Implements subtask 15.6: Develop Production-Ready Error Handling.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from .production_exceptions import (
    MCPException, ErrorSeverity, ErrorCategory, ErrorContext,
    ValidationError, SecurityError, CommunicationError, ToolExecutionError,
    TimeoutError, RateLimitError, ConfigurationError, DependencyError,
    BusinessLogicError, create_exception
)
from .error_handler import MCPErrorCode

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for resilience management."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorHandlerConfig:
    """Configuration for enhanced error handler."""
    enable_monitoring: bool = True
    enable_security_features: bool = True
    enable_rate_limiting: bool = True
    max_recent_errors: int = 1000
    circuit_breaker_threshold: int = 5
    enable_retry_logic: bool = True
    enable_circuit_breaker: bool = True


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring and observability."""
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_hour: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_reset: datetime = field(default_factory=datetime.utcnow)
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.total_errors = 0
        self.errors_by_category.clear()
        self.errors_by_severity.clear()
        self.errors_by_hour.clear()
        self.last_reset = datetime.utcnow()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker patterns."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    error_rate_threshold: float = 0.1  # 10% error rate threshold
    window_size_minutes: int = 5


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class MonitoringHook:
    """Hook for error monitoring and alerting."""
    name: str
    callback: Callable[[MCPException], None]
    severity_filter: Optional[List[ErrorSeverity]] = None
    category_filter: Optional[List[ErrorCategory]] = None


class EnhancedErrorHandler:
    """
    Production-ready error handler with comprehensive error management.
    
    Provides error classification, recovery mechanisms, monitoring,
    security features, and observability for MCP server operations.
    """
    
    def __init__(
        self,
        config: Optional[ErrorHandlerConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_security_features: Optional[bool] = None,
        enable_monitoring: Optional[bool] = None,
        enable_rate_limiting: Optional[bool] = None
    ):
        """
        Initialize enhanced error handler.
        
        Args:
            config: Main error handler configuration
            circuit_breaker_config: Circuit breaker configuration
            rate_limit_config: Rate limiting configuration
            retry_config: Retry logic configuration
            enable_security_features: Enable security error handling (overrides config)
            enable_monitoring: Enable error monitoring and metrics (overrides config)
            enable_rate_limiting: Enable rate limiting (overrides config)
        """
        # Use provided config or default
        self.config = config or ErrorHandlerConfig()
        
        # Override config values with explicit parameters if provided
        self.enable_security_features = (
            enable_security_features 
            if enable_security_features is not None 
            else self.config.enable_security_features
        )
        self.enable_monitoring = (
            enable_monitoring 
            if enable_monitoring is not None 
            else self.config.enable_monitoring
        )
        self.enable_rate_limiting = (
            enable_rate_limiting 
            if enable_rate_limiting is not None 
            else self.config.enable_rate_limiting
        )
        
        # Initialize sub-configurations
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()
        
        # Error tracking and metrics - use config max_recent_errors
        self.error_metrics = ErrorMetrics()
        self.recent_errors = deque(maxlen=self.config.max_recent_errors)
        self.error_correlation_map: Dict[str, List[str]] = defaultdict(list)
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.request_history = deque(maxlen=10000)
        self.error_rate_history = deque(maxlen=1000)
        
        # Monitoring hooks
        self.monitoring_hooks: List[MonitoringHook] = []
        
        # Security monitoring
        self.security_events = deque(maxlen=100)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        logger.info("Enhanced error handler initialized with production features")
    
    @property
    def circuit_breaker_state(self) -> str:
        """Get the overall circuit breaker state (for simple access in tests)."""
        # Return the most restrictive state across all operations
        states = []
        for cb_data in self.circuit_breakers.values():
            if cb_data.get("state") == CircuitBreakerState.OPEN:
                return "open"
            elif cb_data.get("state") == CircuitBreakerState.HALF_OPEN:
                states.append("half_open")
            else:
                states.append("closed")
        
        # Return most restrictive state found
        if "half_open" in states:
            return "half_open"
        return "closed"
    
    def add_monitoring_hook(
        self,
        name: str,
        callback: Callable[[Dict[str, Any]], None],
        severity_filter: Optional[List[str]] = None,
        category_filter: Optional[List[str]] = None
    ) -> None:
        """
        Add a monitoring hook for error notifications.
        
        Args:
            name: Hook name for identification
            callback: Function to call with error data
            severity_filter: List of severities to filter on
            category_filter: List of categories to filter on
        """
        # Convert the test-friendly callback to match the existing register method
        def mcp_callback(exception: MCPException) -> None:
            error_data = {
                "message": exception.message,
                "severity": exception.severity,
                "category": exception.category,
                "correlation_id": exception.correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            callback(error_data)
        
        # Convert string filters to enums if needed
        severity_enum_filter = None
        if severity_filter:
            severity_enum_filter = [ErrorSeverity(s) for s in severity_filter if hasattr(ErrorSeverity, s.upper())]
        
        category_enum_filter = None
        if category_filter:
            category_enum_filter = [ErrorCategory(c) for c in category_filter if hasattr(ErrorCategory, c.upper())]
        
        self.register_monitoring_hook(name, mcp_callback, severity_enum_filter, category_enum_filter)
    
    def handle_exception(
        self,
        exception: Union[Exception, MCPException],
        context: Optional[ErrorContext] = None,
        correlation_id: Optional[str] = None
    ) -> MCPException:
        """
        Handle any exception and convert to MCPException with enhanced features.
        
        Args:
            exception: Exception to handle
            context: Additional context information
            correlation_id: Request correlation ID
            
        Returns:
            Enhanced MCPException with production features
        """
        # Convert to MCPException if needed
        if isinstance(exception, MCPException):
            mcp_exception = exception
        else:
            mcp_exception = self._convert_to_mcp_exception(exception, context, correlation_id)
        
        # Update context with correlation ID if provided
        if correlation_id and mcp_exception.context:
            mcp_exception.context.request_id = correlation_id
        
        # Process error through production pipeline
        self._process_error(mcp_exception)
        
        return mcp_exception
    
    def _convert_to_mcp_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        correlation_id: Optional[str] = None
    ) -> MCPException:
        """Convert standard exception to MCPException."""
        error_message = str(exception)
        exception_type = type(exception).__name__
        
        # Classify exception type - order matters, check more specific terms first
        if "config" in error_message.lower():
            return ConfigurationError(
                message=error_message,
                context=context,
                correlation_id=correlation_id,
                caused_by=exception
            )
        elif "timeout" in error_message.lower():
            return TimeoutError(
                message=error_message,
                context=context,
                correlation_id=correlation_id,
                caused_by=exception
            )
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            return CommunicationError(
                message=error_message,
                context=context,
                correlation_id=correlation_id,
                caused_by=exception
            )
        elif "validation" in error_message.lower() or "invalid" in error_message.lower():
            return ValidationError(
                message=error_message,
                context=context,
                correlation_id=correlation_id,
                caused_by=exception
            )
        else:
            return MCPException(
                message=f"{exception_type}: {error_message}",
                context=context,
                correlation_id=correlation_id,
                caused_by=exception
            )
    
    def _process_error(self, exception: MCPException) -> None:
        """Process error through production pipeline."""
        # Update metrics
        if self.enable_monitoring:
            self._update_error_metrics(exception)
        
        # Security processing
        if self.enable_security_features:
            self._process_security_aspects(exception)
        
        # Circuit breaker processing
        self._update_circuit_breaker(exception)
        
        # Rate limiting check
        self._check_rate_limits(exception)
        
        # Execute monitoring hooks
        self._execute_monitoring_hooks(exception)
        
        # Store for correlation analysis
        self._store_error_for_correlation(exception)
    
    def _update_error_metrics(self, exception: MCPException) -> None:
        """Update error metrics for monitoring."""
        self.error_metrics.total_errors += 1
        self.error_metrics.errors_by_category[exception.category.value] += 1
        self.error_metrics.errors_by_severity[exception.severity.value] += 1
        
        # Track errors by hour
        hour_key = exception.timestamp.strftime("%Y-%m-%d-%H")
        self.error_metrics.errors_by_hour[hour_key] += 1
        
        # Store recent error
        error_record = {
            "timestamp": exception.timestamp,
            "category": exception.category.value,
            "severity": exception.severity.value,
            "correlation_id": exception.correlation_id,
            "recoverable": exception.recoverable
        }
        self.recent_errors.append(error_record)
        
        logger.debug(f"Updated error metrics: {self.error_metrics.total_errors} total errors")
    
    def _process_security_aspects(self, exception: MCPException) -> None:
        """Process security aspects of errors."""
        if exception.category == ErrorCategory.SECURITY:
            # Record security event
            security_event = {
                "timestamp": exception.timestamp,
                "severity": exception.severity.value,
                "correlation_id": exception.correlation_id,
                "message": exception.message,
                "context": exception.context.__dict__ if exception.context else {}
            }
            self.security_events.append(security_event)
            
            # Track suspicious patterns
            if exception.context and exception.context.user_id:
                self.suspicious_patterns[exception.context.user_id] += 1
        
        # Sanitize error message for security
        exception.message = self._sanitize_error_message(exception.message)
    
    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to prevent information leakage."""
        # Remove potential file paths
        import re
        message = re.sub(r'/[a-zA-Z0-9_/.-]+', '[PATH_REDACTED]', message)
        
        # Remove potential API keys or tokens
        message = re.sub(r'[a-zA-Z0-9]{32,}', '[TOKEN_REDACTED]', message)
        
        # Remove potential internal server information
        message = re.sub(r'localhost:\d+', '[SERVER_REDACTED]', message)
        
        return message
    
    def _update_circuit_breaker(self, exception: MCPException) -> None:
        """Update circuit breaker state based on error."""
        if exception.category in [ErrorCategory.DEPENDENCY, ErrorCategory.NETWORK]:
            operation_key = exception.context.operation if exception.context else "unknown"
            
            if operation_key not in self.circuit_breakers:
                self.circuit_breakers[operation_key] = {
                    "state": CircuitBreakerState.CLOSED,
                    "failure_count": 0,
                    "last_failure": None,
                    "success_count": 0
                }
            
            breaker = self.circuit_breakers[operation_key]
            breaker["failure_count"] += 1
            breaker["last_failure"] = datetime.utcnow()
            
            # Check if we should open the circuit
            if (breaker["failure_count"] >= self.circuit_breaker_config.failure_threshold and
                breaker["state"] == CircuitBreakerState.CLOSED):
                breaker["state"] = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened for operation: {operation_key}")
    
    def _check_rate_limits(self, exception: MCPException) -> None:
        """Check and enforce rate limits."""
        now = datetime.utcnow()
        
        # Clean old entries
        cutoff = now - timedelta(minutes=self.rate_limit_config.window_size_minutes)
        while self.error_rate_history and self.error_rate_history[0]["timestamp"] < cutoff:
            self.error_rate_history.popleft()
        
        # Add current error
        self.error_rate_history.append({
            "timestamp": now,
            "severity": exception.severity.value,
            "category": exception.category.value
        })
        
        # Check error rate
        error_count = len(self.error_rate_history)
        total_requests = len(self.request_history)
        
        if total_requests > 0:
            error_rate = error_count / total_requests
            if error_rate > self.rate_limit_config.error_rate_threshold:
                logger.warning(f"High error rate detected: {error_rate:.2%}")
    
    def _execute_monitoring_hooks(self, exception: MCPException) -> None:
        """Execute registered monitoring hooks."""
        for hook in self.monitoring_hooks:
            try:
                # Check severity filter
                if (hook.severity_filter and 
                    exception.severity not in hook.severity_filter):
                    continue
                
                # Check category filter
                if (hook.category_filter and 
                    exception.category not in hook.category_filter):
                    continue
                
                # Execute hook
                hook.callback(exception)
            except Exception as e:
                logger.error(f"Error executing monitoring hook {hook.name}: {e}")
    
    def _store_error_for_correlation(self, exception: MCPException) -> None:
        """Store error for correlation analysis."""
        correlation_id = exception.correlation_id
        if correlation_id:
            self.error_correlation_map[correlation_id].append(exception.message)
    
    def register_monitoring_hook(
        self,
        name: str,
        callback: Callable[[MCPException], None],
        severity_filter: Optional[List[ErrorSeverity]] = None,
        category_filter: Optional[List[ErrorCategory]] = None
    ) -> None:
        """Register a monitoring hook for error notifications."""
        hook = MonitoringHook(
            name=name,
            callback=callback,
            severity_filter=severity_filter,
            category_filter=category_filter
        )
        self.monitoring_hooks.append(hook)
        logger.info(f"Registered monitoring hook: {name}")
    
    def get_circuit_breaker_state(self, operation: str) -> CircuitBreakerState:
        """Get circuit breaker state for an operation."""
        if operation not in self.circuit_breakers:
            return CircuitBreakerState.CLOSED
        
        breaker = self.circuit_breakers[operation]
        
        # Check if we should transition from OPEN to HALF_OPEN
        if (breaker["state"] == CircuitBreakerState.OPEN and
            breaker["last_failure"]):
            time_since_failure = (datetime.utcnow() - breaker["last_failure"]).total_seconds()
            if time_since_failure > self.circuit_breaker_config.recovery_timeout:
                breaker["state"] = CircuitBreakerState.HALF_OPEN
                breaker["success_count"] = 0
                logger.info(f"Circuit breaker transitioned to HALF_OPEN for: {operation}")
        
        return breaker["state"]
    
    def record_success(self, operation: str) -> None:
        """Record successful operation for circuit breaker."""
        if operation in self.circuit_breakers:
            breaker = self.circuit_breakers[operation]
            
            if breaker["state"] == CircuitBreakerState.HALF_OPEN:
                breaker["success_count"] += 1
                if breaker["success_count"] >= self.circuit_breaker_config.half_open_max_calls:
                    breaker["state"] = CircuitBreakerState.CLOSED
                    breaker["failure_count"] = 0
                    logger.info(f"Circuit breaker closed for operation: {operation}")
    
    async def retry_with_backoff(
        self,
        operation: Callable,
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Check circuit breaker
                breaker_state = self.get_circuit_breaker_state(operation_name)
                if breaker_state == CircuitBreakerState.OPEN:
                    raise DependencyError(
                        f"Circuit breaker is open for operation: {operation_name}",
                        dependency_name=operation_name
                    )
                
                # Execute operation
                result = await operation(*args, **kwargs) if asyncio.iscoroutinefunction(operation) else operation(*args, **kwargs)
                
                # Record success
                self.record_success(operation_name)
                return result
                
            except Exception as e:
                last_exception = e
                mcp_exception = self.handle_exception(e)
                
                # Check if we should retry
                if not mcp_exception.recoverable or attempt == self.retry_config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay
                )
                
                # Add jitter
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(f"Retry attempt {attempt + 1} for {operation_name} after {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # All retries failed
        if last_exception:
            raise self.handle_exception(last_exception)
        else:
            raise MCPException(f"All retry attempts failed for operation: {operation_name}")
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        return {
            "total_errors": self.error_metrics.total_errors,
            "errors_by_category": dict(self.error_metrics.errors_by_category),
            "errors_by_severity": dict(self.error_metrics.errors_by_severity),
            "errors_by_hour": dict(self.error_metrics.errors_by_hour),
            "recent_errors_count": len(self.recent_errors),
            "circuit_breakers": {
                op: {
                    "state": breaker["state"].value,
                    "failure_count": breaker["failure_count"]
                }
                for op, breaker in self.circuit_breakers.items()
            },
            "security_events_count": len(self.security_events),
            "monitoring_hooks_count": len(self.monitoring_hooks)
        }
    
    def reset_metrics(self) -> None:
        """Reset all error metrics."""
        self.error_metrics.reset_metrics()
        self.recent_errors.clear()
        self.error_correlation_map.clear()
        logger.info("Error metrics reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status based on error metrics."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        recent_critical_errors = sum(
            1 for error in self.recent_errors
            if error["severity"] == "critical" and error["timestamp"] > hour_ago
        )
        
        recent_error_rate = len([
            error for error in self.recent_errors
            if error["timestamp"] > hour_ago
        ]) / 60  # errors per minute
        
        open_circuit_breakers = sum(
            1 for breaker in self.circuit_breakers.values()
            if breaker["state"] == CircuitBreakerState.OPEN
        )
        
        # Determine health status
        if recent_critical_errors > 0 or open_circuit_breakers > 0:
            health_status = "unhealthy"
        elif recent_error_rate > 10:  # More than 10 errors per minute
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "recent_critical_errors": recent_critical_errors,
            "recent_error_rate": recent_error_rate,
            "open_circuit_breakers": open_circuit_breakers,
            "timestamp": now.isoformat()
        } 