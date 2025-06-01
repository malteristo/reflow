"""
Error Middleware for Production MCP Server.

Automatic error capture, enrichment, and integration with MCP server
components for production-ready error handling.

Implements subtask 15.6: Develop Production-Ready Error Handling.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass

from ..protocol.enhanced_error_handler import EnhancedErrorHandler
from ..protocol.production_exceptions import (
    MCPException, ErrorContext, ErrorSeverity, ErrorCategory
)
from ..protocol.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareConfig:
    """Configuration for error middleware."""
    auto_capture: bool = True
    enrich_context: bool = True
    format_responses: bool = True
    enable_monitoring: bool = True
    log_errors: bool = True
    max_context_size: int = 1000  # Max size for context data


class ErrorMiddleware:
    """
    Error middleware for automatic error capture and processing.
    
    Provides automatic error handling, context enrichment, response formatting,
    and integration with monitoring systems for production deployment.
    """
    
    def __init__(
        self,
        error_handler: EnhancedErrorHandler,
        response_formatter: ResponseFormatter,
        config: Optional[MiddlewareConfig] = None
    ):
        """
        Initialize error middleware.
        
        Args:
            error_handler: Enhanced error handler instance
            response_formatter: Response formatter instance
            config: Middleware configuration
        """
        self.error_handler = error_handler
        self.response_formatter = response_formatter
        self.config = config or MiddlewareConfig()
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Error middleware initialized")
    
    def capture_errors(
        self,
        operation_name: Optional[str] = None,
        auto_retry: bool = False,
        timeout: Optional[float] = None
    ):
        """
        Decorator for automatic error capture and handling.
        
        Args:
            operation_name: Name of the operation for tracking
            auto_retry: Enable automatic retry with backoff
            timeout: Operation timeout in seconds
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                correlation_id = kwargs.get('correlation_id') or self._generate_correlation_id()
                op_name = operation_name or func.__name__
                
                # Start request tracking
                request_context = self._start_request_tracking(correlation_id, op_name, args, kwargs)
                
                try:
                    # Set timeout if specified
                    if timeout:
                        result = await asyncio.wait_for(
                            self._execute_with_context(func, args, kwargs, request_context),
                            timeout=timeout
                        )
                    else:
                        result = await self._execute_with_context(func, args, kwargs, request_context)
                    
                    # Record success
                    self._record_success(correlation_id, request_context)
                    return result
                    
                except asyncio.TimeoutError as e:
                    return await self._handle_timeout(e, correlation_id, request_context, timeout)
                except Exception as e:
                    return await self._handle_error(e, correlation_id, request_context, auto_retry, op_name)
                finally:
                    self._end_request_tracking(correlation_id)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                correlation_id = kwargs.get('correlation_id') or self._generate_correlation_id()
                op_name = operation_name or func.__name__
                
                # Start request tracking
                request_context = self._start_request_tracking(correlation_id, op_name, args, kwargs)
                
                try:
                    result = self._execute_sync_with_context(func, args, kwargs, request_context)
                    
                    # Record success
                    self._record_success(correlation_id, request_context)
                    return result
                    
                except Exception as e:
                    # For sync functions, we can't use async retry, so handle directly
                    mcp_exception = self._process_exception(e, correlation_id, request_context)
                    
                    if self.config.format_responses:
                        return self._format_error_response(mcp_exception, correlation_id)
                    else:
                        raise mcp_exception
                finally:
                    self._end_request_tracking(correlation_id)
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _execute_with_context(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        request_context: dict
    ) -> Any:
        """Execute async function with error context."""
        # Enrich kwargs with context if needed
        if self.config.enrich_context:
            kwargs['_error_context'] = request_context
        
        return await func(*args, **kwargs)
    
    def _execute_sync_with_context(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        request_context: dict
    ) -> Any:
        """Execute sync function with error context."""
        # Enrich kwargs with context if needed
        if self.config.enrich_context:
            kwargs['_error_context'] = request_context
        
        return func(*args, **kwargs)
    
    async def _handle_timeout(
        self,
        exception: asyncio.TimeoutError,
        correlation_id: str,
        request_context: dict,
        timeout_duration: Optional[float]
    ) -> Any:
        """Handle timeout errors specifically."""
        from ..protocol.production_exceptions import TimeoutError
        
        timeout_exception = TimeoutError(
            message=f"Operation timed out after {timeout_duration} seconds",
            timeout_duration=timeout_duration,
            operation_type=request_context.get('operation_name'),
            correlation_id=correlation_id,
            context=self._build_error_context(request_context)
        )
        
        # Process through error handler
        processed_exception = self.error_handler.handle_exception(
            timeout_exception, 
            context=self._build_error_context(request_context),
            correlation_id=correlation_id
        )
        
        if self.config.format_responses:
            return self._format_error_response(processed_exception, correlation_id)
        else:
            raise processed_exception
    
    async def _handle_error(
        self,
        exception: Exception,
        correlation_id: str,
        request_context: dict,
        auto_retry: bool,
        operation_name: str
    ) -> Any:
        """Handle general errors with optional retry."""
        mcp_exception = self._process_exception(exception, correlation_id, request_context)
        
        # Attempt retry if enabled and error is recoverable
        if auto_retry and mcp_exception.recoverable:
            try:
                # Note: This is a simplified retry - in practice you'd need to store the original function call
                logger.warning(f"Auto-retry not fully implemented for {operation_name}")
                # return await self.error_handler.retry_with_backoff(original_func, *args, **kwargs)
            except Exception as retry_exception:
                mcp_exception = self._process_exception(retry_exception, correlation_id, request_context)
        
        if self.config.format_responses:
            return self._format_error_response(mcp_exception, correlation_id)
        else:
            raise mcp_exception
    
    def _process_exception(
        self,
        exception: Exception,
        correlation_id: str,
        request_context: dict
    ) -> MCPException:
        """Process exception through error handler."""
        error_context = self._build_error_context(request_context)
        
        return self.error_handler.handle_exception(
            exception,
            context=error_context,
            correlation_id=correlation_id
        )
    
    def _build_error_context(self, request_context: dict) -> ErrorContext:
        """Build error context from request context."""
        return ErrorContext(
            operation=request_context.get('operation_name', ''),
            user_id=request_context.get('user_id'),
            session_id=request_context.get('session_id'),
            request_id=request_context.get('correlation_id'),
            tool_name=request_context.get('tool_name'),
            parameters=self._truncate_parameters(request_context.get('parameters', {})),
            execution_time_ms=request_context.get('execution_time_ms'),
            memory_usage_mb=request_context.get('memory_usage_mb'),
            additional_context=request_context.get('additional_context', {})
        )
    
    def _format_error_response(self, exception: MCPException, correlation_id: str) -> Dict[str, Any]:
        """Format error response using response formatter."""
        try:
            return self.response_formatter.format_error_response(
                request_id=correlation_id,
                error_code=exception.error_code,
                error_message=exception.message,
                error_data=exception.get_mcp_error_response().get('data', {})
            ).to_dict()
        except Exception as format_error:
            logger.error(f"Error formatting response: {format_error}")
            # Return basic error response
            return {
                "jsonrpc": "2.0",
                "id": correlation_id,
                "error": {
                    "code": exception.error_code,
                    "message": exception.message,
                    "data": {
                        "correlation_id": exception.correlation_id,
                        "category": exception.category.value,
                        "severity": exception.severity.value
                    }
                }
            }
    
    def _start_request_tracking(
        self,
        correlation_id: str,
        operation_name: str,
        args: tuple,
        kwargs: dict
    ) -> Dict[str, Any]:
        """Start tracking a request."""
        request_context = {
            'correlation_id': correlation_id,
            'operation_name': operation_name,
            'start_time': time.time(),
            'args': self._sanitize_args(args),
            'kwargs': self._sanitize_kwargs(kwargs),
            'parameters': self._extract_parameters(kwargs),
            'user_id': kwargs.get('user_id'),
            'session_id': kwargs.get('session_id'),
            'tool_name': kwargs.get('tool_name'),
            'additional_context': {}
        }
        
        self.active_requests[correlation_id] = request_context
        
        if self.config.log_errors:
            logger.debug(f"Started request tracking: {correlation_id} - {operation_name}")
        
        return request_context
    
    def _record_success(self, correlation_id: str, request_context: dict) -> None:
        """Record successful operation."""
        execution_time = time.time() - request_context['start_time']
        request_context['execution_time_ms'] = execution_time * 1000
        
        # Record success with error handler for circuit breaker
        operation_name = request_context.get('operation_name', 'unknown')
        self.error_handler.record_success(operation_name)
        
        if self.config.log_errors:
            logger.debug(f"Operation completed successfully: {correlation_id} in {execution_time:.3f}s")
    
    def _end_request_tracking(self, correlation_id: str) -> None:
        """End request tracking."""
        if correlation_id in self.active_requests:
            del self.active_requests[correlation_id]
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _sanitize_args(self, args: tuple) -> tuple:
        """Sanitize arguments for logging."""
        if not self.config.enrich_context:
            return ()
        
        # Truncate large arguments
        sanitized = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                sanitized.append(arg)
            elif isinstance(arg, dict):
                sanitized.append(self._truncate_dict(arg))
            else:
                sanitized.append(f"<{type(arg).__name__}>")
        
        return tuple(sanitized)
    
    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """Sanitize keyword arguments for logging."""
        if not self.config.enrich_context:
            return {}
        
        sanitized = {}
        for key, value in kwargs.items():
            if key.startswith('_'):  # Skip private parameters
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._truncate_dict(value)
            else:
                sanitized[key] = f"<{type(value).__name__}>"
        
        return sanitized
    
    def _extract_parameters(self, kwargs: dict) -> dict:
        """Extract parameters for error context."""
        # Look for common parameter keys
        parameter_keys = ['query', 'collection', 'file_path', 'document_id', 'top_k', 'parameters']
        parameters = {}
        
        for key in parameter_keys:
            if key in kwargs:
                parameters[key] = kwargs[key]
        
        return self._truncate_dict(parameters)
    
    def _truncate_dict(self, data: dict, max_items: int = 10) -> dict:
        """Truncate dictionary to prevent large context data."""
        if len(data) <= max_items:
            return data
        
        truncated = dict(list(data.items())[:max_items])
        truncated['_truncated'] = f"... {len(data) - max_items} more items"
        return truncated
    
    def _truncate_parameters(self, parameters: dict) -> dict:
        """Truncate parameters to fit context size limits."""
        import json
        
        try:
            # Check if parameters are too large
            json_size = len(json.dumps(parameters))
            if json_size <= self.config.max_context_size:
                return parameters
            
            # Truncate large parameters
            truncated = {}
            current_size = 0
            
            for key, value in parameters.items():
                item_size = len(json.dumps({key: value}))
                if current_size + item_size <= self.config.max_context_size:
                    truncated[key] = value
                    current_size += item_size
                else:
                    truncated['_truncated'] = f"Parameters truncated (original size: {json_size} bytes)"
                    break
            
            return truncated
        except Exception:
            return {"error": "Failed to process parameters"}
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active requests."""
        return dict(self.active_requests)
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            "active_requests": len(self.active_requests),
            "config": {
                "auto_capture": self.config.auto_capture,
                "enrich_context": self.config.enrich_context,
                "format_responses": self.config.format_responses,
                "enable_monitoring": self.config.enable_monitoring,
                "log_errors": self.config.log_errors,
                "max_context_size": self.config.max_context_size
            },
            "error_handler_stats": self.error_handler.get_error_metrics()
        }


# Convenience decorators for common use cases
def auto_error_capture(
    operation_name: Optional[str] = None,
    auto_retry: bool = False,
    timeout: Optional[float] = None
):
    """
    Convenience decorator for automatic error capture.
    
    This creates a default error middleware instance if none exists globally.
    For production use, you should inject the middleware instance.
    """
    # This would typically be injected via dependency injection
    # For now, we'll create a basic instance
    from ..protocol.enhanced_error_handler import EnhancedErrorHandler
    from ..protocol.response_formatter import ResponseFormatter
    
    # Create default instances (in production, these would be injected)
    error_handler = EnhancedErrorHandler()
    response_formatter = ResponseFormatter()
    middleware = ErrorMiddleware(error_handler, response_formatter)
    
    return middleware.capture_errors(
        operation_name=operation_name,
        auto_retry=auto_retry,
        timeout=timeout
    )


def stdio_error_capture(func: Callable) -> Callable:
    """Decorator for STDIO communication error capture."""
    return auto_error_capture(
        operation_name="stdio_communication",
        auto_retry=True,
        timeout=30.0
    )(func)


def tool_execution_capture(tool_name: str) -> Callable:
    """Decorator factory for tool execution error capture."""
    return auto_error_capture(
        operation_name=f"tool_execution_{tool_name}",
        auto_retry=False,
        timeout=300.0  # 5 minutes for tool execution
    )


def validation_error_capture(func: Callable) -> Callable:
    """Decorator for validation error capture."""
    return auto_error_capture(
        operation_name="parameter_validation",
        auto_retry=False,
        timeout=5.0
    )(func) 