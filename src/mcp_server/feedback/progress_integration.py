"""
Progress Reporting Integration Module

This module provides integration between the feedback/progress reporting system
and existing MCP server components including STDIO communication, response formatting,
error handling, MCP tools, and validation systems.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Union, Type, Set
from enum import Enum
import inspect
from functools import wraps

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of system integrations."""
    STDIO_COMMUNICATION = "stdio_communication"
    RESPONSE_FORMATTER = "response_formatter"
    ERROR_HANDLER = "error_handler"
    MCP_TOOLS = "mcp_tools"
    VALIDATION_SYSTEM = "validation_system"


@dataclass
class IntegrationConfig:
    """Configuration for system integrations."""
    enabled_integrations: Set[IntegrationType] = field(default_factory=lambda: set(IntegrationType))
    stdio_config: Dict[str, Any] = field(default_factory=dict)
    response_format_config: Dict[str, Any] = field(default_factory=dict)
    error_handling_config: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    
    def is_integration_enabled(self, integration: IntegrationType) -> bool:
        """Check if a specific integration is enabled."""
        return integration in self.enabled_integrations


@dataclass
class StdioMessage:
    """STDIO communication message structure."""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[Union[str, int]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string for STDIO transmission."""
        return json.dumps(asdict(self), separators=(',', ':'))


@dataclass
class ResponseFormat:
    """Standardized response format for MCP operations."""
    status: str = "success"
    data: Any = None
    progress: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def add_progress_info(self, progress_data: Dict[str, Any]):
        """Add progress information to the response."""
        self.progress = progress_data
    
    def add_feedback_info(self, feedback_data: Dict[str, Any]):
        """Add feedback information to the response."""
        self.feedback = feedback_data


@dataclass
class ErrorContext:
    """Context information for error handling integration."""
    operation_id: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    user_message: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StdioProgressCommunicator:
    """Handles progress communication via STDIO."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.message_queue = []
        self.is_initialized = True
    
    def send_progress_message(self, operation_id: str, progress: float, 
                             message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send progress message via STDIO."""
        stdio_message = StdioMessage(
            method="progress/update",
            params={
                "operation_id": operation_id,
                "progress": progress,
                "message": message,
                "metadata": metadata or {},
                "timestamp": time.time()
            },
            id=f"progress_{operation_id}_{int(time.time() * 1000)}"
        )
        
        self.message_queue.append(stdio_message)
        return stdio_message
    
    def send_status_message(self, operation_id: str, status: str, 
                           details: Optional[Dict[str, Any]] = None):
        """Send status message via STDIO."""
        stdio_message = StdioMessage(
            method="status/update",
            params={
                "operation_id": operation_id,
                "status": status,
                "details": details or {},
                "timestamp": time.time()
            },
            id=f"status_{operation_id}_{int(time.time() * 1000)}"
        )
        
        self.message_queue.append(stdio_message)
        return stdio_message
    
    def get_pending_messages(self) -> List[StdioMessage]:
        """Get all pending messages for transmission."""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages


class ResponseFormatterIntegration:
    """Integrates progress reporting with response formatting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.formatter_cache = {}
        self.is_initialized = True
    
    def format_response_with_progress(self, operation_data: Any, 
                                    progress_info: Optional[Dict[str, Any]] = None,
                                    feedback_info: Optional[Dict[str, Any]] = None) -> ResponseFormat:
        """Format response including progress and feedback information."""
        response = ResponseFormat(
            status="success",
            data=operation_data,
            metadata={
                "formatted_at": time.time(),
                "formatter": "ResponseFormatterIntegration"
            }
        )
        
        if progress_info:
            response.add_progress_info(progress_info)
        
        if feedback_info:
            response.add_feedback_info(feedback_info)
        
        return response
    
    def format_error_response_with_context(self, error_context: ErrorContext) -> ResponseFormat:
        """Format error response with progress context."""
        return ResponseFormat(
            status="error",
            data={
                "error_type": error_context.error_type,
                "error_message": error_context.error_message,
                "user_message": error_context.user_message,
                "recovery_suggestions": error_context.recovery_suggestions
            },
            metadata={
                "operation_id": error_context.operation_id,
                "error_context": error_context.metadata,
                "formatted_at": time.time()
            }
        )
    
    def get_cached_formats(self) -> Dict[str, Any]:
        """Get cached response formats."""
        return self.formatter_cache.copy()


class ErrorHandlerIntegration:
    """Integrates progress reporting with error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_history = []
        self.recovery_strategies = {}
        self.is_initialized = True
    
    def handle_progress_error(self, operation_id: str, error: Exception, 
                            context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle errors that occur during progress reporting."""
        error_context = ErrorContext(
            operation_id=operation_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=None,  # Would include actual stack trace in real implementation
            user_message=f"An error occurred in operation {operation_id}",
            recovery_suggestions=[
                "Retry the operation",
                "Check system resources",
                "Verify operation parameters"
            ],
            metadata=context or {}
        )
        
        self.error_history.append(error_context)
        return error_context
    
    def register_recovery_strategy(self, error_type: str, 
                                 strategy: Callable[[ErrorContext], Any]):
        """Register a recovery strategy for specific error types."""
        self.recovery_strategies[error_type] = strategy
    
    def execute_recovery(self, error_context: ErrorContext) -> Any:
        """Execute recovery strategy for an error."""
        strategy = self.recovery_strategies.get(error_context.error_type)
        if strategy:
            return strategy(error_context)
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }


class McpToolsIntegration:
    """Integrates progress reporting with MCP tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tracked_tools = {}
        self.tool_operations = {}
        self.is_initialized = True
    
    def register_tool_for_tracking(self, tool_name: str, tool_function: Callable):
        """Register an MCP tool for automatic progress tracking."""
        self.tracked_tools[tool_name] = tool_function
        
        @wraps(tool_function)
        def tracked_wrapper(*args, **kwargs):
            operation_id = f"tool_{tool_name}_{int(time.time() * 1000)}"
            self.tool_operations[operation_id] = {
                "tool_name": tool_name,
                "start_time": time.time(),
                "status": "running",
                "args": args,
                "kwargs": kwargs
            }
            
            try:
                result = tool_function(*args, **kwargs)
                self.tool_operations[operation_id]["status"] = "completed"
                self.tool_operations[operation_id]["result"] = result
                return result
            except Exception as e:
                self.tool_operations[operation_id]["status"] = "error"
                self.tool_operations[operation_id]["error"] = str(e)
                raise
            finally:
                self.tool_operations[operation_id]["end_time"] = time.time()
        
        return tracked_wrapper
    
    def get_tool_progress(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get progress information for a specific tool."""
        return [
            op for op in self.tool_operations.values()
            if op["tool_name"] == tool_name
        ]
    
    def get_all_tool_operations(self) -> Dict[str, Any]:
        """Get all tracked tool operations."""
        return self.tool_operations.copy()
    
    def cleanup_completed_operations(self, max_age_seconds: int = 3600):
        """Clean up old completed operations."""
        current_time = time.time()
        to_remove = []
        
        for op_id, op_data in self.tool_operations.items():
            if ("end_time" in op_data and 
                current_time - op_data["end_time"] > max_age_seconds):
                to_remove.append(op_id)
        
        for op_id in to_remove:
            del self.tool_operations[op_id]


class ValidationSystemIntegration:
    """Integrates progress reporting with parameter validation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = {}
        self.validation_history = []
        self.is_initialized = True
    
    def register_validation_rule(self, parameter_name: str, 
                                validator: Callable[[Any], bool],
                                error_message: str = "Validation failed"):
        """Register a validation rule for progress/feedback parameters."""
        self.validation_rules[parameter_name] = {
            "validator": validator,
            "error_message": error_message
        }
    
    def validate_progress_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate progress reporting parameters."""
        validation_result = {
            "valid": True,
            "errors": [],
            "validated_parameters": parameters.copy()
        }
        
        for param_name, param_value in parameters.items():
            if param_name in self.validation_rules:
                rule = self.validation_rules[param_name]
                try:
                    if not rule["validator"](param_value):
                        validation_result["valid"] = False
                        validation_result["errors"].append({
                            "parameter": param_name,
                            "message": rule["error_message"],
                            "value": param_value
                        })
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append({
                        "parameter": param_name,
                        "message": f"Validation error: {str(e)}",
                        "value": param_value
                    })
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def validate_feedback_parameters(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feedback parameters."""
        # Basic feedback validation
        validation_result = {
            "valid": True,
            "errors": [],
            "validated_feedback": feedback_data.copy()
        }
        
        required_fields = ["type", "message"]
        for field in required_fields:
            if field not in feedback_data:
                validation_result["valid"] = False
                validation_result["errors"].append({
                    "field": field,
                    "message": f"Required field '{field}' is missing"
                })
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v["valid"])
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "recent_validations": self.validation_history[-10:] if self.validation_history else []
        }


class ProgressReportingIntegrationManager:
    """Manages all progress reporting integrations."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.integrations = {}
        self.is_initialized = False
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize all enabled integrations."""
        if self.config.is_integration_enabled(IntegrationType.STDIO_COMMUNICATION):
            self.integrations[IntegrationType.STDIO_COMMUNICATION] = StdioProgressCommunicator(
                self.config.stdio_config
            )
        
        if self.config.is_integration_enabled(IntegrationType.RESPONSE_FORMATTER):
            self.integrations[IntegrationType.RESPONSE_FORMATTER] = ResponseFormatterIntegration(
                self.config.response_format_config
            )
        
        if self.config.is_integration_enabled(IntegrationType.ERROR_HANDLER):
            self.integrations[IntegrationType.ERROR_HANDLER] = ErrorHandlerIntegration(
                self.config.error_handling_config
            )
        
        if self.config.is_integration_enabled(IntegrationType.MCP_TOOLS):
            self.integrations[IntegrationType.MCP_TOOLS] = McpToolsIntegration()
        
        if self.config.is_integration_enabled(IntegrationType.VALIDATION_SYSTEM):
            self.integrations[IntegrationType.VALIDATION_SYSTEM] = ValidationSystemIntegration(
                self.config.validation_config
            )
        
        self.is_initialized = True
    
    def get_integration(self, integration_type: IntegrationType) -> Optional[Any]:
        """Get a specific integration instance."""
        return self.integrations.get(integration_type)
    
    def is_integration_available(self, integration_type: IntegrationType) -> bool:
        """Check if an integration is available and initialized."""
        integration = self.integrations.get(integration_type)
        return integration is not None and getattr(integration, 'is_initialized', False)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        status = {}
        for integration_type in IntegrationType:
            integration = self.integrations.get(integration_type)
            status[integration_type.value] = {
                "enabled": self.config.is_integration_enabled(integration_type),
                "available": integration is not None,
                "initialized": getattr(integration, 'is_initialized', False) if integration else False
            }
        return status 