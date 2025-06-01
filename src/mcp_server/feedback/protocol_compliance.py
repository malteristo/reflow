"""
Feedback Protocol Compliance Module

This module ensures all feedback and progress reporting follows proper protocol
standards including JSON-RPC 2.0 and MCP protocol specifications.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Supported content types for feedback messages."""
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    JSON = "application/json"


class FeedbackMessageType(Enum):
    """Types of feedback messages according to MCP protocol."""
    PROGRESS = "progress"
    STATUS = "status"
    NOTIFICATION = "notification"
    RESPONSE = "response"
    ERROR = "error"


@dataclass
class JsonRpcMessage:
    """JSON-RPC 2.0 compliant message structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            "jsonrpc": self.jsonrpc,
            "timestamp": self.timestamp
        }
        
        if self.id is not None:
            data["id"] = self.id
        if self.method is not None:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
            
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class McpFeedbackMessage:
    """MCP protocol compliant feedback message."""
    message_type: FeedbackMessageType
    content: Any
    content_type: ContentType = ContentType.JSON
    operation_id: Optional[str] = None
    progress_percentage: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    ttl: Optional[float] = None  # Time to live in seconds
    
    def to_jsonrpc(self, request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Convert to JSON-RPC format."""
        params = {
            "type": self.message_type.value,
            "content": self.content,
            "contentType": self.content_type.value,
            "metadata": self.metadata,
            "priority": self.priority
        }
        
        if self.operation_id:
            params["operationId"] = self.operation_id
        if self.progress_percentage is not None:
            params["progress"] = self.progress_percentage
        if self.ttl is not None:
            params["ttl"] = self.ttl
            
        return JsonRpcMessage(
            id=request_id,
            method="feedback/message",
            params=params
        )


class ProtocolCompliantFeedbackFormatter:
    """Formats feedback messages according to protocol specifications."""
    
    def __init__(self, protocol_version: str = "1.0"):
        """Initialize the protocol compliant formatter."""
        self.protocol_version = protocol_version
        self.message_cache: Dict[str, JsonRpcMessage] = {}
        self.timing_constraints = {
            "max_response_time": 5.0,  # seconds
            "max_batch_size": 100,
            "heartbeat_interval": 30.0  # seconds
        }
        logger.info(f"Protocol compliant formatter initialized (v{protocol_version})")
    
    def format_progress_message(self, 
                              operation_id: str,
                              progress: float,
                              message: str,
                              metadata: Optional[Dict[str, Any]] = None,
                              request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Format a progress message according to protocol."""
        feedback_msg = McpFeedbackMessage(
            message_type=FeedbackMessageType.PROGRESS,
            content=message,
            content_type=ContentType.TEXT,
            operation_id=operation_id,
            progress_percentage=progress,
            metadata=metadata or {}
        )
        
        jsonrpc_msg = feedback_msg.to_jsonrpc(request_id)
        self._validate_message_timing(jsonrpc_msg)
        return jsonrpc_msg
    
    def format_status_message(self,
                            operation_id: str,
                            status: str,
                            details: Optional[Dict[str, Any]] = None,
                            request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Format a status message according to protocol."""
        feedback_msg = McpFeedbackMessage(
            message_type=FeedbackMessageType.STATUS,
            content={
                "status": status,
                "details": details or {}
            },
            content_type=ContentType.JSON,
            operation_id=operation_id
        )
        
        jsonrpc_msg = feedback_msg.to_jsonrpc(request_id)
        self._validate_message_timing(jsonrpc_msg)
        return jsonrpc_msg
    
    def format_error_message(self,
                           error_code: int,
                           error_message: str,
                           error_data: Optional[Dict[str, Any]] = None,
                           request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Format an error message according to JSON-RPC 2.0."""
        error_obj = {
            "code": error_code,
            "message": error_message
        }
        
        if error_data:
            error_obj["data"] = error_data
            
        jsonrpc_msg = JsonRpcMessage(
            id=request_id,
            error=error_obj
        )
        
        self._validate_message_timing(jsonrpc_msg)
        return jsonrpc_msg
    
    def format_notification(self,
                          message: str,
                          level: str = "info",
                          metadata: Optional[Dict[str, Any]] = None) -> JsonRpcMessage:
        """Format a notification message."""
        feedback_msg = McpFeedbackMessage(
            message_type=FeedbackMessageType.NOTIFICATION,
            content={
                "level": level,
                "message": message
            },
            content_type=ContentType.JSON,
            metadata=metadata or {}
        )
        
        jsonrpc_msg = feedback_msg.to_jsonrpc()
        self._validate_message_timing(jsonrpc_msg)
        return jsonrpc_msg
    
    def validate_content_type(self, content: Any, content_type: ContentType) -> bool:
        """Validate content matches declared content type."""
        try:
            if content_type == ContentType.JSON:
                if isinstance(content, (dict, list)):
                    json.dumps(content)
                    return True
                else:
                    return False
            elif content_type == ContentType.TEXT:
                return isinstance(content, str)
            elif content_type == ContentType.MARKDOWN:
                return isinstance(content, str)
            return False
        except (TypeError, ValueError):
            return False
    
    def validate_jsonrpc_format(self, message: JsonRpcMessage) -> tuple[bool, List[str]]:
        """Validate JSON-RPC 2.0 format compliance."""
        errors = []
        
        # Check required fields
        if message.jsonrpc != "2.0":
            errors.append("Invalid or missing 'jsonrpc' field (must be '2.0')")
        
        # Check message type
        if message.method and message.result:
            errors.append("Message cannot have both 'method' and 'result' fields")
        
        if message.method and message.error:
            errors.append("Message cannot have both 'method' and 'error' fields")
        
        if message.result and message.error:
            errors.append("Message cannot have both 'result' and 'error' fields")
        
        # Validate error format
        if message.error:
            if not isinstance(message.error, dict):
                errors.append("Error field must be an object")
            else:
                if "code" not in message.error:
                    errors.append("Error object must contain 'code' field")
                if "message" not in message.error:
                    errors.append("Error object must contain 'message' field")
                if not isinstance(message.error.get("code"), int):
                    errors.append("Error code must be an integer")
                if not isinstance(message.error.get("message"), str):
                    errors.append("Error message must be a string")
        
        return len(errors) == 0, errors
    
    def validate_mcp_compliance(self, message: JsonRpcMessage) -> tuple[bool, List[str]]:
        """Validate MCP protocol compliance."""
        errors = []
        
        # Check if it's a feedback message
        if message.method == "feedback/message":
            if not message.params:
                errors.append("Feedback message must have params")
                return False, errors
            
            params = message.params
            
            # Check required feedback fields
            if "type" not in params:
                errors.append("Feedback params must contain 'type' field")
            elif params["type"] not in [t.value for t in FeedbackMessageType]:
                errors.append(f"Invalid feedback type: {params['type']}")
            
            if "content" not in params:
                errors.append("Feedback params must contain 'content' field")
            
            if "contentType" not in params:
                errors.append("Feedback params must contain 'contentType' field")
            elif params["contentType"] not in [ct.value for ct in ContentType]:
                errors.append(f"Invalid content type: {params['contentType']}")
            
            # Validate content type consistency
            if "content" in params and "contentType" in params:
                content_type = ContentType(params["contentType"])
                if not self.validate_content_type(params["content"], content_type):
                    errors.append(f"Content does not match declared type {content_type.value}")
        
        return len(errors) == 0, errors
    
    def validate_timing_compliance(self, message: JsonRpcMessage) -> tuple[bool, List[str]]:
        """Validate message timing compliance."""
        errors = []
        
        current_time = time.time()
        message_age = current_time - message.timestamp
        
        # Check response time
        if message_age > self.timing_constraints["max_response_time"]:
            errors.append(f"Message exceeds max response time ({message_age:.2f}s > {self.timing_constraints['max_response_time']}s)")
        
        # Check if message has TTL and if it's expired
        if (message.method == "feedback/message" and 
            message.params and 
            "ttl" in message.params):
            ttl = message.params["ttl"]
            if message_age > ttl:
                errors.append(f"Message has expired (age: {message_age:.2f}s > TTL: {ttl}s)")
        
        return len(errors) == 0, errors
    
    def _validate_message_timing(self, message: JsonRpcMessage) -> None:
        """Internal method to validate message timing."""
        is_valid, errors = self.validate_timing_compliance(message)
        if not is_valid:
            logger.warning(f"Timing validation failed: {'; '.join(errors)}")
    
    def batch_messages(self, messages: List[JsonRpcMessage]) -> List[List[JsonRpcMessage]]:
        """Batch messages according to protocol constraints."""
        if len(messages) <= self.timing_constraints["max_batch_size"]:
            return [messages]
        
        batches = []
        for i in range(0, len(messages), self.timing_constraints["max_batch_size"]):
            batch = messages[i:i + self.timing_constraints["max_batch_size"]]
            batches.append(batch)
        
        return batches


class ProtocolComplianceValidator:
    """Validates all feedback messages for protocol compliance."""
    
    def __init__(self):
        """Initialize the protocol compliance validator."""
        self.formatter = ProtocolCompliantFeedbackFormatter()
        self.validation_cache: Dict[str, bool] = {}
        self.error_counts: Dict[str, int] = {}
        logger.info("Protocol compliance validator initialized")
    
    def validate_feedback_message(self, message: JsonRpcMessage) -> tuple[bool, List[str]]:
        """Comprehensive validation of feedback message."""
        all_errors = []
        
        # JSON-RPC 2.0 validation
        jsonrpc_valid, jsonrpc_errors = self.formatter.validate_jsonrpc_format(message)
        if not jsonrpc_valid:
            all_errors.extend([f"JSON-RPC: {e}" for e in jsonrpc_errors])
        
        # MCP protocol validation
        mcp_valid, mcp_errors = self.formatter.validate_mcp_compliance(message)
        if not mcp_valid:
            all_errors.extend([f"MCP: {e}" for e in mcp_errors])
        
        # Timing validation
        timing_valid, timing_errors = self.formatter.validate_timing_compliance(message)
        if not timing_valid:
            all_errors.extend([f"Timing: {e}" for e in timing_errors])
        
        is_valid = len(all_errors) == 0
        
        # Track validation results
        message_key = f"{message.method}:{message.id}"
        self.validation_cache[message_key] = is_valid
        
        if not is_valid:
            self.error_counts[message_key] = self.error_counts.get(message_key, 0) + 1
            logger.error(f"Protocol validation failed for {message_key}: {'; '.join(all_errors)}")
        
        return is_valid, all_errors
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validated = len(self.validation_cache)
        successful_validations = sum(1 for valid in self.validation_cache.values() if valid)
        
        return {
            "total_validated": total_validated,
            "successful_validations": successful_validations,
            "validation_success_rate": successful_validations / total_validated if total_validated > 0 else 0,
            "error_counts": dict(self.error_counts),
            "most_common_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    async def validate_async_message(self, message: JsonRpcMessage) -> tuple[bool, List[str]]:
        """Asynchronous validation for non-blocking operations."""
        # Simulate async validation (in real implementation, this might involve
        # network calls to validate protocol versions, etc.)
        await asyncio.sleep(0.001)  # Minimal delay to ensure async behavior
        return self.validate_feedback_message(message)


class FeedbackProtocolManager:
    """Manages protocol compliance for all feedback operations."""
    
    def __init__(self):
        """Initialize the feedback protocol manager."""
        self.formatter = ProtocolCompliantFeedbackFormatter()
        self.validator = ProtocolComplianceValidator()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        logger.info("Feedback protocol manager initialized")
    
    def create_compliant_progress_message(self,
                                        operation_id: str,
                                        progress: float,
                                        message: str,
                                        metadata: Optional[Dict[str, Any]] = None,
                                        request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Create a protocol-compliant progress message."""
        jsonrpc_msg = self.formatter.format_progress_message(
            operation_id, progress, message, metadata, request_id
        )
        
        # Validate before returning
        is_valid, errors = self.validator.validate_feedback_message(jsonrpc_msg)
        if not is_valid:
            logger.error(f"Created invalid progress message: {'; '.join(errors)}")
        
        return jsonrpc_msg
    
    def create_compliant_status_message(self,
                                      operation_id: str,
                                      status: str,
                                      details: Optional[Dict[str, Any]] = None,
                                      request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Create a protocol-compliant status message."""
        jsonrpc_msg = self.formatter.format_status_message(
            operation_id, status, details, request_id
        )
        
        # Validate before returning
        is_valid, errors = self.validator.validate_feedback_message(jsonrpc_msg)
        if not is_valid:
            logger.error(f"Created invalid status message: {'; '.join(errors)}")
        
        return jsonrpc_msg
    
    def create_compliant_error_message(self,
                                     error_code: int,
                                     error_message: str,
                                     error_data: Optional[Dict[str, Any]] = None,
                                     request_id: Optional[Union[str, int]] = None) -> JsonRpcMessage:
        """Create a protocol-compliant error message."""
        jsonrpc_msg = self.formatter.format_error_message(
            error_code, error_message, error_data, request_id
        )
        
        # Validate before returning
        is_valid, errors = self.validator.validate_feedback_message(jsonrpc_msg)
        if not is_valid:
            logger.error(f"Created invalid error message: {'; '.join(errors)}")
        
        return jsonrpc_msg
    
    def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protocol compliance statistics."""
        validation_stats = self.validator.get_validation_statistics()
        
        return {
            "protocol_version": self.formatter.protocol_version,
            "active_operations": len(self.active_operations),
            "validation_statistics": validation_stats,
            "timing_constraints": self.formatter.timing_constraints
        } 