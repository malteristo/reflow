"""
Comprehensive logging configuration and management system.

This module provides structured logging with JSON format, rotation,
performance metrics, and audit trail capabilities.

Implements FR-LG-001 through FR-LG-006: Comprehensive logging system.
Optimized for memory efficiency and performance.
"""

import json
import logging
import logging.handlers
import time
import threading
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, TextIO, List
from dataclasses import dataclass
from functools import lru_cache


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    """Log format types."""
    STANDARD = "standard"
    JSON = "json"
    DETAILED = "detailed"


class OptimizedJSONFormatter(logging.Formatter):
    """
    Optimized JSON formatter with caching and lazy evaluation.
    Reduces CPU overhead for repeated similar log messages.
    """
    
    def __init__(self):
        super().__init__()
        self._cache_size = 100
        self._format_cache = {}
        self._cache_lock = threading.Lock()
    
    @lru_cache(maxsize=256)
    def _get_base_log_data(self, name: str, levelname: str, module: str, funcName: str, lineno: int) -> Dict[str, Any]:
        """Cache base log data for repeated log entries."""
        return {
            "logger": name,
            "level": levelname,
            "module": module,
            "function": funcName,
            "line": lineno
        }
    
    def _extract_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields efficiently, avoiding expensive iterations where possible."""
        extra_data = {}
        
        # Check for pre-computed extra_data first (fastest path)
        if hasattr(record, 'extra_data'):
            extra_data.update(record.extra_data)
        
        # Only iterate through attributes for high-priority logs to reduce overhead
        if record.levelno >= logging.WARNING:
            # Define standard LogRecord attributes to exclude (cached set for performance)
            standard_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info', 'message', 'extra_data'
            }
            
            # Only check for extra attributes on high-priority logs
            for attr_name in record.__dict__:
                if not attr_name.startswith('_') and attr_name not in standard_attrs:
                    attr_value = record.__dict__[attr_name]
                    if not callable(attr_value):
                        extra_data[attr_name] = attr_value
        
        return extra_data
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with optimizations."""
        # Create cache key for similar log entries
        cache_key = f"{record.name}:{record.levelname}:{record.module}:{record.funcName}"
        
        # Get base data (cached)
        base_data = self._get_base_log_data(
            record.name, record.levelname, record.module, 
            record.funcName, record.lineno
        )
        
        # Build complete log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "message": record.getMessage(),
            **base_data
        }
        
        # Add extra fields efficiently
        extra_data = self._extract_extra_fields(record)
        if extra_data:
            log_data.update(extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Use fast JSON serialization with optimizations
        try:
            return json.dumps(log_data, default=str, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            return json.dumps({
                "timestamp": log_data["timestamp"],
                "level": log_data["level"],
                "logger": log_data["logger"],
                "message": str(record.getMessage()),
                "serialization_error": "Failed to serialize additional data"
            }, separators=(',', ':'))


class BufferedPerformanceLogger:
    """
    Optimized performance logger with buffering and batch writes.
    Reduces I/O overhead for high-frequency performance metrics.
    """
    
    def __init__(self, log_file: Optional[Path] = None, buffer_size: int = 50, flush_interval: float = 5.0):
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.logger = logging.getLogger("performance")
        self._buffer = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._last_flush = time.time()
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup the buffered performance logger."""
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            formatter = OptimizedJSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _should_flush(self) -> bool:
        """Determine if buffer should be flushed."""
        return (len(self._buffer) >= self.buffer_size or 
                time.time() - self._last_flush >= self.flush_interval)
    
    def _flush_buffer(self) -> None:
        """Flush buffered metrics to log file."""
        with self._buffer_lock:
            if not self._buffer:
                return
            
            # Process all buffered entries
            while self._buffer:
                log_data = self._buffer.popleft()
                
                record = self.logger.makeRecord(
                    name=self.logger.name,
                    level=logging.INFO,
                    fn="",
                    lno=0,
                    msg=log_data.get("message", "Performance metric"),
                    args=(),
                    exc_info=None
                )
                
                # Add metric data
                record.extra_data = log_data
                self.logger.handle(record)
            
            self._last_flush = time.time()
    
    @contextmanager
    def time_operation(self, operation_name: str, **context):
        """Context manager for timing operations with optimized logging."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Only log operations that take significant time for low-priority operations
            if duration > 0.001 or context.get('force_log', False):
                log_data = {
                    "message": f"Operation timing: {operation_name}",
                    "operation": operation_name,
                    "duration": duration,
                    **context
                }
                
                with self._buffer_lock:
                    self._buffer.append(log_data)
                    
                    if self._should_flush():
                        self._flush_buffer()
    
    def log_metric(self, metric_name: str, value: Union[int, float], unit: str = "", **context) -> None:
        """Log a performance metric with buffering."""
        log_data = {
            "message": f"Metric: {metric_name}",
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            **context
        }
        
        with self._buffer_lock:
            self._buffer.append(log_data)
            
            if self._should_flush():
                self._flush_buffer()
    
    def force_flush(self) -> None:
        """Force flush of all buffered metrics."""
        self._flush_buffer()


class LoggingManager:
    """
    Comprehensive logging manager with optimized performance.
    """
    
    def __init__(
        self,
        log_level: LogLevel = LogLevel.INFO,
        log_format: LogFormat = LogFormat.STANDARD,
        log_file: Optional[Path] = None,
        enable_console: bool = True,
        enable_rotation: bool = False,
        max_file_size: str = "10MB",
        backup_count: int = 5
    ):
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self.enable_console = enable_console
        self.enable_rotation = enable_rotation
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """Setup the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup formatters
        formatters = self._create_formatters()
        
        # Add console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level.value)
            console_handler.setFormatter(formatters[self.log_format])
            root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.log_file:
            file_handler = self._create_file_handler()
            file_handler.setLevel(self.log_level.value)
            file_handler.setFormatter(formatters[self.log_format])
            root_logger.addHandler(file_handler)
    
    def _create_formatters(self) -> Dict[LogFormat, logging.Formatter]:
        """Create different log formatters with optimizations."""
        return {
            LogFormat.STANDARD: logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            LogFormat.JSON: OptimizedJSONFormatter(),
            LogFormat.DETAILED: logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        }
    
    def _create_file_handler(self) -> logging.Handler:
        """Create appropriate file handler based on rotation settings."""
        if self.enable_rotation:
            # Parse max file size
            if self.max_file_size.endswith('KB'):
                max_bytes = int(self.max_file_size[:-2]) * 1024
            elif self.max_file_size.endswith('MB'):
                max_bytes = int(self.max_file_size[:-2]) * 1024 * 1024
            elif self.max_file_size.endswith('GB'):
                max_bytes = int(self.max_file_size[:-2]) * 1024 * 1024 * 1024
            else:
                max_bytes = int(self.max_file_size)
            
            return logging.handlers.RotatingFileHandler(
                filename=self.log_file,
                maxBytes=max_bytes,
                backupCount=self.backup_count
            )
        else:
            return logging.FileHandler(self.log_file)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger instance."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level.value)
            self._loggers[name] = logger
        
        return self._loggers[name]


class StructuredLogger:
    """
    Structured logger with optimized context handling.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        format_type: LogFormat = LogFormat.JSON
    ):
        self.name = name
        self.log_file = log_file
        self.format_type = format_type
        
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup the structured logger."""
        # Create formatter
        if self.format_type == LogFormat.JSON:
            formatter = OptimizedJSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add file handler if specified
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log_with_extra(self, level: int, message: str, **kwargs) -> None:
        """Log message with extra structured data using optimized approach."""
        # Pre-process kwargs to avoid expensive operations during logging
        extra_data = {}
        for key, value in kwargs.items():
            try:
                # Test serializability early to avoid issues in formatter
                json.dumps(value, default=str)
                extra_data[key] = value
            except (TypeError, ValueError):
                extra_data[key] = str(value)
        
        # Create a custom log record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add pre-processed extra data
        record.extra_data = extra_data
        
        # Handle the record
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log_with_extra(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log_with_extra(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log_with_extra(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log_with_extra(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        self._log_with_extra(logging.CRITICAL, message, **kwargs)


# For backward compatibility, keep the original PerformanceLogger but use optimized version
class PerformanceLogger(BufferedPerformanceLogger):
    """Performance logger with backward compatibility."""
    pass


class AuditLogger:
    """
    Logger specialized for user action auditing with optimizations.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup the audit logger."""
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            formatter = OptimizedJSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        **context
    ) -> None:
        """Log a user action for audit purposes with optimization."""
        # Pre-validate details for serializability
        validated_details = {}
        if details:
            for key, value in details.items():
                try:
                    json.dumps(value, default=str)
                    validated_details[key] = value
                except (TypeError, ValueError):
                    validated_details[key] = str(value)
        
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"User action: {action}",
            args=(),
            exc_info=None
        )
        
        record.extra_data = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": validated_details,
            "success": success,
            **context
        }
        
        self.logger.handle(record) 