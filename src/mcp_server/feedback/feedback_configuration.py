"""
Feedback Configuration for Structured Feedback and Progress Reporting.

This module provides comprehensive configuration management for all feedback
and progress reporting components including frequency settings, verbosity levels,
custom templates, and system-wide configuration loading.
"""

import json
import os
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class VerbosityLevel(Enum):
    """Verbosity levels for feedback messages."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    VERBOSE = "verbose"
    DEBUG = "debug"


class FeedbackType(Enum):
    """Types of feedback that can be configured."""
    PROGRESS = "progress"
    STATUS = "status"
    ERROR = "error"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ProgressReportingFrequencyConfig:
    """Configuration for progress reporting frequency."""
    update_interval_seconds: float = 1.0
    batch_size: int = 10
    max_events_in_memory: int = 1000
    min_progress_change_threshold: float = 1.0  # Minimum % change to report
    max_frequency_hz: float = 10.0  # Maximum updates per second
    enable_adaptive_frequency: bool = True  # Adjust frequency based on operation speed
    
    def get_effective_update_interval(self, operation_duration_estimate: Optional[float] = None) -> float:
        """Get effective update interval based on operation characteristics."""
        if not self.enable_adaptive_frequency or operation_duration_estimate is None:
            return self.update_interval_seconds
        
        # For longer operations, reduce frequency to avoid spam
        if operation_duration_estimate > 60:  # > 1 minute
            return max(self.update_interval_seconds, 2.0)
        elif operation_duration_estimate > 300:  # > 5 minutes
            return max(self.update_interval_seconds, 5.0)
        
        return self.update_interval_seconds


@dataclass
class FeedbackVerbosityConfig:
    """Configuration for feedback verbosity levels."""
    global_verbosity: VerbosityLevel = VerbosityLevel.STANDARD
    per_type_verbosity: Dict[str, VerbosityLevel] = field(default_factory=dict)
    include_timestamps: bool = True
    include_operation_context: bool = True
    include_performance_metrics: bool = False
    include_debug_info: bool = False
    
    def get_verbosity_for_type(self, feedback_type: FeedbackType) -> VerbosityLevel:
        """Get verbosity level for a specific feedback type."""
        return self.per_type_verbosity.get(feedback_type.value, self.global_verbosity)
    
    def should_include_detail(self, feedback_type: FeedbackType, detail_type: str) -> bool:
        """Check if a specific detail should be included based on verbosity."""
        verbosity = self.get_verbosity_for_type(feedback_type)
        
        detail_requirements = {
            "timestamps": VerbosityLevel.MINIMAL,
            "operation_context": VerbosityLevel.STANDARD,
            "performance_metrics": VerbosityLevel.DETAILED,
            "debug_info": VerbosityLevel.DEBUG,
            "stack_traces": VerbosityLevel.VERBOSE,
            "internal_state": VerbosityLevel.DEBUG
        }
        
        required_level = detail_requirements.get(detail_type, VerbosityLevel.STANDARD)
        verbosity_hierarchy = [
            VerbosityLevel.MINIMAL,
            VerbosityLevel.STANDARD,
            VerbosityLevel.DETAILED,
            VerbosityLevel.VERBOSE,
            VerbosityLevel.DEBUG
        ]
        
        return verbosity_hierarchy.index(verbosity) >= verbosity_hierarchy.index(required_level)


@dataclass
class FeedbackTemplate:
    """Template for feedback messages."""
    name: str
    pattern: str
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    description: str = ""
    
    def format_message(self, variables: Dict[str, Any]) -> str:
        """Format the template with provided variables."""
        # Check required variables
        missing_vars = [var for var in self.required_variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format the message
        try:
            return self.pattern.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template formatting failed: {e}")


@dataclass
class FeedbackTemplateConfig:
    """Configuration for feedback templates."""
    templates: Dict[str, FeedbackTemplate] = field(default_factory=dict)
    default_template_set: str = "standard"
    enable_custom_templates: bool = True
    
    def get_template(self, template_name: str) -> Optional[FeedbackTemplate]:
        """Get a specific template by name."""
        return self.templates.get(template_name)
    
    def add_template(self, template: FeedbackTemplate):
        """Add a new template."""
        self.templates[template.name] = template
    
    def load_default_templates(self):
        """Load default feedback templates."""
        default_templates = [
            FeedbackTemplate(
                name="progress_update",
                pattern="Progress: {progress}% - {message}",
                required_variables=["progress", "message"],
                optional_variables=["eta", "current_item"],
                description="Standard progress update template"
            ),
            FeedbackTemplate(
                name="progress_with_eta",
                pattern="Progress: {progress}% - {message} (ETA: {eta})",
                required_variables=["progress", "message", "eta"],
                description="Progress update with ETA"
            ),
            FeedbackTemplate(
                name="error_message",
                pattern="Error in {operation}: {error_message}",
                required_variables=["operation", "error_message"],
                optional_variables=["suggestions", "recovery_actions"],
                description="Standard error message template"
            ),
            FeedbackTemplate(
                name="suggestion_feedback",
                pattern="Suggestion: {suggestion} (Confidence: {confidence})",
                required_variables=["suggestion", "confidence"],
                optional_variables=["reasoning", "example"],
                description="Suggestion feedback template"
            )
        ]
        
        for template in default_templates:
            self.add_template(template)


@dataclass
class FeedbackSystemConfig:
    """Overall system configuration for feedback components."""
    enabled: bool = True
    max_concurrent_operations: int = 100
    operation_timeout_seconds: float = 300.0
    enable_caching: bool = True
    cache_size_limit: int = 1000
    log_level: str = "INFO"
    enable_metrics_collection: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class FeedbackConfiguration:
    """Manages configuration for feedback and progress reporting systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize feedback configuration."""
        self.frequency_config = ProgressReportingFrequencyConfig()
        self.verbosity_config = FeedbackVerbosityConfig()
        self.template_config = FeedbackTemplateConfig()
        self.system_config = FeedbackSystemConfig()
        
        self.config_path = config_path
        self.last_loaded = None
        self.is_loaded = False
        
        # Load default templates
        self.template_config.load_default_templates()
        
        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """Load configuration from file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load frequency configuration
            if "frequency" in config_data:
                freq_data = config_data["frequency"]
                self.frequency_config = ProgressReportingFrequencyConfig(**freq_data)
            
            # Load verbosity configuration
            if "verbosity" in config_data:
                verb_data = config_data["verbosity"]
                if "global_verbosity" in verb_data:
                    verb_data["global_verbosity"] = VerbosityLevel(verb_data["global_verbosity"])
                if "per_type_verbosity" in verb_data:
                    verb_data["per_type_verbosity"] = {
                        k: VerbosityLevel(v) for k, v in verb_data["per_type_verbosity"].items()
                    }
                self.verbosity_config = FeedbackVerbosityConfig(**verb_data)
            
            # Load template configuration
            if "templates" in config_data:
                template_data = config_data["templates"]
                if "custom_templates" in template_data:
                    for template_info in template_data["custom_templates"]:
                        template = FeedbackTemplate(**template_info)
                        self.template_config.add_template(template)
                
                if "default_template_set" in template_data:
                    self.template_config.default_template_set = template_data["default_template_set"]
                
                if "enable_custom_templates" in template_data:
                    self.template_config.enable_custom_templates = template_data["enable_custom_templates"]
            
            # Load system configuration
            if "system" in config_data:
                system_data = config_data["system"]
                self.system_config = FeedbackSystemConfig(**system_data)
            
            self.config_path = config_path
            self.last_loaded = time.time()
            self.is_loaded = True
            
            logger.info(f"Successfully loaded feedback configuration from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        try:
            save_path = config_path or self.config_path
            if not save_path:
                raise ValueError("No configuration path specified")
            
            config_data = {
                "frequency": asdict(self.frequency_config),
                "verbosity": {
                    "global_verbosity": self.verbosity_config.global_verbosity.value,
                    "per_type_verbosity": {
                        k: v.value for k, v in self.verbosity_config.per_type_verbosity.items()
                    },
                    "include_timestamps": self.verbosity_config.include_timestamps,
                    "include_operation_context": self.verbosity_config.include_operation_context,
                    "include_performance_metrics": self.verbosity_config.include_performance_metrics,
                    "include_debug_info": self.verbosity_config.include_debug_info
                },
                "templates": {
                    "custom_templates": [
                        asdict(template) for template in self.template_config.templates.values()
                        if template.name not in ["progress_update", "progress_with_eta", "error_message", "suggestion_feedback"]
                    ],
                    "default_template_set": self.template_config.default_template_set,
                    "enable_custom_templates": self.template_config.enable_custom_templates
                },
                "system": asdict(self.system_config)
            }
            
            config_file = Path(save_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Successfully saved feedback configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_frequency_config(self) -> ProgressReportingFrequencyConfig:
        """Get progress reporting frequency configuration."""
        return self.frequency_config
    
    def set_frequency_config(self, config: ProgressReportingFrequencyConfig):
        """Set progress reporting frequency configuration."""
        self.frequency_config = config
    
    def get_verbosity_config(self) -> FeedbackVerbosityConfig:
        """Get feedback verbosity configuration."""
        return self.verbosity_config
    
    def set_verbosity_level(self, level: VerbosityLevel, feedback_type: Optional[FeedbackType] = None):
        """Set verbosity level globally or for specific feedback type."""
        if feedback_type:
            self.verbosity_config.per_type_verbosity[feedback_type.value] = level
        else:
            self.verbosity_config.global_verbosity = level
    
    def get_template_config(self) -> FeedbackTemplateConfig:
        """Get feedback template configuration."""
        return self.template_config
    
    def add_custom_template(self, template: FeedbackTemplate):
        """Add a custom feedback template."""
        if self.template_config.enable_custom_templates:
            self.template_config.add_template(template)
        else:
            raise ValueError("Custom templates are disabled")
    
    def get_system_config(self) -> FeedbackSystemConfig:
        """Get system configuration."""
        return self.system_config
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed."""
        if not self.config_path or not self.last_loaded:
            return False
        
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                return False
            
            file_mtime = config_file.stat().st_mtime
            if file_mtime > self.last_loaded:
                return self.load_config(self.config_path)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check configuration file changes: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "is_loaded": self.is_loaded,
            "config_path": self.config_path,
            "last_loaded": self.last_loaded,
            "frequency": {
                "update_interval": self.frequency_config.update_interval_seconds,
                "batch_size": self.frequency_config.batch_size,
                "max_frequency": self.frequency_config.max_frequency_hz,
                "adaptive_frequency": self.frequency_config.enable_adaptive_frequency
            },
            "verbosity": {
                "global_level": self.verbosity_config.global_verbosity.value,
                "custom_levels": len(self.verbosity_config.per_type_verbosity),
                "include_timestamps": self.verbosity_config.include_timestamps,
                "include_debug": self.verbosity_config.include_debug_info
            },
            "templates": {
                "total_templates": len(self.template_config.templates),
                "custom_templates_enabled": self.template_config.enable_custom_templates,
                "default_set": self.template_config.default_template_set
            },
            "system": {
                "enabled": self.system_config.enabled,
                "max_concurrent": self.system_config.max_concurrent_operations,
                "caching_enabled": self.system_config.enable_caching,
                "metrics_enabled": self.system_config.enable_metrics_collection
            }
        } 