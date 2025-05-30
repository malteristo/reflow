"""
Configuration file paths and constants for Research Agent.

This module provides the ConfigPaths dataclass containing default paths
and constants used throughout the configuration system.

Implements FR-CF-001: Configuration-driven behavior with centralized paths.
"""

from dataclasses import dataclass


@dataclass
class ConfigPaths:
    """Configuration file paths and constants."""
    
    DEFAULT_CONFIG_FILE: str = "researchagent.config.json"
    DEFAULT_CONFIG_DIR: str = "./config/defaults"
    SCHEMA_DIR: str = "./config/schema"
    ENV_FILE: str = ".env"
    DEFAULT_CONFIG_SCHEMA: str = "config_schema.json" 