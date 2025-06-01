"""
Validation Registry for centralized validation rule management.

Provides a centralized system for managing validation rules,
custom validators, and validation configuration.

Implements subtask 15.4: Implement Parameter Validation Logic.
"""

import logging
from typing import Dict, Any, List, Callable, Optional, Union
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationRule:
    """
    Represents a single validation rule.
    """
    
    def __init__(self, 
                 name: str,
                 validator_func: Callable[[Any], Dict[str, Any]],
                 level: ValidationLevel = ValidationLevel.ERROR,
                 description: str = "",
                 category: str = "general"):
        """
        Initialize a validation rule.
        
        Args:
            name: Unique name for the rule
            validator_func: Function that validates a value
            level: Severity level of validation failures
            description: Human-readable description of the rule
            category: Category of validation (security, business, etc.)
        """
        self.name = name
        self.validator_func = validator_func
        self.level = level
        self.description = description
        self.category = category
    
    def validate(self, value: Any) -> Dict[str, Any]:
        """
        Execute the validation rule.
        
        Args:
            value: Value to validate
            
        Returns:
            Dict containing validation result
        """
        try:
            result = self.validator_func(value)
            result["rule_name"] = self.name
            result["level"] = self.level.value
            result["category"] = self.category
            return result
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation rule '{self.name}' failed: {str(e)}"],
                "rule_name": self.name,
                "level": self.level.value,
                "category": "validation_error"
            }


class ValidationRegistry:
    """
    Centralized registry for validation rules and configuration.
    
    Manages validation rules across the MCP server and provides
    a consistent interface for applying validation logic.
    """
    
    def __init__(self):
        """Initialize the validation registry."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Storage for validation rules
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_categories: Dict[str, List[str]] = {}
        
        # Configuration
        self.config = {
            "strict_mode": False,  # If True, warnings become errors
            "enabled_categories": ["security", "business_logic", "schema_validation"],
            "disabled_rules": [],
            "custom_error_messages": {}
        }
        
        # Initialize default rules
        self._register_default_rules()
    
    def register_rule(self, rule: ValidationRule) -> None:
        """
        Register a validation rule.
        
        Args:
            rule: ValidationRule instance to register
        """
        self.rules[rule.name] = rule
        
        # Track rule categories
        if rule.category not in self.rule_categories:
            self.rule_categories[rule.category] = []
        
        if rule.name not in self.rule_categories[rule.category]:
            self.rule_categories[rule.category].append(rule.name)
        
        self.logger.debug(f"Registered validation rule: {rule.name} (category: {rule.category})")
    
    def unregister_rule(self, rule_name: str) -> bool:
        """
        Unregister a validation rule.
        
        Args:
            rule_name: Name of the rule to unregister
            
        Returns:
            True if rule was removed, False if not found
        """
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            del self.rules[rule_name]
            
            # Remove from category tracking
            if rule.category in self.rule_categories:
                if rule_name in self.rule_categories[rule.category]:
                    self.rule_categories[rule.category].remove(rule_name)
            
            self.logger.debug(f"Unregistered validation rule: {rule_name}")
            return True
        return False
    
    def validate_value(self, value: Any, rule_names: List[str]) -> Dict[str, Any]:
        """
        Validate a value against specified rules.
        
        Args:
            value: Value to validate
            rule_names: List of rule names to apply
            
        Returns:
            Dict containing aggregated validation result
        """
        errors = []
        warnings = []
        info = []
        
        for rule_name in rule_names:
            if rule_name in self.config["disabled_rules"]:
                continue
            
            if rule_name not in self.rules:
                self.logger.warning(f"Validation rule not found: {rule_name}")
                continue
            
            rule = self.rules[rule_name]
            
            # Skip if category is disabled
            if rule.category not in self.config["enabled_categories"]:
                continue
            
            result = rule.validate(value)
            
            # Categorize results by level
            if not result.get("valid", True):
                if result.get("errors"):
                    if rule.level == ValidationLevel.ERROR or self.config["strict_mode"]:
                        errors.extend(result["errors"])
                    elif rule.level == ValidationLevel.WARNING:
                        warnings.extend(result["errors"])
                    else:
                        info.extend(result["errors"])
            
            # Add warnings and info messages if present
            if result.get("warnings"):
                warnings.extend(result["warnings"])
            
            if result.get("info"):
                info.extend(result["info"])
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info
        }
    
    def validate_by_category(self, value: Any, categories: List[str]) -> Dict[str, Any]:
        """
        Validate a value against all rules in specified categories.
        
        Args:
            value: Value to validate
            categories: List of categories to validate against
            
        Returns:
            Dict containing aggregated validation result
        """
        rule_names = []
        for category in categories:
            if category in self.rule_categories:
                rule_names.extend(self.rule_categories[category])
        
        return self.validate_value(value, rule_names)
    
    def get_rules_by_category(self, category: str) -> List[ValidationRule]:
        """
        Get all rules in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of ValidationRule instances
        """
        if category not in self.rule_categories:
            return []
        
        return [self.rules[rule_name] for rule_name in self.rule_categories[category]]
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update registry configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.logger.debug(f"Updated validation registry configuration: {config}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current registry configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config.copy()
    
    def export_rules(self) -> Dict[str, Any]:
        """
        Export all rules for backup or transfer.
        
        Returns:
            Dict containing rule definitions
        """
        exported = {
            "rules": {},
            "categories": self.rule_categories.copy(),
            "config": self.config.copy()
        }
        
        for name, rule in self.rules.items():
            exported["rules"][name] = {
                "name": rule.name,
                "level": rule.level.value,
                "description": rule.description,
                "category": rule.category
            }
        
        return exported
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        
        # Email validation rule
        def validate_email(value: str) -> Dict[str, Any]:
            """Basic email validation."""
            import re
            if not isinstance(value, str):
                return {"valid": False, "errors": ["Email must be a string"]}
            
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                return {"valid": False, "errors": ["Invalid email format"]}
            
            return {"valid": True, "errors": []}
        
        self.register_rule(ValidationRule(
            name="email_format",
            validator_func=validate_email,
            level=ValidationLevel.ERROR,
            description="Validates email address format",
            category="format"
        ))
        
        # URL validation rule
        def validate_url(value: str) -> Dict[str, Any]:
            """Basic URL validation."""
            import re
            if not isinstance(value, str):
                return {"valid": False, "errors": ["URL must be a string"]}
            
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, value):
                return {"valid": False, "errors": ["Invalid URL format"]}
            
            return {"valid": True, "errors": []}
        
        self.register_rule(ValidationRule(
            name="url_format",
            validator_func=validate_url,
            level=ValidationLevel.ERROR,
            description="Validates URL format",
            category="format"
        ))
        
        # Non-empty string validation
        def validate_non_empty_string(value: str) -> Dict[str, Any]:
            """Validate non-empty string."""
            if not isinstance(value, str):
                return {"valid": False, "errors": ["Value must be a string"]}
            
            if not value.strip():
                return {"valid": False, "errors": ["Value cannot be empty"]}
            
            return {"valid": True, "errors": []}
        
        self.register_rule(ValidationRule(
            name="non_empty_string",
            validator_func=validate_non_empty_string,
            level=ValidationLevel.ERROR,
            description="Validates that string is non-empty",
            category="basic"
        ))
        
        # Positive integer validation
        def validate_positive_integer(value: Any) -> Dict[str, Any]:
            """Validate positive integer."""
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return {"valid": False, "errors": ["Value must be an integer"]}
            
            if value <= 0:
                return {"valid": False, "errors": ["Value must be positive"]}
            
            return {"valid": True, "errors": []}
        
        self.register_rule(ValidationRule(
            name="positive_integer",
            validator_func=validate_positive_integer,
            level=ValidationLevel.ERROR,
            description="Validates positive integer values",
            category="numeric"
        ))
    
    def get_available_rules(self) -> Dict[str, List[str]]:
        """
        Get all available rules grouped by category.
        
        Returns:
            Dict mapping categories to lists of rule names
        """
        return self.rule_categories.copy()
    
    def get_rule_info(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            Dict containing rule information or None if not found
        """
        if rule_name not in self.rules:
            return None
        
        rule = self.rules[rule_name]
        return {
            "name": rule.name,
            "description": rule.description,
            "level": rule.level.value,
            "category": rule.category,
            "enabled": rule_name not in self.config["disabled_rules"],
            "category_enabled": rule.category in self.config["enabled_categories"]
        } 