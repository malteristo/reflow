"""
JSON Schema Validator for MCP tool parameters.

Provides comprehensive JSON schema validation with detailed error reporting
and field-specific validation messages.

Implements subtask 15.4: Implement Parameter Validation Logic.
"""

import logging
from typing import Dict, Any, List, Optional
import jsonschema
from jsonschema import Draft7Validator, validators

logger = logging.getLogger(__name__)


class JSONSchemaValidator:
    """
    Validates parameters against JSON Schema specifications.
    
    Provides comprehensive validation with detailed error messages
    and support for custom validation rules.
    """
    
    def __init__(self):
        """Initialize the JSON schema validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create validator class with custom error handling
        def extend_with_default(validator_class):
            """Extend validator to set default values and better error handling."""
            validate_properties = validator_class.VALIDATORS["properties"]
            
            def set_defaults(validator, properties, instance, schema):
                for property, subschema in properties.items():
                    if "default" in subschema:
                        instance.setdefault(property, subschema["default"])
                
                # Continue with normal validation
                for error in validate_properties(validator, properties, instance, schema):
                    yield error
            
            return validators.create(
                meta_schema=validator_class.META_SCHEMA,
                validators=dict(validator_class.VALIDATORS, properties=set_defaults),
            )
        
        self.ValidatorClass = extend_with_default(Draft7Validator)
    
    def validate(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against a JSON schema.
        
        Args:
            parameters: Parameters to validate
            schema: JSON schema to validate against
            
        Returns:
            Dict containing validation result:
            {
                "valid": bool,
                "errors": List[Dict] with error details,
                "sanitized_params": Dict with sanitized parameters
            }
        """
        try:
            # Create validator instance
            validator = self.ValidatorClass(schema)
            
            # Make a copy for sanitization
            sanitized_params = parameters.copy()
            
            # Collect validation errors
            errors = []
            
            # Validate the parameters
            validation_errors = list(validator.iter_errors(sanitized_params))
            
            for error in validation_errors:
                # Build a detailed error message
                error_info = {
                    "field": ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root",
                    "message": error.message,
                    "value": error.instance if hasattr(error, 'instance') else None,
                    "schema_path": ".".join(str(p) for p in error.schema_path) if error.schema_path else "",
                    "category": "schema_validation"
                }
                
                # Add more specific error context
                if error.validator == "required":
                    error_info["message"] = f"Required field '{error.validator_value}' is missing"
                elif error.validator == "type":
                    error_info["message"] = f"Field must be of type '{error.validator_value}', got '{type(error.instance).__name__}'"
                elif error.validator == "enum":
                    error_info["message"] = f"Value must be one of {error.validator_value}, got '{error.instance}'"
                elif error.validator == "minimum":
                    error_info["message"] = f"Value must be >= {error.validator_value}, got {error.instance}"
                elif error.validator == "maximum":
                    error_info["message"] = f"Value must be <= {error.validator_value}, got {error.instance}"
                elif error.validator == "minLength":
                    error_info["message"] = f"Value must be at least {error.validator_value} characters long"
                elif error.validator == "maxLength":
                    error_info["message"] = f"Value must be at most {error.validator_value} characters long"
                elif error.validator == "pattern":
                    error_info["message"] = f"Value does not match required pattern: {error.validator_value}"
                
                errors.append(error_info)
            
            # Apply defaults from schema if validation passed
            if not errors:
                self._apply_defaults(sanitized_params, schema)
            
            result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "sanitized_params": sanitized_params
            }
            
            self.logger.debug(f"Schema validation result: valid={result['valid']}, errors={len(errors)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [{
                    "field": "schema",
                    "message": f"Schema validation error: {str(e)}",
                    "category": "validation_error"
                }],
                "sanitized_params": parameters
            }
    
    def _apply_defaults(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Apply default values from schema to parameters.
        
        Args:
            parameters: Parameters to apply defaults to (modified in place)
            schema: Schema containing default values
        """
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if "default" in prop_schema and prop not in parameters:
                    parameters[prop] = prop_schema["default"]
                elif prop in parameters and "properties" in prop_schema:
                    # Recursively apply defaults for nested objects
                    if isinstance(parameters[prop], dict):
                        self._apply_defaults(parameters[prop], prop_schema)
    
    def validate_field(self, value: Any, field_schema: Dict[str, Any], field_name: str = "field") -> Dict[str, Any]:
        """
        Validate a single field against its schema.
        
        Args:
            value: Value to validate
            field_schema: Schema for the field
            field_name: Name of the field for error reporting
            
        Returns:
            Dict containing validation result for the field
        """
        schema = {
            "type": "object",
            "properties": {
                field_name: field_schema
            },
            "required": [field_name] if field_schema.get("required", False) else []
        }
        
        parameters = {field_name: value}
        result = self.validate(parameters, schema)
        
        # Adjust field names in errors
        for error in result["errors"]:
            if error["field"] == field_name:
                error["field"] = field_name
        
        return result
    
    def create_collection_schema(self) -> Dict[str, Any]:
        """
        Create a standard JSON schema for collection parameters.
        
        Returns:
            JSON schema for collection management parameters
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "delete", "info", "rename"],
                    "description": "Action to perform on collections"
                },
                "collection_name": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Name of the collection"
                },
                "new_name": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "New name for collection (rename action)"
                },
                "description": {
                    "type": "string",
                    "maxLength": 500,
                    "description": "Description for the collection"
                },
                "collection_type": {
                    "type": "string",
                    "enum": ["general", "project", "research"],
                    "default": "general",
                    "description": "Type of collection"
                }
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"const": "create"}}
                    },
                    "then": {
                        "required": ["collection_name"]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"const": "delete"}}
                    },
                    "then": {
                        "required": ["collection_name"]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"const": "info"}}
                    },
                    "then": {
                        "required": ["collection_name"]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"const": "rename"}}
                    },
                    "then": {
                        "required": ["collection_name", "new_name"]
                    }
                }
            ]
        }
    
    def create_query_schema(self) -> Dict[str, Any]:
        """
        Create a standard JSON schema for query parameters.
        
        Returns:
            JSON schema for query parameters
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10000,
                    "description": "Search query text"
                },
                "collections": {
                    "type": ["string", "null"],
                    "pattern": "^[a-zA-Z0-9_,-]+$",
                    "description": "Comma-separated collection names"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Number of results to return"
                },
                "document_context": {
                    "type": ["string", "null"],
                    "maxLength": 5000,
                    "description": "Current document context for enhanced search"
                }
            },
            "required": ["query"]
        }
    
    def create_documents_schema(self) -> Dict[str, Any]:
        """
        Create a standard JSON schema for document ingestion parameters.
        
        Returns:
            JSON schema for document parameters
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_document", "ingest_folder", "list_documents", "remove_document"],
                    "description": "Action to perform with documents"
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                    "description": "File or folder path"
                },
                "document_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Document ID for removal"
                },
                "collection": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Target collection name"
                },
                "recursive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Process folders recursively"
                },
                "file_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^\\.[a-zA-Z0-9]+$"
                    },
                    "description": "File extensions to include"
                }
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"enum": ["add_document", "ingest_folder"]}}
                    },
                    "then": {
                        "required": ["path", "collection"]
                    }
                },
                {
                    "if": {
                        "properties": {"action": {"const": "remove_document"}}
                    },
                    "then": {
                        "required": ["document_id"]
                    }
                }
            ]
        } 