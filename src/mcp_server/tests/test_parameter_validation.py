"""
Test suite for Parameter Validation Logic.

Tests comprehensive parameter validation for all MCP tools
including security validation, JSON schema validation, and business logic validation.

Implements TDD for subtask 15.4: Implement Parameter Validation Logic.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock

# These imports will be enhanced during implementation
try:
    from src.mcp_server.tools.base_tool import BaseMCPTool, ToolValidationError
    from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
    from src.mcp_server.validation.security_validator import SecurityValidator
    from src.mcp_server.validation.business_validator import BusinessValidator
    from src.mcp_server.validation.validation_registry import ValidationRegistry
except ImportError:
    # Expected during RED phase
    pass


class TestJSONSchemaValidation:
    """Test JSON schema-based parameter validation."""
    
    def test_json_schema_validator_exists(self):
        """Test that JSONSchemaValidator class exists."""
        # This will fail initially - RED phase
        from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
        assert JSONSchemaValidator is not None
    
    def test_json_schema_validator_initialization(self):
        """Test JSONSchemaValidator can be initialized."""
        from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
        validator = JSONSchemaValidator()
        assert validator is not None
    
    def test_validate_against_schema_success(self):
        """Test successful validation against JSON schema."""
        from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
        
        validator = JSONSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
        
        parameters = {"query": "test query", "top_k": 10}
        result = validator.validate(parameters, schema)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_against_schema_failure(self):
        """Test validation failure against JSON schema."""
        from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
        
        validator = JSONSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
        
        # Missing required field
        parameters = {"top_k": 10}
        result = validator.validate(parameters, schema)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("query" in error["message"] for error in result["errors"])
    
    def test_validate_nested_schema(self):
        """Test validation of nested JSON schema."""
        from src.mcp_server.validation.json_schema_validator import JSONSchemaValidator
        
        validator = JSONSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "delete"]},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "required": ["action", "metadata"]
        }
        
        parameters = {
            "action": "create",
            "metadata": {
                "name": "test collection",
                "description": "test description"
            }
        }
        result = validator.validate(parameters, schema)
        assert result["valid"] is True


class TestSecurityValidation:
    """Test security-focused parameter validation."""
    
    def test_security_validator_exists(self):
        """Test that SecurityValidator class exists."""
        from src.mcp_server.validation.security_validator import SecurityValidator
        assert SecurityValidator is not None
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts."""
        from src.mcp_server.validation.security_validator import SecurityValidator
        
        validator = SecurityValidator()
        
        # Valid paths
        assert validator.validate_file_path("document.md")["valid"] is True
        assert validator.validate_file_path("folder/document.md")["valid"] is True
        
        # Path traversal attempts
        assert validator.validate_file_path("../../../etc/passwd")["valid"] is False
        assert validator.validate_file_path("..\\..\\windows\\system32")["valid"] is False
        assert validator.validate_file_path("/etc/passwd")["valid"] is False
    
    def test_input_sanitization(self):
        """Test input sanitization for SQL injection and XSS prevention."""
        from src.mcp_server.validation.security_validator import SecurityValidator
        
        validator = SecurityValidator()
        
        # Safe inputs
        assert validator.sanitize_text_input("normal query text")["safe"] is True
        assert validator.sanitize_text_input("search for python")["safe"] is True
        
        # Potentially dangerous inputs
        assert validator.sanitize_text_input("'; DROP TABLE users; --")["safe"] is False
        assert validator.sanitize_text_input("<script>alert('xss')</script>")["safe"] is False
        assert validator.sanitize_text_input("UNION SELECT * FROM passwords")["safe"] is False
    
    def test_collection_name_validation(self):
        """Test validation of collection names for security and format."""
        from src.mcp_server.validation.security_validator import SecurityValidator
        
        validator = SecurityValidator()
        
        # Valid collection names
        assert validator.validate_collection_name("my_collection")["valid"] is True
        assert validator.validate_collection_name("project-docs")["valid"] is True
        assert validator.validate_collection_name("collection123")["valid"] is True
        
        # Invalid collection names
        assert validator.validate_collection_name("../admin")["valid"] is False
        assert validator.validate_collection_name("col name")["valid"] is False  # spaces
        assert validator.validate_collection_name("very_long_collection_name_that_exceeds_maximum_length_limit")["valid"] is False
        assert validator.validate_collection_name("")["valid"] is False  # empty
    
    def test_project_name_validation(self):
        """Test validation of project names."""
        from src.mcp_server.validation.security_validator import SecurityValidator
        
        validator = SecurityValidator()
        
        # Valid project names
        assert validator.validate_project_name("my_project")["valid"] is True
        assert validator.validate_project_name("research-2024")["valid"] is True
        
        # Invalid project names
        assert validator.validate_project_name("../admin")["valid"] is False
        assert validator.validate_project_name("project with spaces")["valid"] is False
        assert validator.validate_project_name("")["valid"] is False


class TestBusinessValidation:
    """Test business logic parameter validation."""
    
    def test_business_validator_exists(self):
        """Test that BusinessValidator class exists."""
        from src.mcp_server.validation.business_validator import BusinessValidator
        assert BusinessValidator is not None
    
    def test_file_type_validation(self):
        """Test validation of supported file types."""
        from src.mcp_server.validation.business_validator import BusinessValidator
        
        validator = BusinessValidator()
        
        # Supported file types
        assert validator.validate_file_extension(".md")["valid"] is True
        assert validator.validate_file_extension(".txt")["valid"] is True
        assert validator.validate_file_extension(".pdf")["valid"] is True
        
        # Unsupported file types
        assert validator.validate_file_extension(".exe")["valid"] is False
        assert validator.validate_file_extension(".bat")["valid"] is False
        assert validator.validate_file_extension(".sh")["valid"] is False
    
    def test_collection_type_validation(self):
        """Test validation of collection types."""
        from src.mcp_server.validation.business_validator import BusinessValidator
        
        validator = BusinessValidator()
        
        # Valid collection types
        assert validator.validate_collection_type("general")["valid"] is True
        assert validator.validate_collection_type("project")["valid"] is True
        assert validator.validate_collection_type("research")["valid"] is True
        
        # Invalid collection types
        assert validator.validate_collection_type("invalid")["valid"] is False
        assert validator.validate_collection_type("")["valid"] is False
    
    def test_query_length_validation(self):
        """Test validation of query length limits."""
        from src.mcp_server.validation.business_validator import BusinessValidator
        
        validator = BusinessValidator()
        
        # Valid query lengths
        assert validator.validate_query_content("short query")["valid"] is True
        assert validator.validate_query_content("a" * 100)["valid"] is True  # 100 chars
        
        # Invalid query lengths
        assert validator.validate_query_content("")["valid"] is False  # empty
        assert validator.validate_query_content("a" * 15000)["valid"] is False  # too long
    
    def test_top_k_parameter_validation(self):
        """Test validation of top_k parameter ranges."""
        from src.mcp_server.validation.business_validator import BusinessValidator
        
        validator = BusinessValidator()
        
        # Valid top_k values
        assert validator.validate_top_k_parameter(1)["valid"] is True
        assert validator.validate_top_k_parameter(10)["valid"] is True
        assert validator.validate_top_k_parameter(100)["valid"] is True
        
        # Invalid top_k values
        assert validator.validate_top_k_parameter(0)["valid"] is False
        assert validator.validate_top_k_parameter(-1)["valid"] is False
        assert validator.validate_top_k_parameter(1000)["valid"] is False


class TestValidationRegistry:
    """Test centralized validation rule registry."""
    
    def test_validation_registry_exists(self):
        """Test that ValidationRegistry class exists."""
        from src.mcp_server.validation.validation_registry import ValidationRegistry
        assert ValidationRegistry is not None
    
    def test_registry_initialization(self):
        """Test ValidationRegistry initialization."""
        from src.mcp_server.validation.validation_registry import ValidationRegistry
        
        registry = ValidationRegistry()
        assert registry is not None
        assert hasattr(registry, 'register_rule')
        assert hasattr(registry, 'validate_value')
    
    def test_register_validation_rule(self):
        """Test registering custom validation rules."""
        from src.mcp_server.validation.validation_registry import ValidationRegistry, ValidationRule
        
        registry = ValidationRegistry()
        
        # Register a custom rule
        def validate_email(value):
            has_at = "@" in value
            has_dot = "." in value
            return {
                "valid": has_at and has_dot, 
                "errors": [] if (has_at and has_dot) else ["Invalid email format"]
            }
        
        rule = ValidationRule("email", validate_email, description="Email validation")
        registry.register_rule(rule)
        
        # Debug: Check what rules are available
        print(f"DEBUG: Available rules: {list(registry.rules.keys())}")
        
        # Test the registered rule
        result = registry.validate_value("test@example.com", ["email"])
        print(f"DEBUG: Result for 'test@example.com': {result}")
        assert result["valid"] is True
        
        result = registry.validate_value("notanemail", ["email"])  # Changed to truly invalid email
        print(f"DEBUG: Result for 'notanemail': {result}")  # Debug print
        assert result["valid"] is False
    
    def test_composite_validation(self):
        """Test combining multiple validation rules."""
        from src.mcp_server.validation.validation_registry import ValidationRegistry
        
        registry = ValidationRegistry()
        
        # Test multiple validations using existing rules
        result = registry.validate_by_category("test@example.com", ["format"])
        
        assert "valid" in result
        assert "errors" in result


class TestEnhancedBaseMCPTool:
    """Test enhanced parameter validation in BaseMCPTool."""
    
    def test_enhanced_validation_integration(self):
        """Test that enhanced validation is integrated into BaseMCPTool."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        
        # Create a test tool implementation
        class TestTool(BaseMCPTool):
            def get_tool_name(self):
                return "test_tool"
            def get_tool_description(self):
                return "Test tool"
            def execute(self, parameters):
                return {"status": "success"}
        
        tool = TestTool()
        
        # Test that enhanced validation methods exist
        assert hasattr(tool, 'validate_parameters_enhanced')
        assert hasattr(tool, 'get_validation_config')
    
    def test_comprehensive_parameter_validation(self):
        """Test comprehensive parameter validation flow."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        
        tool = QueryKnowledgeBaseTool()
        
        # Valid parameters
        valid_params = {
            "query": "test query",
            "collections": "valid_collection",
            "top_k": 10
        }
        
        result = tool.validate_parameters_enhanced(valid_params)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validation_error_aggregation(self):
        """Test that validation errors are properly aggregated."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        
        tool = QueryKnowledgeBaseTool()
        
        # Invalid parameters with multiple issues
        invalid_params = {
            "query": "",  # empty query
            "collections": "../admin",  # path traversal
            "top_k": 1000  # out of range
        }
        
        result = tool.validate_parameters_enhanced(invalid_params)
        assert result["valid"] is False
        assert len(result["errors"]) >= 3  # Should have multiple validation errors
        
        # Check that errors are categorized
        error_types = [error["category"] for error in result["errors"]]
        assert "business_logic" in error_types
        assert "security" in error_types 