"""
Business Logic Validator for MCP tool parameters.

Provides domain-specific validation rules for Research Agent
operations including file types, query validation, and business constraints.

Implements subtask 15.4: Implement Parameter Validation Logic.
"""

import logging
from typing import Dict, Any, List, Set, Optional
import re

logger = logging.getLogger(__name__)


class BusinessValidator:
    """
    Validates parameters against business logic and domain rules.
    
    Provides validation for:
    - File types and formats
    - Query constraints
    - Collection business rules
    - Parameter ranges and limits
    """
    
    def __init__(self):
        """Initialize the business validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Supported file types for document ingestion
        self.supported_document_types = {
            '.md', '.txt', '.pdf', '.doc', '.docx', '.rtf',
            '.json', '.yaml', '.yml', '.xml', '.csv'
        }
        
        self.supported_code_types = {
            '.py', '.js', '.ts', '.html', '.css', '.java',
            '.cpp', '.c', '.go', '.rs', '.php', '.rb'
        }
        
        # Collection types
        self.valid_collection_types = {
            'general', 'project', 'research', 'documentation', 'code'
        }
        
        # Query constraints
        self.min_query_length = 2
        self.max_query_length = 10000
        self.min_top_k = 1
        self.max_top_k = 100
        
        # Collection name constraints
        self.min_collection_name_length = 1
        self.max_collection_name_length = 100
        
        # Project constraints
        self.min_project_name_length = 1
        self.max_project_name_length = 100
        
        # Document size limits (in characters for text content)
        self.max_document_content_length = 1000000  # 1M characters
        
        # Reserved keywords that shouldn't be used as names
        self.reserved_keywords = {
            'admin', 'root', 'system', 'config', 'default',
            'null', 'undefined', 'none', 'empty', 'temp',
            'test', 'debug', 'api', 'auth', 'user'
        }
    
    def validate_file_extension(self, extension: str) -> Dict[str, Any]:
        """
        Validate file extension against supported types.
        
        Args:
            extension: File extension to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        
        if not extension or not isinstance(extension, str):
            errors.append("File extension must be a non-empty string")
            return {
                "valid": False,
                "errors": errors,
                "file_type": None
            }
        
        # Normalize extension
        normalized_ext = extension.lower()
        if not normalized_ext.startswith('.'):
            normalized_ext = f'.{normalized_ext}'
        
        # Determine file type category
        file_type = None
        if normalized_ext in self.supported_document_types:
            file_type = "document"
        elif normalized_ext in self.supported_code_types:
            file_type = "code"
        else:
            errors.append(f"Unsupported file extension '{normalized_ext}'. "
                         f"Supported types: {sorted(self.supported_document_types | self.supported_code_types)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "file_type": file_type,
            "normalized_extension": normalized_ext if file_type else ""
        }
    
    def validate_collection_type(self, collection_type: str) -> Dict[str, Any]:
        """
        Validate collection type against valid options.
        
        Args:
            collection_type: Collection type to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        
        if not collection_type or not isinstance(collection_type, str):
            errors.append("Collection type must be a non-empty string")
            return {
                "valid": False,
                "errors": errors,
                "normalized_type": ""
            }
        
        normalized_type = collection_type.lower().strip()
        
        if normalized_type not in self.valid_collection_types:
            errors.append(f"Invalid collection type '{collection_type}'. "
                         f"Valid types: {sorted(self.valid_collection_types)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "normalized_type": normalized_type if errors == [] else ""
        }
    
    def validate_query_content(self, query: str) -> Dict[str, Any]:
        """
        Validate query content for business requirements.
        
        Args:
            query: Query content to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        warnings = []
        
        if not query or not isinstance(query, str):
            errors.append("Query must be a non-empty string")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "sanitized_query": ""
            }
        
        # Check length constraints
        if len(query) < self.min_query_length:
            errors.append(f"Query too short (minimum {self.min_query_length} characters)")
        
        if len(query) > self.max_query_length:
            errors.append(f"Query too long (maximum {self.max_query_length} characters)")
        
        # Check for meaningful content
        stripped_query = query.strip()
        if not stripped_query:
            errors.append("Query cannot be empty or only whitespace")
        
        # Check for very short queries that might not be meaningful
        if len(stripped_query) < 3:
            warnings.append("Very short queries may not return meaningful results")
        
        # Check for repeated characters (likely not a real query)
        if len(set(stripped_query.replace(' ', ''))) < 2:
            warnings.append("Query appears to contain mostly repeated characters")
        
        # Check for excessive punctuation
        punctuation_ratio = sum(1 for c in stripped_query if not c.isalnum() and c != ' ') / len(stripped_query)
        if punctuation_ratio > 0.5:
            warnings.append("Query contains excessive punctuation which may affect results")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "sanitized_query": stripped_query if errors == [] else ""
        }
    
    def validate_top_k_parameter(self, top_k: Any) -> Dict[str, Any]:
        """
        Validate top_k parameter for query operations.
        
        Args:
            top_k: Top K value to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        
        # Check type
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                errors.append("top_k must be an integer")
                return {
                    "valid": False,
                    "errors": errors,
                    "normalized_value": self.min_top_k
                }
        
        # Check range
        if top_k < self.min_top_k:
            errors.append(f"top_k must be at least {self.min_top_k}")
        
        if top_k > self.max_top_k:
            errors.append(f"top_k cannot exceed {self.max_top_k}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "normalized_value": max(self.min_top_k, min(top_k, self.max_top_k))
        }
    
    def validate_collection_name_format(self, name: str) -> Dict[str, Any]:
        """
        Validate collection name format and business rules.
        
        Args:
            name: Collection name to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        warnings = []
        
        if not name or not isinstance(name, str):
            errors.append("Collection name must be a non-empty string")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "normalized_name": ""
            }
        
        # Check length
        if len(name) < self.min_collection_name_length:
            errors.append(f"Collection name too short (minimum {self.min_collection_name_length} characters)")
        
        if len(name) > self.max_collection_name_length:
            errors.append(f"Collection name too long (maximum {self.max_collection_name_length} characters)")
        
        # Check format (alphanumeric, underscores, hyphens)
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            errors.append("Collection name can only contain letters, numbers, underscores, and hyphens")
        
        # Check for reserved keywords
        if name.lower() in self.reserved_keywords:
            errors.append(f"'{name}' is a reserved keyword and cannot be used as a collection name")
        
        # Check for good naming practices
        if name.startswith('_') or name.startswith('-'):
            warnings.append("Collection names should not start with underscore or hyphen")
        
        if name.endswith('_') or name.endswith('-'):
            warnings.append("Collection names should not end with underscore or hyphen")
        
        if '--' in name or '__' in name:
            warnings.append("Avoid multiple consecutive underscores or hyphens in collection names")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "normalized_name": name.strip() if errors == [] else ""
        }
    
    def validate_project_name_format(self, name: str) -> Dict[str, Any]:
        """
        Validate project name format and business rules.
        
        Args:
            name: Project name to validate
            
        Returns:
            Dict containing validation result
        """
        errors = []
        warnings = []
        
        if not name or not isinstance(name, str):
            errors.append("Project name must be a non-empty string")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "normalized_name": ""
            }
        
        # Check length
        if len(name) < self.min_project_name_length:
            errors.append(f"Project name too short (minimum {self.min_project_name_length} characters)")
        
        if len(name) > self.max_project_name_length:
            errors.append(f"Project name too long (maximum {self.max_project_name_length} characters)")
        
        # Check format
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            errors.append("Project name can only contain letters, numbers, underscores, and hyphens")
        
        # Check for reserved keywords
        if name.lower() in self.reserved_keywords:
            errors.append(f"'{name}' is a reserved keyword and cannot be used as a project name")
        
        # Business rule: Project names should start with a letter
        if not name[0].isalpha():
            warnings.append("Project names should start with a letter for better readability")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "normalized_name": name.strip() if errors == [] else ""
        }
    
    def validate_document_content_length(self, content_length: int) -> Dict[str, Any]:
        """
        Validate document content length against business limits.
        
        Args:
            content_length: Length of document content in characters
            
        Returns:
            Dict containing validation result
        """
        errors = []
        warnings = []
        
        if not isinstance(content_length, int) or content_length < 0:
            errors.append("Content length must be a non-negative integer")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings
            }
        
        # Check maximum length
        if content_length > self.max_document_content_length:
            errors.append(f"Document content too large ({content_length} characters). "
                         f"Maximum allowed: {self.max_document_content_length} characters")
        
        # Warning for very large documents
        if content_length > self.max_document_content_length * 0.8:
            warnings.append("Document is very large and may take longer to process")
        
        # Warning for very small documents
        if content_length < 100:
            warnings.append("Document is very small and may not provide meaningful search results")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def validate_collections_parameter(self, collections: str) -> Dict[str, Any]:
        """
        Validate collections parameter format.
        
        Args:
            collections: Comma-separated collection names
            
        Returns:
            Dict containing validation result
        """
        errors = []
        warnings = []
        parsed_collections = []
        
        if collections is None:
            return {
                "valid": True,
                "errors": errors,
                "warnings": warnings,
                "parsed_collections": []
            }
        
        if not isinstance(collections, str):
            errors.append("Collections parameter must be a string or null")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "parsed_collections": []
            }
        
        # Parse comma-separated names
        collection_names = [name.strip() for name in collections.split(',') if name.strip()]
        
        if not collection_names:
            warnings.append("Collections parameter is empty")
            return {
                "valid": True,
                "errors": errors,
                "warnings": warnings,
                "parsed_collections": []
            }
        
        # Validate each collection name
        for name in collection_names:
            validation = self.validate_collection_name_format(name)
            if not validation["valid"]:
                errors.extend([f"Collection '{name}': {error}" for error in validation["errors"]])
            else:
                parsed_collections.append(validation["normalized_name"])
            
            # Add warnings with collection context
            if validation["warnings"]:
                warnings.extend([f"Collection '{name}': {warning}" for warning in validation["warnings"]])
        
        # Check for duplicates
        if len(parsed_collections) != len(set(parsed_collections)):
            warnings.append("Duplicate collection names detected")
            parsed_collections = list(set(parsed_collections))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "parsed_collections": parsed_collections
        }
    
    def get_business_rules(self) -> Dict[str, Any]:
        """
        Get current business validation rules.
        
        Returns:
            Dict containing business rule configuration
        """
        return {
            "supported_document_types": list(self.supported_document_types),
            "supported_code_types": list(self.supported_code_types),
            "valid_collection_types": list(self.valid_collection_types),
            "query_length_limits": {
                "min": self.min_query_length,
                "max": self.max_query_length
            },
            "top_k_limits": {
                "min": self.min_top_k,
                "max": self.max_top_k
            },
            "collection_name_limits": {
                "min": self.min_collection_name_length,
                "max": self.max_collection_name_length
            },
            "project_name_limits": {
                "min": self.min_project_name_length,
                "max": self.max_project_name_length
            },
            "max_document_content_length": self.max_document_content_length,
            "reserved_keywords": list(self.reserved_keywords)
        } 