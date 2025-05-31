"""
Project-specific exceptions for Research Agent.

This module defines exceptions related to project management operations,
including project-collection linking, metadata storage, and context detection.

Implements FR-KB-005: Project and collection management error handling.
"""

from .vector_store_exceptions import VectorStoreError


class ProjectError(Exception):
    """Base exception for project-related operations."""
    
    def __init__(self, message: str, project_name: str = None):
        super().__init__(message)
        self.project_name = project_name


class ProjectNotFoundError(ProjectError):
    """Raised when a project cannot be found."""
    
    def __init__(self, project_name: str):
        message = f"Project '{project_name}' not found"
        super().__init__(message, project_name)


class ProjectAlreadyExistsError(ProjectError):
    """Raised when attempting to create a project that already exists."""
    
    def __init__(self, project_name: str):
        message = f"Project '{project_name}' already exists"
        super().__init__(message, project_name)


class CollectionAlreadyLinkedError(ProjectError):
    """Raised when attempting to link a collection that's already linked to a project."""
    
    def __init__(self, collection_name: str, project_name: str):
        message = f"Collection '{collection_name}' is already linked to project '{project_name}'"
        super().__init__(message, project_name)
        self.collection_name = collection_name


class CollectionNotLinkedError(ProjectError):
    """Raised when attempting to unlink a collection that's not linked to a project."""
    
    def __init__(self, collection_name: str, project_name: str):
        message = f"Collection '{collection_name}' is not linked to project '{project_name}'"
        super().__init__(message, project_name)
        self.collection_name = collection_name


class ProjectContextError(ProjectError):
    """Raised when project context detection or setting fails."""
    
    def __init__(self, message: str, context_path: str = None):
        super().__init__(message)
        self.context_path = context_path


class ProjectMetadataError(ProjectError):
    """Raised when project metadata operations fail."""
    
    def __init__(self, message: str, project_name: str = None, operation: str = None):
        super().__init__(message, project_name)
        self.operation = operation 