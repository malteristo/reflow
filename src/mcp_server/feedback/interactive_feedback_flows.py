"""
Interactive Feedback Flows for MCP Server

This module provides interactive feedback patterns for multi-step operations
including collection management, document conflict resolution, query refinement,
and error recovery workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import time
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of interactive feedback flows."""
    COLLECTION_MANAGEMENT = "collection_management"
    DOCUMENT_CONFLICT_RESOLUTION = "document_conflict_resolution"
    QUERY_REFINEMENT = "query_refinement"
    ERROR_RECOVERY = "error_recovery"


class InteractionState(Enum):
    """States of interactive workflow sessions."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    WAITING_USER_INPUT = "waiting_user_input"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class UserInteraction:
    """Represents a user interaction in a workflow."""
    interaction_id: str
    interaction_type: InteractionType
    state: InteractionState
    prompt: str
    options: List[str] = field(default_factory=list)
    user_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSession:
    """Manages an interactive workflow session."""
    session_id: str
    interaction_type: InteractionType
    state: InteractionState
    interactions: List[UserInteraction] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class InteractiveFeedbackFlows:
    """Manages interactive feedback workflows for multi-step operations."""
    
    def __init__(self):
        """Initialize the interactive feedback flows manager."""
        self.active_sessions: Dict[str, WorkflowSession] = {}
        self.session_history: List[WorkflowSession] = []
        self.interaction_handlers: Dict[InteractionType, Callable] = {
            InteractionType.COLLECTION_MANAGEMENT: self._handle_collection_management,
            InteractionType.DOCUMENT_CONFLICT_RESOLUTION: self._handle_document_conflict,
            InteractionType.QUERY_REFINEMENT: self._handle_query_refinement,
            InteractionType.ERROR_RECOVERY: self._handle_error_recovery
        }
        logger.info("Interactive feedback flows manager initialized")
    
    def start_collection_management_flow(self, context: Dict[str, Any]) -> WorkflowSession:
        """Start an interactive collection management workflow."""
        session_id = str(uuid4())
        session = WorkflowSession(
            session_id=session_id,
            interaction_type=InteractionType.COLLECTION_MANAGEMENT,
            state=InteractionState.INITIATED,
            context=context
        )
        
        # Create initial interaction
        initial_interaction = UserInteraction(
            interaction_id=str(uuid4()),
            interaction_type=InteractionType.COLLECTION_MANAGEMENT,
            state=InteractionState.WAITING_USER_INPUT,
            prompt="Select collection management action:",
            options=[
                "Create new collection",
                "Select existing collection", 
                "Modify collection settings",
                "Cancel operation"
            ],
            context=context
        )
        
        session.interactions.append(initial_interaction)
        session.state = InteractionState.IN_PROGRESS
        self.active_sessions[session_id] = session
        
        logger.info(f"Started collection management flow: {session_id}")
        return session
    
    def start_document_conflict_resolution(self, context: Dict[str, Any]) -> WorkflowSession:
        """Start an interactive document conflict resolution workflow."""
        session_id = str(uuid4())
        session = WorkflowSession(
            session_id=session_id,
            interaction_type=InteractionType.DOCUMENT_CONFLICT_RESOLUTION,
            state=InteractionState.INITIATED,
            context=context
        )
        
        # Create conflict resolution interaction
        conflict_interaction = UserInteraction(
            interaction_id=str(uuid4()),
            interaction_type=InteractionType.DOCUMENT_CONFLICT_RESOLUTION,
            state=InteractionState.WAITING_USER_INPUT,
            prompt="Document conflict detected. Choose resolution strategy:",
            options=[
                "Keep existing document",
                "Replace with new document",
                "Merge documents",
                "Keep both with different names",
                "Cancel import"
            ],
            context=context,
            metadata={
                "conflict_type": context.get("conflict_type", "duplicate"),
                "existing_document": context.get("existing_document"),
                "new_document": context.get("new_document")
            }
        )
        
        session.interactions.append(conflict_interaction)
        session.state = InteractionState.IN_PROGRESS
        self.active_sessions[session_id] = session
        
        logger.info(f"Started document conflict resolution: {session_id}")
        return session
    
    def start_query_refinement_flow(self, context: Dict[str, Any]) -> WorkflowSession:
        """Start an interactive query refinement workflow."""
        session_id = str(uuid4())
        session = WorkflowSession(
            session_id=session_id,
            interaction_type=InteractionType.QUERY_REFINEMENT,
            state=InteractionState.INITIATED,
            context=context
        )
        
        # Create query refinement interaction
        refinement_interaction = UserInteraction(
            interaction_id=str(uuid4()),
            interaction_type=InteractionType.QUERY_REFINEMENT,
            state=InteractionState.WAITING_USER_INPUT,
            prompt="Query results may be improved. Choose refinement option:",
            options=[
                "Make query more specific",
                "Broaden search scope",
                "Add related terms",
                "Change search collections",
                "Use original query"
            ],
            context=context,
            metadata={
                "original_query": context.get("query"),
                "result_count": context.get("result_count", 0),
                "suggestions": context.get("suggestions", [])
            }
        )
        
        session.interactions.append(refinement_interaction)
        session.state = InteractionState.IN_PROGRESS
        self.active_sessions[session_id] = session
        
        logger.info(f"Started query refinement flow: {session_id}")
        return session
    
    def start_error_recovery_flow(self, context: Dict[str, Any]) -> WorkflowSession:
        """Start an interactive error recovery workflow."""
        session_id = str(uuid4())
        session = WorkflowSession(
            session_id=session_id,
            interaction_type=InteractionType.ERROR_RECOVERY,
            state=InteractionState.INITIATED,
            context=context
        )
        
        # Create error recovery interaction
        recovery_interaction = UserInteraction(
            interaction_id=str(uuid4()),
            interaction_type=InteractionType.ERROR_RECOVERY,
            state=InteractionState.WAITING_USER_INPUT,
            prompt="Error occurred. Choose recovery action:",
            options=[
                "Retry operation",
                "Modify parameters and retry", 
                "Skip problematic item",
                "Cancel operation",
                "Get detailed error information"
            ],
            context=context,
            metadata={
                "error_type": context.get("error_type"),
                "error_message": context.get("error_message"),
                "recoverable": context.get("recoverable", True)
            }
        )
        
        session.interactions.append(recovery_interaction)
        session.state = InteractionState.IN_PROGRESS
        self.active_sessions[session_id] = session
        
        logger.info(f"Started error recovery flow: {session_id}")
        return session
    
    def process_user_response(self, session_id: str, response: str) -> Optional[UserInteraction]:
        """Process user response for an active session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        session = self.active_sessions[session_id]
        
        # Find the current waiting interaction
        current_interaction = None
        for interaction in reversed(session.interactions):
            if interaction.state == InteractionState.WAITING_USER_INPUT:
                current_interaction = interaction
                break
        
        if not current_interaction:
            logger.warning(f"No waiting interaction found for session: {session_id}")
            return None
        
        # Update interaction with user response
        current_interaction.user_response = response
        current_interaction.state = InteractionState.COMPLETED
        session.updated_at = time.time()
        
        # Process the response based on interaction type
        handler = self.interaction_handlers.get(session.interaction_type)
        if handler:
            next_interaction = handler(session, current_interaction, response)
            if next_interaction:
                session.interactions.append(next_interaction)
            else:
                # Workflow completed
                session.state = InteractionState.COMPLETED
                self._complete_session(session_id)
        
        logger.info(f"Processed user response for session: {session_id}")
        return current_interaction
    
    def get_session(self, session_id: str) -> Optional[WorkflowSession]:
        """Get a workflow session by ID."""
        return self.active_sessions.get(session_id) or next(
            (s for s in self.session_history if s.session_id == session_id), None
        )
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active workflow session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.state = InteractionState.CANCELLED
            session.updated_at = time.time()
            self._complete_session(session_id)
            logger.info(f"Cancelled session: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[WorkflowSession]:
        """Get all active workflow sessions."""
        return list(self.active_sessions.values())
    
    def _handle_collection_management(self, session: WorkflowSession, 
                                    interaction: UserInteraction, 
                                    response: str) -> Optional[UserInteraction]:
        """Handle collection management workflow responses."""
        if response == "Create new collection":
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.COLLECTION_MANAGEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Enter name for new collection:",
                options=[],
                context=session.context,
                metadata={"action": "create", "step": "naming"}
            )
        elif response == "Select existing collection":
            available_collections = session.context.get("available_collections", [])
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.COLLECTION_MANAGEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Select collection:",
                options=available_collections + ["Cancel"],
                context=session.context,
                metadata={"action": "select"}
            )
        elif response == "Cancel operation":
            session.state = InteractionState.CANCELLED
            return None
        
        # Handle naming step for new collection
        if interaction.metadata.get("action") == "create" and interaction.metadata.get("step") == "naming":
            session.context["new_collection_name"] = response
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.COLLECTION_MANAGEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Select collection type:",
                options=["general", "project", "research", "Cancel"],
                context=session.context,
                metadata={"action": "create", "step": "type_selection"}
            )
        
        # Handle type selection for new collection
        if (interaction.metadata.get("action") == "create" and 
            interaction.metadata.get("step") == "type_selection" and 
            response != "Cancel"):
            session.context["collection_type"] = response
            session.context["collection_created"] = True
            return None  # Workflow complete
        
        return None
    
    def _handle_document_conflict(self, session: WorkflowSession,
                                interaction: UserInteraction,
                                response: str) -> Optional[UserInteraction]:
        """Handle document conflict resolution workflow responses."""
        if response == "Keep existing document":
            session.context["resolution"] = "keep_existing"
            return None
        elif response == "Replace with new document":
            session.context["resolution"] = "replace"
            return None
        elif response == "Merge documents":
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.DOCUMENT_CONFLICT_RESOLUTION,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Select merge strategy:",
                options=[
                    "Combine content sections",
                    "Append new content",
                    "Manual merge",
                    "Cancel merge"
                ],
                context=session.context,
                metadata={"action": "merge"}
            )
        elif response == "Keep both with different names":
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.DOCUMENT_CONFLICT_RESOLUTION,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Enter suffix for new document name:",
                options=[],
                context=session.context,
                metadata={"action": "rename"}
            )
        elif response == "Cancel import":
            session.context["resolution"] = "cancel"
            return None
        
        # Handle merge strategy selection
        if interaction.metadata.get("action") == "merge" and response != "Cancel merge":
            session.context["merge_strategy"] = response
            session.context["resolution"] = "merge"
            return None
        
        # Handle rename suffix input
        if interaction.metadata.get("action") == "rename":
            session.context["rename_suffix"] = response
            session.context["resolution"] = "rename"
            return None
        
        return None
    
    def _handle_query_refinement(self, session: WorkflowSession,
                               interaction: UserInteraction,
                               response: str) -> Optional[UserInteraction]:
        """Handle query refinement workflow responses."""
        if response == "Make query more specific":
            suggestions = session.context.get("specific_suggestions", [
                "Add technical terms",
                "Specify time period",
                "Include specific domain"
            ])
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.QUERY_REFINEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="How would you like to make the query more specific?",
                options=suggestions + ["Custom modification"],
                context=session.context,
                metadata={"refinement_type": "specific"}
            )
        elif response == "Broaden search scope":
            session.context["refinement"] = "broaden"
            session.context["refined_query"] = session.context.get("query", "") + " OR related topics"
            return None
        elif response == "Add related terms":
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.QUERY_REFINEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Enter additional search terms (comma-separated):",
                options=[],
                context=session.context,
                metadata={"refinement_type": "add_terms"}
            )
        elif response == "Change search collections":
            available_collections = session.context.get("available_collections", [])
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.QUERY_REFINEMENT,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Select collections to search:",
                options=available_collections,
                context=session.context,
                metadata={"refinement_type": "collections"}
            )
        elif response == "Use original query":
            session.context["refinement"] = "none"
            return None
        
        # Handle specific refinement options
        if interaction.metadata.get("refinement_type") == "specific":
            session.context["specific_refinement"] = response
            session.context["refinement"] = "specific"
            return None
        elif interaction.metadata.get("refinement_type") == "add_terms":
            additional_terms = response.split(",")
            session.context["additional_terms"] = [term.strip() for term in additional_terms]
            session.context["refinement"] = "add_terms"
            return None
        elif interaction.metadata.get("refinement_type") == "collections":
            session.context["selected_collections"] = response.split(",")
            session.context["refinement"] = "collections"
            return None
        
        return None
    
    def _handle_error_recovery(self, session: WorkflowSession,
                             interaction: UserInteraction,
                             response: str) -> Optional[UserInteraction]:
        """Handle error recovery workflow responses."""
        if response == "Retry operation":
            session.context["recovery_action"] = "retry"
            return None
        elif response == "Modify parameters and retry":
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.ERROR_RECOVERY,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Which parameters would you like to modify?",
                options=[
                    "Timeout settings",
                    "Collection selection",
                    "File format options",
                    "Custom parameters"
                ],
                context=session.context,
                metadata={"action": "modify_params"}
            )
        elif response == "Skip problematic item":
            session.context["recovery_action"] = "skip"
            return None
        elif response == "Cancel operation":
            session.context["recovery_action"] = "cancel"
            session.state = InteractionState.CANCELLED
            return None
        elif response == "Get detailed error information":
            error_details = session.context.get("error_details", "No additional details available")
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.ERROR_RECOVERY,
                state=InteractionState.WAITING_USER_INPUT,
                prompt=f"Error details: {error_details}\n\nChoose next action:",
                options=[
                    "Retry operation",
                    "Modify parameters and retry",
                    "Skip problematic item",
                    "Cancel operation"
                ],
                context=session.context,
                metadata={"action": "after_details"}
            )
        
        # Handle parameter modification
        if interaction.metadata.get("action") == "modify_params":
            session.context["parameter_modification"] = response
            return UserInteraction(
                interaction_id=str(uuid4()),
                interaction_type=InteractionType.ERROR_RECOVERY,
                state=InteractionState.WAITING_USER_INPUT,
                prompt="Enter new parameter value:",
                options=[],
                context=session.context,
                metadata={"action": "param_value", "param_type": response}
            )
        elif interaction.metadata.get("action") == "param_value":
            param_type = interaction.metadata.get("param_type")
            session.context["modified_parameters"] = {param_type: response}
            session.context["recovery_action"] = "modify_and_retry"
            return None
        
        return None
    
    def _complete_session(self, session_id: str) -> None:
        """Complete a workflow session and move it to history."""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            session.updated_at = time.time()
            self.session_history.append(session)
            logger.info(f"Completed session: {session_id}")
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a workflow session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "interaction_type": session.interaction_type.value,
            "state": session.state.value,
            "interaction_count": len(session.interactions),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "context": session.context,
            "completed_interactions": [
                i for i in session.interactions 
                if i.state == InteractionState.COMPLETED
            ]
        } 