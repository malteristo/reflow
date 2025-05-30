"""
Transaction management and context handling.

This module provides transaction support for document insertion operations
with automatic rollback capabilities and context management.
"""

import logging
from contextlib import contextmanager
from typing import Any, List, Optional
from uuid import uuid4


class TransactionManager:
    """Transaction manager for document insertion operations."""
    
    def __init__(
        self, 
        vector_store: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize transaction manager.
        
        Args:
            vector_store: Vector store instance with transaction support
            logger: Optional logger instance
        """
        self.vector_store = vector_store
        self.logger = logger or logging.getLogger(__name__)
        self._transaction_stack: List[str] = []
    
    @contextmanager
    def transaction_context(self):
        """
        Context manager for transaction operations.
        
        Provides automatic transaction management with rollback on exceptions.
        """
        transaction_id = str(uuid4())
        self.logger.debug(f"Beginning transaction {transaction_id}")
        
        try:
            self.begin_transaction()
            yield transaction_id
            self.commit_transaction()
            self.logger.debug(f"Transaction {transaction_id} committed successfully")
        except Exception as e:
            self.rollback_transaction()
            self.logger.error(f"Transaction {transaction_id} rolled back due to error: {e}")
            raise
    
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if hasattr(self.vector_store, 'begin_transaction'):
            self.vector_store.begin_transaction()
        else:
            # Mock transaction for testing
            pass
        
        transaction_id = str(uuid4())
        self._transaction_stack.append(transaction_id)
        self.logger.debug(f"Transaction {transaction_id} started")
    
    def commit_transaction(self) -> None:
        """Commit current transaction."""
        if hasattr(self.vector_store, 'commit_transaction'):
            self.vector_store.commit_transaction()
        else:
            # Mock transaction for testing
            pass
        
        if self._transaction_stack:
            transaction_id = self._transaction_stack.pop()
            self.logger.debug(f"Transaction {transaction_id} committed")
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction."""
        if hasattr(self.vector_store, 'rollback_transaction'):
            self.vector_store.rollback_transaction()
        else:
            # Mock transaction for testing
            pass
        
        if self._transaction_stack:
            transaction_id = self._transaction_stack.pop()
            self.logger.debug(f"Transaction {transaction_id} rolled back") 