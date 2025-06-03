"""
Configuration system integration hooks for model change detection.

This module provides automatic model registration and configuration change callbacks
to integrate the model change detection system with the Research Agent configuration
management and embedding service initialization.

Part of Task 35 - Model Change Detection Integration Phase 3.
"""

import logging
from typing import Any, Dict, Optional, Callable
import threading
from functools import wraps

from .detector import ModelChangeDetector
from .fingerprint import ModelFingerprint
from .events import ModelChangeEvent

logger = logging.getLogger(__name__)


class ConfigurationIntegrationHooks:
    """
    Configuration system integration hooks for automatic model change detection.
    
    Provides automatic model registration during embedding service initialization
    and configuration change callbacks to trigger model change detection when
    configuration changes occur.
    
    Features:
        - Automatic model registration on service initialization
        - Configuration change callbacks with debouncing
        - Collection metadata integration
        - Query compatibility validation hooks
    """
    
    def __init__(self):
        """Initialize configuration integration hooks."""
        self._detector = ModelChangeDetector()
        self._callbacks: Dict[str, Callable] = {}
        self._callback_lock = threading.Lock()
        self._debounce_timers: Dict[str, threading.Timer] = {}
        self._debounce_delay = 5.0  # 5 second debounce delay
        
        logger.info("Configuration integration hooks initialized")
    
    def register_embedding_service_automatically(self, embedding_service) -> Optional[ModelChangeEvent]:
        """
        Automatically register an embedding service's model when the service is initialized.
        
        This method should be called during embedding service initialization to automatically
        register the model in the change detection system. It generates a model fingerprint
        and registers it if a change is detected.
        
        Args:
            embedding_service: The embedding service instance (LocalEmbeddingService or APIEmbeddingService)
            
        Returns:
            ModelChangeEvent if a change was detected and registered, None otherwise
            
        Example:
            >>> hooks = ConfigurationIntegrationHooks()
            >>> service = LocalEmbeddingService("new-model")
            >>> event = hooks.register_embedding_service_automatically(service)
            >>> if event:
            ...     print(f"Registered model change: {event.change_type}")
        """
        try:
            # Generate fingerprint using the service's built-in method
            current_fingerprint = embedding_service.generate_model_fingerprint()
            
            # Check if registration is needed
            change_detected = self._detector.detect_change(current_fingerprint)
            
            if change_detected:
                # Register the model and return the event
                event = self._detector.register_model_with_event(current_fingerprint)
                
                logger.info(
                    f"Automatically registered model '{event.model_name}' "
                    f"(type: {event.change_type}, reindex: {event.requires_reindexing})"
                )
                
                # Trigger callbacks for model change
                self._trigger_callbacks('model_change', {
                    'event': event,
                    'service': embedding_service,
                    'fingerprint': current_fingerprint
                })
                
                return event
            else:
                logger.debug(f"Model '{current_fingerprint.model_name}' already registered and up to date")
                return None
                
        except Exception as e:
            logger.error(f"Failed to automatically register embedding service: {e}")
            return None
    
    def add_configuration_change_callback(self, callback_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback to be triggered when configuration changes occur.
        
        Callbacks are debounced to avoid excessive triggering during rapid configuration changes.
        
        Args:
            callback_name: Unique name for the callback
            callback: Function to call with change information
            
        Example:
            >>> hooks = ConfigurationIntegrationHooks()
            >>> def on_model_change(change_info):
            ...     print(f"Model configuration changed: {change_info}")
            >>> hooks.add_configuration_change_callback("my_handler", on_model_change)
        """
        with self._callback_lock:
            self._callbacks[callback_name] = callback
            logger.debug(f"Added configuration change callback: {callback_name}")
    
    def remove_configuration_change_callback(self, callback_name: str) -> None:
        """
        Remove a configuration change callback.
        
        Args:
            callback_name: Name of the callback to remove
        """
        with self._callback_lock:
            if callback_name in self._callbacks:
                del self._callbacks[callback_name]
                logger.debug(f"Removed configuration change callback: {callback_name}")
    
    def trigger_configuration_change(self, change_type: str, change_data: Dict[str, Any]) -> None:
        """
        Trigger configuration change detection and callbacks.
        
        This method should be called by the configuration system when configuration
        changes occur that might affect model fingerprints.
        
        Args:
            change_type: Type of configuration change ('embedding_model', 'api_config', etc.)
            change_data: Dictionary containing the changed configuration data
            
        Example:
            >>> hooks = ConfigurationIntegrationHooks()
            >>> hooks.trigger_configuration_change('embedding_model', {
            ...     'old_model': 'old-model-name',
            ...     'new_model': 'new-model-name'
            ... })
        """
        logger.info(f"Configuration change detected: {change_type}")
        
        # Debounce the callback triggering
        self._debounced_trigger_callbacks(change_type, change_data)
    
    def _debounced_trigger_callbacks(self, change_type: str, change_data: Dict[str, Any]) -> None:
        """
        Trigger callbacks with debouncing to avoid excessive calls.
        
        Args:
            change_type: Type of configuration change
            change_data: Change data to pass to callbacks
        """
        # Cancel existing timer for this change type
        if change_type in self._debounce_timers:
            self._debounce_timers[change_type].cancel()
        
        # Create new debounced timer
        timer = threading.Timer(
            self._debounce_delay,
            self._trigger_callbacks,
            args=(change_type, change_data)
        )
        timer.start()
        self._debounce_timers[change_type] = timer
    
    def _trigger_callbacks(self, change_type: str, change_data: Dict[str, Any]) -> None:
        """
        Trigger all registered callbacks with change information.
        
        Args:
            change_type: Type of change that occurred
            change_data: Data about the change
        """
        with self._callback_lock:
            callbacks_to_run = self._callbacks.copy()
        
        for callback_name, callback in callbacks_to_run.items():
            try:
                callback({
                    'change_type': change_type,
                    'data': change_data,
                    'timestamp': logger.time() if hasattr(logger, 'time') else None
                })
            except Exception as e:
                logger.error(f"Configuration change callback '{callback_name}' failed: {e}")
    
    def create_service_initialization_decorator(self):
        """
        Create a decorator for embedding service initialization methods.
        
        This decorator automatically registers the model when an embedding service
        is initialized, integrating seamlessly with existing service classes.
        
        Returns:
            Decorator function that can be applied to __init__ methods
            
        Example:
            >>> hooks = ConfigurationIntegrationHooks()
            >>> auto_register = hooks.create_service_initialization_decorator()
            >>> 
            >>> class MyEmbeddingService:
            ...     @auto_register
            ...     def __init__(self, model_name):
            ...         # Service initialization code
            ...         pass
        """
        def decorator(init_method):
            @wraps(init_method)
            def wrapper(service_instance, *args, **kwargs):
                # Call the original __init__ method
                result = init_method(service_instance, *args, **kwargs)
                
                # Automatically register the service after initialization
                try:
                    self.register_embedding_service_automatically(service_instance)
                except Exception as e:
                    logger.warning(f"Auto-registration failed for {service_instance.__class__.__name__}: {e}")
                
                return result
            return wrapper
        return decorator


# Global instance for easy access across the application
_global_hooks_instance: Optional[ConfigurationIntegrationHooks] = None
_global_hooks_lock = threading.Lock()


def get_integration_hooks() -> ConfigurationIntegrationHooks:
    """
    Get the global configuration integration hooks instance.
    
    Returns:
        Global ConfigurationIntegrationHooks instance (singleton pattern)
    """
    global _global_hooks_instance
    
    if _global_hooks_instance is None:
        with _global_hooks_lock:
            if _global_hooks_instance is None:
                _global_hooks_instance = ConfigurationIntegrationHooks()
    
    return _global_hooks_instance


def auto_register_embedding_service(embedding_service) -> Optional[ModelChangeEvent]:
    """
    Convenience function to automatically register an embedding service.
    
    Args:
        embedding_service: The embedding service to register
        
    Returns:
        ModelChangeEvent if a change was detected and registered, None otherwise
    """
    hooks = get_integration_hooks()
    return hooks.register_embedding_service_automatically(embedding_service)


def add_config_change_callback(callback_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
    """
    Convenience function to add a configuration change callback.
    
    Args:
        callback_name: Unique name for the callback
        callback: Function to call with change information
    """
    hooks = get_integration_hooks()
    hooks.add_configuration_change_callback(callback_name, callback)


def trigger_config_change(change_type: str, change_data: Dict[str, Any]) -> None:
    """
    Convenience function to trigger configuration change detection.
    
    Args:
        change_type: Type of configuration change
        change_data: Dictionary containing the changed configuration data
    """
    hooks = get_integration_hooks()
    hooks.trigger_configuration_change(change_type, change_data) 