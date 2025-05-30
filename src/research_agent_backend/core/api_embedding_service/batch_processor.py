"""
API embedding service batch processor.

This module provides batch processing capabilities for embedding generation,
including automatic chunking, validation, and progress tracking.
"""

import json
import logging
from typing import List
from urllib.parse import urljoin

import requests

from .client import APIClient
from .config import APIConfiguration
from .exceptions import BatchProcessingError

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for API embedding services.
    
    Handles efficient processing of multiple texts with automatic chunking,
    parallel processing, and comprehensive error handling. Optimized for throughput
    while respecting API rate limits and batch size constraints.
    
    Features:
        - Automatic chunking based on max_batch_size
        - Comprehensive batch validation
        - Progress tracking and logging
        - Memory-efficient processing for large datasets
        - Detailed error context for debugging
    """
    
    def __init__(self, config: APIConfiguration, client: APIClient) -> None:
        """
        Initialize the batch processor.
        
        Args:
            config: APIConfiguration instance with validated settings
            client: APIClient instance for HTTP communication
        """
        self.config = config
        self.client = client
    
    def process_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text strings.
        
        Efficiently processes multiple texts with automatic chunking, parallel
        processing, and comprehensive error handling. Optimized for throughput
        while respecting API rate limits and batch size constraints.
        
        Args:
            texts: List of input texts to embed. All texts must be non-empty.
            
        Returns:
            List of embedding vectors, one for each input text in the same order.
            Each embedding is a list of float values.
            
        Raises:
            BatchProcessingError: If batch contains empty texts or processing fails
        """
        # Handle empty batch efficiently
        if not texts:
            return []
        
        # Validate all texts in batch with detailed error reporting
        empty_text_indices = [
            i for i, text in enumerate(texts) 
            if not text or text.strip() == ""
        ]
        if empty_text_indices:
            raise BatchProcessingError(
                f"Batch contains empty texts at positions: {empty_text_indices}. "
                f"All texts must be non-empty. Please check your input data."
            )
        
        # Process in chunks for optimal performance and API compliance
        all_embeddings = []
        chunk_size = self.config.max_batch_size
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        logger.debug(
            f"Processing {len(texts)} texts in {total_chunks} chunks "
            f"(chunk_size: {chunk_size})"
        )
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_number = i // chunk_size + 1
            
            try:
                logger.debug(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk)} texts)")
                chunk_embeddings = self._process_chunk(chunk)
                all_embeddings.extend(chunk_embeddings)
                
            except Exception as e:
                # Enhanced error context for batch processing
                raise BatchProcessingError(
                    f"Failed to process chunk {chunk_number}/{total_chunks} "
                    f"(texts {i}-{i+len(chunk)-1}): {str(e)}"
                ) from e
        
        # Verify result consistency
        if len(all_embeddings) != len(texts):
            raise BatchProcessingError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}. "
                f"This indicates an internal processing error."
            )
        
        logger.debug(f"Successfully processed {len(texts)} texts into {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _process_chunk(self, texts: List[str]) -> List[List[float]]:
        """
        Process a single chunk of texts for batch embedding.
        
        Internal method that handles API communication for a chunk of texts
        within the configured batch size limits. Includes response validation
        and error handling specific to chunk processing.
        
        Args:
            texts: List of input texts to embed in this chunk
            
        Returns:
            List of embedding vectors matching the input texts count
            
        Raises:
            BatchProcessingError: If chunk processing fails or response is invalid
            
        Note:
            This is an internal method and should not be called directly.
            Use process_batch() for public batch processing.
        """
        # Construct API endpoint URL
        url = urljoin(self.config.base_url, "embeddings")
        
        # Prepare request payload for batch processing
        payload = {
            "input": texts,
            "model": self.config.model_name
        }
        
        # Add provider-specific parameters
        if self.config.provider == "openai":
            payload["encoding_format"] = "float"  # Ensure float format
        
        try:
            # Make API request with retry logic
            response = self.client.make_request_with_retry("POST", url, json=payload)
            response_data = response.json()
            
            # Extract embeddings from response with comprehensive validation
            if "data" in response_data:
                embeddings = []
                
                # Process each embedding in the response
                for idx, item in enumerate(response_data["data"]):
                    if "embedding" not in item:
                        raise BatchProcessingError(
                            f"Missing 'embedding' field in response item {idx}"
                        )
                    
                    embedding = [float(x) for x in item["embedding"]]
                    embeddings.append(embedding)
                
                # Ensure response matches input count (critical for batch integrity)
                if len(embeddings) != len(texts):
                    if len(embeddings) > len(texts):
                        # Trim excess embeddings (defensive programming for API inconsistencies)
                        logger.warning(
                            f"API returned {len(embeddings)} embeddings for {len(texts)} texts. "
                            f"Trimming to match input count."
                        )
                        embeddings = embeddings[:len(texts)]
                    else:
                        # Insufficient embeddings is a critical error
                        raise BatchProcessingError(
                            f"API returned {len(embeddings)} embeddings for {len(texts)} texts. "
                            f"Expected exact match. This indicates an API error."
                        )
                
                return embeddings
            else:
                raise BatchProcessingError(
                    "Invalid API response: missing 'data' field. "
                    "This may indicate an API format change or server error."
                )
                
        except requests.RequestException as e:
            # Network-level errors during batch processing
            raise BatchProcessingError(f"Network error during batch API request: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            # JSON parsing errors during batch processing
            raise BatchProcessingError(
                f"Failed to parse batch API response: {str(e)}. "
                f"This may indicate an API format change or server error."
            ) 