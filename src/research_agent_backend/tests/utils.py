"""Test utility functions for Research Agent backend testing."""

import random
import string
import subprocess
import coverage
from typing import List, Dict, Any
from pathlib import Path


def create_test_document(
    title: str = None,
    content: str = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create a test document with optional customization."""
    if title is None:
        title = f"Test Document {random.randint(1000, 9999)}"
    
    if content is None:
        content = f"""
        # {title}
        
        This is test content for {title}.
        
        ## Section 1
        Some content about testing and development.
        
        ## Section 2  
        Additional content with random data: {''.join(random.choices(string.ascii_letters, k=50))}
        """
    
    if metadata is None:
        metadata = {
            "source": "test",
            "category": "test_data",
            "created_at": "2025-01-01T00:00:00Z"
        }
    
    return {
        "title": title,
        "content": content.strip(),
        "metadata": metadata
    }


def create_test_embeddings(dimension: int = 5, count: int = 1) -> List[List[float]]:
    """Create test embeddings with specified dimensions."""
    embeddings = []
    for i in range(count):
        # Generate normalized random embeddings
        embedding = [random.random() for _ in range(dimension)]
        # Simple normalization
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        embeddings.append(embedding)
    
    return embeddings if count > 1 else embeddings[0]


def assert_coverage_meets_threshold(
    source_path: str,
    threshold: float = 95.0,
    exclude_patterns: List[str] = None
) -> bool:
    """Assert that test coverage meets the specified threshold."""
    if exclude_patterns is None:
        exclude_patterns = ["*/tests/*", "*/test_*", "*/__pycache__/*"]
    
    try:
        # Run coverage check
        result = subprocess.run([
            "python", "-m", "pytest", 
            f"--cov={source_path}",
            "--cov-report=term-missing",
            "--cov-fail-under", str(threshold)
        ], capture_output=True, text=True)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Coverage check failed: {e}")
        return False


def cleanup_test_data(temp_dirs: List[str] = None, temp_files: List[str] = None):
    """Clean up test data and temporary files."""
    if temp_dirs:
        for temp_dir in temp_dirs:
            path = Path(temp_dir)
            if path.exists() and path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
    
    if temp_files:
        for temp_file in temp_files:
            path = Path(temp_file)
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass  # Ignore if can't delete 