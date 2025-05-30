#!/usr/bin/env python3
"""
Quick File Size Check - For Daily Development Workflow

Provides a quick summary of critical file size issues for daily monitoring.
"""

import os
import sys
from pathlib import Path

# Add scripts directory to path to import check_file_size
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from check_file_size import find_python_files, count_lines, categorize_files_by_size

def quick_check():
    """Run a quick file size check and show only critical issues."""
    print("🔍 Quick File Size Check")
    print("=" * 30)
    
    root_path = Path(__file__).parent.parent  # Go up from scripts/ to project root
    
    # Find and analyze files
    exclude_patterns = ['__pycache__', '.venv', 'venv', '.git', 'node_modules']
    python_files = find_python_files(root_path, exclude_patterns)
    
    if not python_files:
        print("No Python files found!")
        return 0
    
    # Count lines for each file
    file_sizes = []
    for file_path in python_files:
        line_count = count_lines(file_path)
        if line_count > 0:
            file_sizes.append((file_path, line_count))
    
    categories = categorize_files_by_size(file_sizes)
    
    # Show critical issues only
    if categories['hard_limit']:
        print(f"🚨 CRITICAL: {len(categories['hard_limit'])} files exceeding 1,000 lines:")
        for file_path, lines in sorted(categories['hard_limit'], key=lambda x: x[1], reverse=True)[:3]:
            rel_path = file_path.relative_to(root_path)
            print(f"   📄 {rel_path}: {lines:,} lines")
        if len(categories['hard_limit']) > 3:
            print(f"   ... and {len(categories['hard_limit']) - 3} more critical files")
    
    if categories['soft_limit']:
        print(f"⚠️  WARNING: {len(categories['soft_limit'])} files exceeding 500 lines")
    
    if categories['warning']:
        print(f"📋 REVIEW: {len(categories['warning'])} files need architectural review (300+ lines)")
    
    print(f"✅ {len(categories['good'])} files properly sized")
    
    # Return exit code for CI integration
    if categories['hard_limit']:
        return 2  # Critical issues
    elif categories['soft_limit']:
        return 1  # Warnings
    return 0

if __name__ == "__main__":
    sys.exit(quick_check()) 