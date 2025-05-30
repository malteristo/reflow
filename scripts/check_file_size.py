#!/usr/bin/env python3
"""
File Size Monitoring Utility

Checks Python files for size thresholds and provides warnings for files
that need architectural refactoring per file_organization.mdc rules.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Size thresholds from file_organization.mdc
WARNING_THRESHOLD = 300    # Files over this need architectural review
SOFT_LIMIT = 500          # Files over this should be split
HARD_LIMIT = 1000         # Files should never exceed this

def count_lines(file_path: Path) -> int:
    """Count lines in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0

def find_python_files(root_dir: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Find all Python files in the project."""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.venv', 'venv', '.git', 'node_modules']
    
    python_files = []
    for file_path in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue
        python_files.append(file_path)
    
    return python_files

def categorize_files_by_size(files: List[Tuple[Path, int]]) -> Dict[str, List[Tuple[Path, int]]]:
    """Categorize files by size thresholds."""
    categories = {
        'good': [],        # < 300 lines
        'warning': [],     # 300-499 lines  
        'soft_limit': [],  # 500-999 lines
        'hard_limit': []   # 1000+ lines
    }
    
    for file_path, line_count in files:
        if line_count >= HARD_LIMIT:
            categories['hard_limit'].append((file_path, line_count))
        elif line_count >= SOFT_LIMIT:
            categories['soft_limit'].append((file_path, line_count))
        elif line_count >= WARNING_THRESHOLD:
            categories['warning'].append((file_path, line_count))
        else:
            categories['good'].append((file_path, line_count))
    
    return categories

def print_file_size_report(categories: Dict[str, List[Tuple[Path, int]]], show_good: bool = False):
    """Print a formatted file size report."""
    total_files = sum(len(files) for files in categories.values())
    
    print(f"\n📊 File Size Analysis ({total_files} Python files)")
    print("=" * 50)
    
    # Critical issues first
    if categories['hard_limit']:
        print(f"\n🚨 CRITICAL: Files exceeding hard limit ({HARD_LIMIT}+ lines)")
        print("   These files MUST be refactored immediately:")
        for file_path, lines in sorted(categories['hard_limit'], key=lambda x: x[1], reverse=True):
            print(f"   📄 {file_path.relative_to(Path.cwd())}: {lines:,} lines")
    
    if categories['soft_limit']:
        print(f"\n⚠️  WARNING: Files exceeding soft limit ({SOFT_LIMIT}-{HARD_LIMIT-1} lines)")
        print("   These files should be evaluated for splitting:")
        for file_path, lines in sorted(categories['soft_limit'], key=lambda x: x[1], reverse=True):
            print(f"   📄 {file_path.relative_to(Path.cwd())}: {lines:,} lines")
    
    if categories['warning']:
        print(f"\n📋 REVIEW: Files approaching limits ({WARNING_THRESHOLD}-{SOFT_LIMIT-1} lines)")
        print("   These files need architectural review:")
        for file_path, lines in sorted(categories['warning'], key=lambda x: x[1], reverse=True):
            print(f"   📄 {file_path.relative_to(Path.cwd())}: {lines:,} lines")
    
    if show_good and categories['good']:
        print(f"\n✅ GOOD: Files within limits (< {WARNING_THRESHOLD} lines)")
        print(f"   {len(categories['good'])} files are properly sized")
        
        # Show largest files in good category
        good_files = sorted(categories['good'], key=lambda x: x[1], reverse=True)[:5]
        for file_path, lines in good_files:
            print(f"   📄 {file_path.relative_to(Path.cwd())}: {lines} lines")
        if len(categories['good']) > 5:
            print(f"   ... and {len(categories['good']) - 5} more")
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"   ✅ Good files: {len(categories['good'])}")
    print(f"   📋 Review needed: {len(categories['warning'])}")
    print(f"   ⚠️  Soft limit exceeded: {len(categories['soft_limit'])}")
    print(f"   🚨 Hard limit exceeded: {len(categories['hard_limit'])}")
    
    # Recommendations
    problem_files = len(categories['warning']) + len(categories['soft_limit']) + len(categories['hard_limit'])
    if problem_files > 0:
        print(f"\n💡 Recommendations:")
        print(f"   • Review file_organization.mdc for refactoring guidelines")
        print(f"   • Consider extracting utilities, classes, or modules")
        print(f"   • Maintain test coverage during refactoring")
        print(f"   • Update TaskMaster subtasks with architectural plans")

def main():
    parser = argparse.ArgumentParser(description="Check Python file sizes against organizational standards")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: current directory)")
    parser.add_argument("--max-lines", type=int, default=SOFT_LIMIT, help=f"Custom soft limit (default: {SOFT_LIMIT})")
    parser.add_argument("--show-good", action="store_true", help="Show files within size limits")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--exclude", nargs="*", default=[], help="Additional patterns to exclude")
    
    args = parser.parse_args()
    
    # Use custom limit if provided
    soft_limit = args.max_lines if args.max_lines != SOFT_LIMIT else SOFT_LIMIT
    
    root_path = Path(args.root).resolve()
    if not root_path.exists():
        print(f"Error: Directory {root_path} does not exist")
        return 1
    
    print(f"🔍 Scanning Python files in: {root_path}")
    
    # Find and analyze files
    exclude_patterns = ['__pycache__', '.venv', 'venv', '.git', 'node_modules'] + args.exclude
    python_files = find_python_files(root_path, exclude_patterns)
    
    if not python_files:
        print("No Python files found!")
        return 0
    
    # Count lines for each file
    file_sizes = []
    for file_path in python_files:
        line_count = count_lines(file_path)
        if line_count > 0:  # Skip files we couldn't read
            file_sizes.append((file_path, line_count))
    
    # Categorize and report (use custom soft_limit if provided)
    if soft_limit != SOFT_LIMIT:
        # Use custom categorization
        categories = categorize_files_by_custom_limit(file_sizes, soft_limit)
    else:
        categories = categorize_files_by_size(file_sizes)
    
    if args.format == "json":
        import json
        result = {}
        for category, files in categories.items():
            result[category] = [{"file": str(f.relative_to(root_path)), "lines": lines} for f, lines in files]
        print(json.dumps(result, indent=2))
    else:
        print_file_size_report(categories, args.show_good)
    
    # Return exit code based on violations
    exit_code = 0
    if categories['hard_limit']:
        exit_code = 2  # Critical issues
    elif categories['soft_limit']:
        exit_code = 1  # Warnings
    
    return exit_code

def categorize_files_by_custom_limit(files: List[Tuple[Path, int]], custom_soft_limit: int) -> Dict[str, List[Tuple[Path, int]]]:
    """Categorize files by custom size thresholds."""
    categories = {
        'good': [],        # < WARNING_THRESHOLD lines
        'warning': [],     # WARNING_THRESHOLD to custom_soft_limit-1 lines  
        'soft_limit': [],  # custom_soft_limit to HARD_LIMIT-1 lines
        'hard_limit': []   # HARD_LIMIT+ lines
    }
    
    for file_path, line_count in files:
        if line_count >= HARD_LIMIT:
            categories['hard_limit'].append((file_path, line_count))
        elif line_count >= custom_soft_limit:
            categories['soft_limit'].append((file_path, line_count))
        elif line_count >= WARNING_THRESHOLD:
            categories['warning'].append((file_path, line_count))
        else:
            categories['good'].append((file_path, line_count))
    
    return categories

if __name__ == "__main__":
    sys.exit(main()) 