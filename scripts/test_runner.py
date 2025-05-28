#!/usr/bin/env python3
"""
Test Runner for Research Agent - TDD Workflow Support

Simple script to run different categories of tests during development.
Optimized for AI-assisted TDD workflows.

Usage:
    python scripts/test_runner.py unit          # Run unit tests only
    python scripts/test_runner.py integration   # Run integration tests only
    python scripts/test_runner.py cli           # Run CLI tests only
    python scripts/test_runner.py mcp           # Run MCP tests only
    python scripts/test_runner.py async         # Run async tests only
    python scripts/test_runner.py red           # Run TDD red phase tests
    python scripts/test_runner.py green         # Run TDD green phase tests
    python scripts/test_runner.py all           # Run all tests
    python scripts/test_runner.py coverage      # Run with detailed coverage
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if test_type == "unit":
        cmd = base_cmd + ["-m", "unit", "-v"]
    elif test_type == "integration":
        cmd = base_cmd + ["-m", "integration", "-v", "--durations=10"]
    elif test_type == "e2e":
        cmd = base_cmd + ["-m", "e2e", "-v", "--durations=20"]
    elif test_type == "cli":
        cmd = base_cmd + ["-m", "cli", "-v"]
    elif test_type == "mcp":
        cmd = base_cmd + ["-m", "mcp", "-v"]
    elif test_type == "async":
        cmd = base_cmd + ["-m", "async", "-v"]
    elif test_type == "red":
        cmd = base_cmd + ["-m", "tdd_red", "-v", "--tb=short"]
        print("Running TDD RED phase tests (these should fail by design)")
    elif test_type == "green":
        cmd = base_cmd + ["-m", "tdd_green", "-v"]
        print("Running TDD GREEN phase tests (minimal implementations)")
    elif test_type == "all":
        cmd = base_cmd + ["-v"]
    elif test_type == "coverage":
        cmd = base_cmd + ["--cov=src", "--cov-report=html", "--cov-report=term-missing", "-v"]
    elif test_type == "watch":
        print("Watch mode - install pytest-watch: pip install pytest-watch")
        cmd = ["ptw", "--", "-v", "-m", "not slow"]
    else:
        print(f"Unknown test type: {test_type}")
        print(__doc__)
        sys.exit(1)
    
    # Run the tests
    exit_code = run_command(cmd)
    
    if test_type == "coverage" and exit_code == 0:
        coverage_dir = Path("htmlcov/index.html")
        if coverage_dir.exists():
            print(f"\nCoverage report generated: {coverage_dir.absolute()}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 