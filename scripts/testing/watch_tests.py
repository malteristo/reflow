#!/usr/bin/env python3
"""
Continuous Test Watcher for Research Agent Backend

This script watches for file changes and automatically runs relevant tests,
supporting TDD workflow with immediate feedback.

Usage:
    python scripts/testing/watch_tests.py [--mode=unit|integration|all]
    
Features:
- Automatic test discovery and execution
- Smart test selection based on changed files
- TDD phase detection and reporting
- Coverage monitoring with thresholds
- Colored output for easy status identification
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set, Dict, Optional
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import re


class TDDTestRunner:
    """Smart test runner that supports TDD workflow."""
    
    def __init__(self, mode: str = "unit", root_path: Optional[Path] = None):
        self.mode = mode
        self.root_path = root_path or Path.cwd()
        self.last_run_time = 0
        self.test_results_cache: Dict[str, bool] = {}
        
    def get_related_test_files(self, changed_file: Path) -> List[str]:
        """Find test files related to the changed source file."""
        test_files = []
        
        # Convert source file path to potential test paths
        if changed_file.name.endswith('.py') and 'test' not in changed_file.name:
            # Direct test file mapping
            test_name = f"test_{changed_file.stem}.py"
            
            # Look for test files in test directories
            for test_dir in ["tests", "test"]:
                for root, dirs, files in os.walk(self.root_path / "src"):
                    if test_dir in root:
                        test_path = Path(root) / test_name
                        if test_path.exists():
                            test_files.append(str(test_path))
        
        return test_files
    
    def detect_tdd_phase(self, test_output: str) -> str:
        """Analyze test output to detect current TDD phase."""
        if "FAILED" in test_output or "ERROR" in test_output:
            if "not implemented" in test_output.lower() or "skip" in test_output.lower():
                return "RED"
            return "RED (Implementation Issues)"
        elif "PASSED" in test_output:
            if "minimal" in test_output.lower() or "basic" in test_output.lower():
                return "GREEN" 
            return "GREEN (Ready for Refactor)"
        return "UNKNOWN"
    
    def run_tests(self, test_files: List[str] = None, changed_file: Path = None) -> bool:
        """Run tests and provide TDD feedback."""
        current_time = time.time()
        if current_time - self.last_run_time < 2:  # Debounce rapid changes
            return True
            
        self.last_run_time = current_time
        
        # Determine which tests to run
        if test_files:
            test_targets = " ".join(test_files)
        else:
            if self.mode == "unit":
                test_targets = "-m unit"
            elif self.mode == "integration": 
                test_targets = "-m integration"
            else:
                test_targets = ""
        
        # Build pytest command
        cmd = f"python -m pytest {test_targets} --tb=short -v"
        if self.mode != "all":
            cmd += f" --cov=src --cov-report=term-missing"
            
        print(f"\n🔄 Running tests... ({self.mode} mode)")
        if changed_file:
            print(f"📝 Changed file: {changed_file}")
            
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.root_path
            )
            
            # Analyze results
            tdd_phase = self.detect_tdd_phase(result.stdout + result.stderr)
            coverage_match = re.search(r'TOTAL.*?(\d+)%', result.stdout)
            coverage = coverage_match.group(1) if coverage_match else "Unknown"
            
            # Report results with TDD context
            if result.returncode == 0:
                print(f"✅ Tests PASSED | TDD Phase: {tdd_phase} | Coverage: {coverage}%")
                if tdd_phase == "GREEN" and coverage != "Unknown":
                    if int(coverage) >= 95:
                        print("🎯 Excellent coverage! Ready for refactoring.")
                    else:
                        print("⚠️  Consider adding more tests before refactoring.")
            else:
                print(f"❌ Tests FAILED | TDD Phase: {tdd_phase}")
                print("📋 Failed test output:")
                print(result.stdout[-500:])  # Show last 500 chars of output
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system events and trigger test runs."""
    
    def __init__(self, test_runner: TDDTestRunner):
        self.test_runner = test_runner
        self.ignored_patterns = {'.git', '__pycache__', '.pytest_cache', '.coverage'}
        
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        return any(pattern in file_path for pattern in self.ignored_patterns)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or self.should_ignore(event.src_path):
            return
            
        file_path = Path(event.src_path)
        
        # Only react to Python files
        if file_path.suffix != '.py':
            return
            
        print(f"\n📂 File changed: {file_path.name}")
        
        # Find and run related tests
        if 'test' in file_path.name:
            # Test file changed, run just that test
            self.test_runner.run_tests([str(file_path)])
        else:
            # Source file changed, find related tests
            related_tests = self.test_runner.get_related_test_files(file_path)
            if related_tests:
                self.test_runner.run_tests(related_tests, file_path)
            else:
                # No specific tests found, run mode-appropriate tests
                self.test_runner.run_tests(changed_file=file_path)


def main():
    """Main entry point for the test watcher."""
    parser = argparse.ArgumentParser(description="Continuous TDD Test Watcher")
    parser.add_argument(
        "--mode", 
        choices=["unit", "integration", "all"],
        default="unit",
        help="Test mode to run (default: unit)"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Root path to watch (default: current directory)"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Starting TDD test watcher in {args.mode} mode...")
    print(f"📁 Watching: {args.path}")
    print("🚀 Make changes to Python files to trigger tests!")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Initialize test runner and file watcher
    test_runner = TDDTestRunner(mode=args.mode, root_path=args.path)
    event_handler = FileChangeHandler(test_runner)
    observer = Observer()
    observer.schedule(event_handler, str(args.path / "src"), recursive=True)
    observer.start()
    
    # Run initial test suite
    print("🎬 Running initial test suite...")
    test_runner.run_tests()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping test watcher...")
        observer.stop()
    
    observer.join()
    print("👋 Test watcher stopped.")


if __name__ == "__main__":
    main() 