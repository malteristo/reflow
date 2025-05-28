#!/usr/bin/env python3
"""
TDD Cycle Automation Script for Research Agent Backend

This script automates the Red-Green-Refactor TDD cycle by:
1. Running tests to confirm RED phase (failing tests)
2. Prompting for minimal implementation (GREEN phase)  
3. Running tests to confirm GREEN phase (passing tests)
4. Prompting for refactoring (REFACTOR phase)
5. Running final tests to ensure no regressions

Usage:
    python scripts/testing/run_tdd_cycle.py [--test-file=path] [--module=name]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class TDDCycleRunner:
    """Automates the TDD Red-Green-Refactor cycle."""
    
    def __init__(self, test_file: Optional[Path] = None, module_name: Optional[str] = None):
        self.test_file = test_file
        self.module_name = module_name
        self.root_path = Path.cwd()
        
    def run_tests(self, phase: str) -> bool:
        """Run tests and report results for the given TDD phase."""
        print(f"\n🔄 Running tests for {phase} phase...")
        
        if self.test_file:
            cmd = f"python -m pytest {self.test_file} -v --tb=short"
        else:
            cmd = "python -m pytest -m unit -v --tb=short"
            
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.root_path
            )
            
            if phase == "RED":
                if result.returncode != 0:
                    print(f"✅ {phase} phase confirmed: Tests are failing as expected")
                    print("📋 Failing test output:")
                    print(result.stdout[-500:])
                    return True
                else:
                    print(f"❌ {phase} phase failed: Tests should be failing but they're passing")
                    return False
            else:  # GREEN or REFACTOR
                if result.returncode == 0:
                    print(f"✅ {phase} phase confirmed: Tests are passing")
                    return True
                else:
                    print(f"❌ {phase} phase failed: Tests should be passing but they're failing")
                    print("📋 Failed test output:")
                    print(result.stdout[-500:])
                    return False
                    
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def wait_for_user_action(self, phase: str, action: str):
        """Wait for user to complete the required action for the phase."""
        print(f"\n🎯 {phase} Phase Action Required:")
        print(f"   {action}")
        print("\nPress Enter when you have completed this action...")
        input()
    
    def run_cycle(self):
        """Execute the complete TDD cycle."""
        print("🚀 Starting TDD Red-Green-Refactor Cycle")
        print("=" * 50)
        
        # RED Phase
        print("\n🔴 RED PHASE: Write failing tests that define desired behavior")
        if not self.test_file:
            self.wait_for_user_action(
                "RED",
                "Write failing tests that define the expected behavior of your feature"
            )
        
        if not self.run_tests("RED"):
            print("❌ RED phase incomplete. Please ensure tests are failing.")
            return False
            
        # GREEN Phase
        print("\n🟢 GREEN PHASE: Write minimal code to make tests pass")
        self.wait_for_user_action(
            "GREEN", 
            "Write the minimal amount of code needed to make the tests pass"
        )
        
        if not self.run_tests("GREEN"):
            print("❌ GREEN phase incomplete. Please fix implementation to make tests pass.")
            return False
            
        # REFACTOR Phase
        print("\n🔵 REFACTOR PHASE: Improve code quality while keeping tests green")
        self.wait_for_user_action(
            "REFACTOR",
            "Refactor and improve your code quality without changing behavior"
        )
        
        if not self.run_tests("REFACTOR"):
            print("❌ REFACTOR phase incomplete. Refactoring broke the tests.")
            return False
            
        print("\n🎉 TDD Cycle Complete!")
        print("✅ All phases passed successfully")
        print("🎯 Your feature is now implemented with proper test coverage")
        return True


def main():
    """Main entry point for TDD cycle runner."""
    parser = argparse.ArgumentParser(description="TDD Cycle Automation")
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Specific test file to run (default: run unit tests)"
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Module name being developed"
    )
    
    args = parser.parse_args()
    
    if args.test_file and not args.test_file.exists():
        print(f"❌ Test file not found: {args.test_file}")
        sys.exit(1)
    
    runner = TDDCycleRunner(test_file=args.test_file, module_name=args.module)
    
    success = runner.run_cycle()
    
    if not success:
        print("\n❌ TDD cycle incomplete. Please address the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 