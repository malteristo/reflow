#!/usr/bin/env python3
"""
Coverage Check Script for Research Agent Backend

This script validates test coverage meets TDD requirements and provides
detailed reporting on coverage gaps.

Usage:
    python scripts/testing/coverage_check.py [--threshold=95] [--html] [--fail-on-missing]
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional


class CoverageChecker:
    """Check and validate test coverage for TDD compliance."""
    
    def __init__(self, threshold: int = 95, generate_html: bool = False):
        self.threshold = threshold
        self.generate_html = generate_html
        self.root_path = Path.cwd()
        
    def run_coverage_analysis(self) -> Dict:
        """Run coverage analysis and return results."""
        print(f"🔍 Running coverage analysis with {self.threshold}% threshold...")
        
        # Build coverage command
        cmd = "python -m pytest --cov=src --cov-report=json --cov-report=term-missing"
        if self.generate_html:
            cmd += " --cov-report=html"
            
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.root_path
            )
            
            # Load JSON coverage report
            coverage_json_path = self.root_path / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)
            else:
                coverage_data = {}
                
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "coverage_data": coverage_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "coverage_data": {}
            }
    
    def analyze_coverage_gaps(self, coverage_data: Dict) -> List[Dict]:
        """Analyze coverage data to identify gaps and missing coverage."""
        gaps = []
        
        if "files" not in coverage_data:
            return gaps
            
        for file_path, file_data in coverage_data["files"].items():
            if file_path.startswith("src/") and "test" not in file_path:
                coverage_percent = file_data.get("summary", {}).get("percent_covered", 0)
                
                if coverage_percent < self.threshold:
                    missing_lines = file_data.get("missing_lines", [])
                    gaps.append({
                        "file": file_path,
                        "coverage": coverage_percent,
                        "missing_lines": missing_lines,
                        "missing_count": len(missing_lines)
                    })
                    
        return gaps
    
    def generate_coverage_report(self, coverage_data: Dict, gaps: List[Dict]):
        """Generate detailed coverage report."""
        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
        
        print(f"\n📊 Coverage Analysis Report")
        print("=" * 50)
        print(f"Overall Coverage: {total_coverage:.1f}%")
        print(f"Required Threshold: {self.threshold}%")
        
        if total_coverage >= self.threshold:
            print("✅ Coverage threshold met!")
        else:
            print(f"❌ Coverage below threshold by {self.threshold - total_coverage:.1f}%")
            
        if gaps:
            print(f"\n📋 Coverage Gaps ({len(gaps)} files need improvement):")
            print("-" * 40)
            
            for gap in sorted(gaps, key=lambda x: x["coverage"]):
                print(f"📁 {gap['file']}")
                print(f"   Coverage: {gap['coverage']:.1f}% (missing {gap['missing_count']} lines)")
                if gap['missing_lines']:
                    lines_preview = gap['missing_lines'][:5]
                    lines_str = ", ".join(map(str, lines_preview))
                    if len(gap['missing_lines']) > 5:
                        lines_str += f"... (+{len(gap['missing_lines']) - 5} more)"
                    print(f"   Missing lines: {lines_str}")
                print()
        else:
            print("\n✅ No coverage gaps found!")
            
        if self.generate_html:
            print(f"\n📄 HTML report generated in: htmlcov/index.html")
            
    def suggest_tdd_improvements(self, gaps: List[Dict]):
        """Suggest TDD-specific improvements for coverage gaps."""
        if not gaps:
            return
            
        print("\n🎯 TDD Improvement Suggestions:")
        print("=" * 40)
        
        for gap in gaps[:3]:  # Top 3 most critical gaps
            print(f"\n📁 {gap['file']}")
            print("💡 Suggested TDD approach:")
            print("   1. Write failing tests for missing lines")
            print("   2. Run tests to confirm RED phase")
            print("   3. Implement minimal code to pass (GREEN)")
            print("   4. Refactor while keeping tests green")
            
        if len(gaps) > 3:
            print(f"\n... and {len(gaps) - 3} more files need attention")
            
    def check_compliance(self) -> bool:
        """Check if current coverage meets TDD compliance requirements."""
        results = self.run_coverage_analysis()
        
        if not results["success"]:
            print(f"❌ Coverage analysis failed: {results['error']}")
            return False
            
        gaps = self.analyze_coverage_gaps(results["coverage_data"])
        self.generate_coverage_report(results["coverage_data"], gaps)
        
        total_coverage = results["coverage_data"].get("totals", {}).get("percent_covered", 0)
        
        if total_coverage >= self.threshold:
            print(f"\n🎉 TDD Coverage Compliance: PASSED")
            return True
        else:
            print(f"\n❌ TDD Coverage Compliance: FAILED")
            self.suggest_tdd_improvements(gaps)
            return False


def main():
    """Main entry point for coverage checker."""
    parser = argparse.ArgumentParser(description="TDD Coverage Compliance Checker")
    parser.add_argument(
        "--threshold",
        type=int,
        default=95,
        help="Coverage threshold percentage (default: 95)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with error code if coverage below threshold"
    )
    
    args = parser.parse_args()
    
    if args.threshold < 0 or args.threshold > 100:
        print("❌ Threshold must be between 0 and 100")
        sys.exit(1)
        
    checker = CoverageChecker(
        threshold=args.threshold,
        generate_html=args.html
    )
    
    compliance_passed = checker.check_compliance()
    
    if args.fail_on_missing and not compliance_passed:
        print("\n💥 Exiting with error due to insufficient coverage")
        sys.exit(1)
    elif compliance_passed:
        print("\n✅ Coverage check completed successfully")
    else:
        print("\n⚠️  Coverage check completed with warnings")


if __name__ == "__main__":
    main() 