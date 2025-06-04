#!/usr/bin/env python3
"""
Simple Root Cause Suggestion Engine

Provides practical, actionable fix suggestions based on test failure patterns.
Integrates with existing test analysis tools to offer targeted recommendations.
"""

import json
import re
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FixSuggestion:
    """Simple fix suggestion with actionable steps."""
    category: str
    description: str
    steps: List[str]
    estimated_minutes: int
    confidence: float
    affected_tests: List[str]
    priority: str  # "high", "medium", "low"

class SimpleSuggestionEngine:
    """
    A practical suggestion engine that provides actionable fix recommendations
    based on test failure patterns from existing analysis tools.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.scripts_dir = self.project_root / "scripts"
        
        # Load existing analysis data
        self.test_results = self._load_test_analysis()
        self.priority_data = self._load_priority_analysis()
        
        # Simple pattern-based fix suggestions
        self.fix_patterns = {
            "assertion_failure": self._get_assertion_fixes,
            "configuration_error": self._get_config_fixes,
            "mock_error": self._get_mock_fixes,
            "missing_method": self._get_missing_method_fixes,
            "not_implemented": self._get_not_implemented_fixes,
            "type_error": self._get_type_error_fixes,
            "import_error": self._get_import_fixes
        }
    
    def _load_test_analysis(self) -> Dict[str, Any]:
        """Load test analysis results."""
        analysis_file = self.scripts_dir / "test_analysis_results.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                return json.load(f)
        return {}
    
    def _load_priority_analysis(self) -> Dict[str, Any]:
        """Load priority analysis results."""
        priority_file = self.scripts_dir / "test_priority_analysis.json"
        if priority_file.exists():
            with open(priority_file) as f:
                return json.load(f)
        return {}
    
    def generate_suggestions(self, category: str = None) -> List[FixSuggestion]:
        """Generate fix suggestions for specified category or all categories."""
        suggestions = []
        
        if not self.test_results or "failure_patterns" not in self.test_results:
            print("No test analysis data found. Run test_analyzer.py first.")
            return suggestions
        
        patterns = self.test_results["failure_patterns"]
        
        # Generate suggestions for each pattern (patterns is a list)
        for pattern in patterns:
            pattern_name = pattern.get("pattern_id", "")
            if category and pattern_name != category:
                continue
                
            if pattern_name in self.fix_patterns:
                # Convert pattern data to expected format
                pattern_data = {
                    "count": pattern.get("frequency", 0),
                    "examples": pattern.get("test_names", [])
                }
                fix_suggestions = self.fix_patterns[pattern_name](pattern_data)
                suggestions.extend(fix_suggestions)
        
        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x.priority],
            x.confidence
        ), reverse=True)
        
        return suggestions
    
    def _get_assertion_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for assertion failures."""
        suggestions = []
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        # Common assertion failure patterns
        suggestions.append(FixSuggestion(
            category="assertion_failure",
            description=f"Review and update {count} assertion failures - likely outdated test expectations",
            steps=[
                "1. Run failing tests individually to understand expected vs actual behavior",
                "2. Check if tests expect legacy functionality that has been updated",
                "3. Update test assertions to match current implementation",
                "4. Verify business logic is correct before updating tests",
                "5. Group similar assertion patterns for batch fixes"
            ],
            estimated_minutes=count * 5,  # 5 minutes per test
            confidence=0.8,
            affected_tests=tests[:10],  # Show first 10 as examples
            priority="medium"
        ))
        
        # Look for specific patterns
        if any("expected" in str(test).lower() for test in tests):
            suggestions.append(FixSuggestion(
                category="assertion_failure",
                description="Update expected values in test assertions",
                steps=[
                    "1. Identify tests with hardcoded expected values",
                    "2. Update expected values to match current behavior",
                    "3. Consider using dynamic expected values where appropriate"
                ],
                estimated_minutes=count * 3,
                confidence=0.9,
                affected_tests=[t for t in tests if "expected" in str(t).lower()][:5],
                priority="high"
            ))
        
        return suggestions
    
    def _get_config_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for configuration errors."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="configuration_error",
            description=f"Fix {count} configuration and setup issues",
            steps=[
                "1. Check test configuration files for missing or incorrect settings",
                "2. Ensure test environment variables are properly set",
                "3. Verify test data files and fixtures are accessible",
                "4. Update test setup/teardown methods for consistent state",
                "5. Standardize configuration across test modules"
            ],
            estimated_minutes=count * 8,  # 8 minutes per config issue
            confidence=0.9,
            affected_tests=tests[:10],
            priority="high"  # Config issues often block multiple tests
        )]
    
    def _get_mock_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for mock errors."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="mock_error",
            description=f"Standardize {count} mock configurations and fix setup issues",
            steps=[
                "1. Review mock setup patterns across failing tests",
                "2. Create standardized mock fixtures for common objects",
                "3. Update mock return values to match expected interfaces",
                "4. Fix mock method signatures and property access",
                "5. Add proper mock isolation between tests"
            ],
            estimated_minutes=count * 6,  # 6 minutes per mock issue
            confidence=0.85,
            affected_tests=tests[:10],
            priority="high"
        )]
    
    def _get_missing_method_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for missing method errors."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="missing_method",
            description=f"Implement {count} missing methods and attributes",
            steps=[
                "1. Identify missing methods from AttributeError messages",
                "2. Add method stubs with appropriate signatures",
                "3. Implement basic functionality or NotImplementedError",
                "4. Update class interfaces to match test expectations",
                "5. Consider interface compatibility requirements"
            ],
            estimated_minutes=count * 10,  # 10 minutes per missing method
            confidence=0.75,
            affected_tests=tests[:10],
            priority="medium"
        )]
    
    def _get_not_implemented_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for NotImplementedError."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="not_implemented",
            description=f"Complete {count} placeholder implementations",
            steps=[
                "1. Locate NotImplementedError placeholders in codebase",
                "2. Implement basic functionality for each placeholder",
                "3. Add proper error handling and validation",
                "4. Write minimal implementation to pass tests",
                "5. Consider grouping similar implementations for efficiency"
            ],
            estimated_minutes=count * 15,  # 15 minutes per implementation
            confidence=0.7,
            affected_tests=tests[:10],
            priority="low"  # Usually in performance tests, less critical
        )]
    
    def _get_type_error_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for type errors."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="type_error",
            description=f"Fix {count} type and method signature mismatches",
            steps=[
                "1. Review TypeError messages for signature mismatches",
                "2. Update method signatures to match expected interfaces",
                "3. Add proper type annotations where missing",
                "4. Fix argument count and parameter names",
                "5. Ensure compatibility with existing callers"
            ],
            estimated_minutes=count * 7,  # 7 minutes per type error
            confidence=0.8,
            affected_tests=tests[:10],
            priority="medium"
        )]
    
    def _get_import_fixes(self, pattern_data: Dict[str, Any]) -> List[FixSuggestion]:
        """Generate fixes for import errors."""
        count = pattern_data.get("count", 0)
        tests = pattern_data.get("examples", [])
        
        return [FixSuggestion(
            category="import_error",
            description=f"Resolve {count} import and module loading issues",
            steps=[
                "1. Check for missing files or modules",
                "2. Fix import paths and module structure",
                "3. Add __init__.py files where needed",
                "4. Update PYTHONPATH if necessary",
                "5. Verify module dependencies are installed"
            ],
            estimated_minutes=count * 12,  # 12 minutes per import issue
            confidence=0.9,
            affected_tests=tests[:10],
            priority="high"  # Import errors often block many tests
        )]
    
    def print_suggestions(self, suggestions: List[FixSuggestion]):
        """Print suggestions in a readable format."""
        if not suggestions:
            print("No suggestions generated. Check test analysis data.")
            return
        
        print(f"\n🔧 ROOT CAUSE SUGGESTIONS ({len(suggestions)} recommendations)")
        print("=" * 70)
        
        for i, suggestion in enumerate(suggestions, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[suggestion.priority]
            
            print(f"\n{i}. {priority_icon} {suggestion.description}")
            print(f"   Category: {suggestion.category}")
            print(f"   Time Estimate: {suggestion.estimated_minutes} minutes")
            print(f"   Confidence: {suggestion.confidence:.0%}")
            print(f"   Priority: {suggestion.priority.upper()}")
            
            print("\n   📋 Action Steps:")
            for step in suggestion.steps:
                print(f"      {step}")
            
            if suggestion.affected_tests:
                print(f"\n   🧪 Example Tests ({len(suggestion.affected_tests)}):")
                for test in suggestion.affected_tests[:3]:  # Show first 3
                    print(f"      • {test}")
                if len(suggestion.affected_tests) > 3:
                    print(f"      ... and {len(suggestion.affected_tests) - 3} more")
            
            print("-" * 50)
    
    def export_suggestions(self, suggestions: List[FixSuggestion], filename: str = None):
        """Export suggestions to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fix_suggestions_{timestamp}.json"
        
        output_file = self.scripts_dir / filename
        
        # Convert to JSON-serializable format
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_suggestions": len(suggestions),
            "suggestions": [asdict(suggestion) for suggestion in suggestions]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Suggestions exported to: {output_file}")
        return output_file

def main():
    """Main CLI interface."""
    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category not in ["assertion_failure", "configuration_error", "mock_error", 
                           "missing_method", "not_implemented", "type_error", "import_error"]:
            print(f"Unknown category: {category}")
            print("Available categories: assertion_failure, configuration_error, mock_error,")
            print("                     missing_method, not_implemented, type_error, import_error")
            sys.exit(1)
    else:
        category = None
    
    # Create suggestion engine
    engine = SimpleSuggestionEngine()
    
    # Generate suggestions
    suggestions = engine.generate_suggestions(category)
    
    # Display results
    engine.print_suggestions(suggestions)
    
    # Export to file
    if suggestions:
        engine.export_suggestions(suggestions)
        
        # Quick summary
        total_time = sum(s.estimated_minutes for s in suggestions)
        high_priority = len([s for s in suggestions if s.priority == "high"])
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total recommendations: {len(suggestions)}")
        print(f"   High priority items: {high_priority}")
        print(f"   Estimated total time: {total_time} minutes ({total_time/60:.1f} hours)")

if __name__ == "__main__":
    main() 