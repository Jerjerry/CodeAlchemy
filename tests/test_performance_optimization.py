import ast
import unittest
import sys
import os
import timeit
import inspect
import textwrap
import time
from functools import lru_cache
import math

# Explicitly add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ai.models.performance_optimization import PerformanceOptimizer

class ProductionPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test class - create shared optimizer instance."""
        cls.optimizer = PerformanceOptimizer()

    def setUp(self):
        """Set up each test - clear any caches."""
        # Clear any cached results between tests
        if hasattr(self.optimizer, '_analyze_complexity'):
            self.optimizer._analyze_complexity.cache_clear()

    def test_critical_performance_scenarios(self):
        """Comprehensive test of critical performance detection scenarios."""
        test_scenarios = [
            # Scenario 1: Nested Quadratic Loop - Critical Performance Issue
            {
                'code': textwrap.dedent("""
                def quadratic_search(items, target):
                    for i in range(len(items)):
                        for j in range(len(items)):
                            if items[i] == target and items[j] == target:
                                return True
                    return False
                """),
                'expected_bottlenecks': ['nested loop', 'O(n^2)'],
                'expected_suggestions': ['complexity', 'algorithm']
            },
            
            # Scenario 2: Recursive Fibonacci - Exponential Complexity
            {
                'code': textwrap.dedent("""
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                """),
                'expected_bottlenecks': ['recursive function', 'stack overflow'],
                'expected_suggestions': ['recursion', 'memoization']
            }
        ]

        for scenario in test_scenarios:
            tree = ast.parse(scenario['code'])
            
            # Bottleneck Detection
            bottlenecks = self.optimizer._detect_bottlenecks(tree)
            self.assertGreater(len(bottlenecks), 0, f"Failed to detect bottlenecks in scenario: {scenario['code']}")
            
            # Verify bottleneck types
            bottleneck_descriptions = [b['description'].lower() for b in bottlenecks]
            for expected in scenario['expected_bottlenecks']:
                self.assertTrue(
                    any(expected.lower() in desc for desc in bottleneck_descriptions), 
                    f"Missing expected bottleneck: {expected}"
                )
            
            # Optimization Suggestions
            suggestions = self.optimizer._generate_optimization_suggestions(tree)
            self.assertGreater(len(suggestions), 0, f"Failed to generate suggestions in scenario: {scenario['code']}")
            
            # Verify suggestion types
            suggestion_types = [s['type'].lower() for s in suggestions]
            suggestion_descriptions = [s['description'].lower() for s in suggestions]
            suggestion_texts = [s.get('suggestion', '').lower() for s in suggestions]
            
            for expected in scenario['expected_suggestions']:
                self.assertTrue(
                    any(expected.lower() in type_ for type_ in suggestion_types) or 
                    any(expected.lower() in desc for desc in suggestion_descriptions) or
                    any(expected.lower() in text for text in suggestion_texts), 
                    f"Missing expected suggestion type: {expected}"
                )

    def test_performance_overhead(self):
        """
        Ensure the performance analysis has reasonable overhead.
        Robust test with multiple scenarios and adaptive thresholds.
        """
        # Prepare test scenarios
        test_scenarios = [
            # Simple function
            """
def simple_function(x):
    return x * 2
            """,
            
            # Recursive function
            """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
            """,
            
            # Nested loop function
            """
def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            cell = 0
            for k in range(len(B)):
                cell += A[i][k] * B[k][j]
            row.append(cell)
        result.append(row)
    return result
            """
        ]
        
        # Performance measurement parameters
        num_iterations = 200
        base_overhead_threshold = 1000  # Base percentage threshold
        
        # Baseline time measurement
        baseline_times = []
        analysis_times = []
        
        for scenario in test_scenarios:
            # Parse the scenario
            tree = ast.parse(scenario)
            
            # Baseline time measurement with higher precision
            baseline_start = time.perf_counter_ns()
            for _ in range(num_iterations):
                pass  # Baseline empty loop
            baseline_end = time.perf_counter_ns()
            baseline_avg_time = (baseline_end - baseline_start) / num_iterations
            baseline_times.append(baseline_avg_time)
            
            # Performance analysis time measurement
            analysis_start = time.perf_counter_ns()
            for _ in range(num_iterations):
                self.optimizer._generate_optimization_suggestions(tree)
            analysis_end = time.perf_counter_ns()
            analysis_avg_time = (analysis_end - analysis_start) / num_iterations
            analysis_times.append(analysis_avg_time)
        
        # Compute overall performance metrics
        baseline_avg = max(baseline_times) if baseline_times else 1
        analysis_avg = max(analysis_times) if analysis_times else 1
        
        # Adaptive overhead calculation
        relative_overhead = (
            (analysis_avg - baseline_avg) / baseline_avg * 100 
            if baseline_avg > 0 else 0
        )
        
        # Dynamic overhead threshold based on baseline time
        # Smaller baseline times allow higher relative overhead
        adaptive_threshold = base_overhead_threshold * (1 + math.log(baseline_avg, 10))
        
        # Logging for debugging
        print(f"\nPerformance Analysis Details:")
        print(f"Baseline Average Time (ns): {baseline_avg:.2f}")
        print(f"Analysis Average Time (ns): {analysis_avg:.2f}")
        print(f"Relative Overhead: {relative_overhead:.2f}%")
        print(f"Adaptive Overhead Threshold: {adaptive_threshold:.2f}%")
        
        # Assertions with adaptive thresholds
        self.assertTrue(
            relative_overhead <= adaptive_threshold,
            f"Performance analysis overhead too high: {relative_overhead:.2f}% "
            f"(Threshold: {adaptive_threshold:.2f}%)"
        )

    def test_edge_case_robustness(self):
        """Test framework's robustness with various edge cases."""
        edge_cases = [
            "def empty_function(): pass",  # Empty function
            "x = lambda: None",  # Lambda function
            "class EmptyClass: pass"  # Empty class
        ]

        for case in edge_cases:
            try:
                tree = ast.parse(case)
                bottlenecks = self.optimizer._detect_bottlenecks(tree)
                suggestions = self.optimizer._generate_optimization_suggestions(tree)
                
                # Ensure no exceptions and reasonable output
                self.assertTrue(isinstance(bottlenecks, list))
                self.assertTrue(isinstance(suggestions, list))
                
                # Optionally, check that no bottlenecks or suggestions are generated for trivial cases
                self.assertEqual(len(bottlenecks), 0, f"Should not generate bottlenecks for case: {case}")
                self.assertEqual(len(suggestions), 0, f"Should not generate suggestions for case: {case}")
            except Exception as e:
                self.fail(f"Failed to handle edge case: {case}. Error: {e}")

if __name__ == '__main__':
    unittest.main()
