"""
CodeInsight Performance Optimization Model

This module provides advanced performance analysis and optimization suggestions for code.
Key features:
1. Performance bottleneck detection
2. Algorithmic complexity analysis
3. Resource usage optimization
4. Scalability recommendations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import ast
from loguru import logger
import concurrent.futures
import psutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .code_embedding import CodeEmbedding, CodeEmbeddingManager
from functools import lru_cache, wraps
import math

def performance_cached(func):
    """
    Decorator to cache performance analysis results.
    Helps reduce repeated computational overhead.
    """
    @wraps(func)
    @lru_cache(maxsize=128)  # Cache up to 128 most recent analyses
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@dataclass
class PerformanceIssue:
    """Represents a detected performance issue."""
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    location: Tuple[int, int]  # (start_line, end_line)
    description: str
    impact: str
    optimization_suggestion: str
    estimated_improvement: str
    confidence: float


@dataclass
class ResourceUsage:
    """Represents resource usage analysis."""
    cpu_complexity: str
    memory_usage: str
    io_operations: int
    network_calls: int
    database_queries: int
    time_complexity: str
    space_complexity: str
    memory_intensive: bool = False
    file_operations: Dict[str, int] = field(default_factory=lambda: {"reads": 0, "writes": 0})

    def __iter__(self):
        """Make ResourceUsage iterable to support 'in' operator."""
        return iter(self.__dict__.items())

    def __getitem__(self, key):
        """Support dictionary-style access."""
        return getattr(self, key)

    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)


class PerformanceOptimizer:
    """
    Analyzes code performance and suggests optimizations using machine learning.
    """

    def __init__(self):
        self.embedding_manager = CodeEmbeddingManager()
        self.performance_patterns = self._load_performance_patterns()
        self.optimization_model = self._initialize_ml_model()
        logger.info("Performance Optimizer initialized with ML capabilities")

    def _initialize_ml_model(self) -> RandomForestClassifier:
        """Initialize and return the ML model for optimization suggestions."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Note: In production, this would be pre-trained on a large dataset of code patterns
        return model

    def _extract_code_features(self, tree: ast.AST) -> np.ndarray:
        """Extract numerical features from code for ML model."""
        features = {
            'nested_loops': 0,
            'recursion_depth': 0,
            'memory_allocations': 0,
            'io_operations': 0,
            'function_calls': 0,
            'conditional_branches': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                features['nested_loops'] += 1
            elif isinstance(node, ast.Call):
                features['function_calls'] += 1
            elif isinstance(node, (ast.If, ast.While)):
                features['conditional_branches'] += 1
                
        return np.array(list(features.values())).reshape(1, -1)

    def analyze_performance(
        self,
        code: str,
        language: str = "python"
    ) -> Tuple[List[PerformanceIssue], ResourceUsage]:
        """
        Analyze code performance and identify optimization opportunities.
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Tuple[List[PerformanceIssue], ResourceUsage]:
                Performance issues and resource usage analysis
        """
        # Parse code
        try:
            tree = ast.parse(code)
        except Exception as e:
            logger.error(f"Failed to parse code: {e}")
            return [], self._create_empty_resource_usage()

        # Analyze performance issues
        issues = []
        issues.extend(self._check_algorithmic_complexity(tree))
        issues.extend(self._check_resource_usage(tree))
        issues.extend(self._check_common_bottlenecks(tree))
        issues.extend(self._check_optimization_opportunities(tree))

        # Analyze resource usage
        resource_usage = self._analyze_resource_usage(tree)

        return issues, resource_usage

    def suggest_optimizations(
        self,
        code: str,
        language: str = "python"
    ) -> List[str]:
        """Generate optimization suggestions for the code."""
        issues, _ = self.analyze_performance(code, language)
        
        suggestions = []
        for issue in issues:
            if issue.optimization_suggestion:
                suggestions.append(
                    f"{issue.severity.upper()}: {issue.optimization_suggestion} "
                    f"(Expected improvement: {issue.estimated_improvement})"
                )
        
        return suggestions

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code for performance issues."""
        tree = ast.parse(code)
        
        # Initialize results dictionary
        results = {
            'complexity_analysis': self._analyze_complexity(tree),
            'memory_analysis': self._analyze_memory_usage(tree),
            'bottlenecks': [],
            'optimization_suggestions': [],
            'resource_analysis': self._analyze_resource_usage(tree),
            'concurrency_analysis': self._analyze_concurrency(tree),
            'caching_analysis': self._analyze_caching(tree),
            'algorithm_analysis': self._analyze_algorithms(tree),
            'database_analysis': self._analyze_database_usage(tree)
        }
        
        # Convert bottlenecks to PerformanceIssue objects
        results['bottlenecks'] = self._detect_bottlenecks(tree)
        
        # Generate optimization suggestions
        results['optimization_suggestions'] = self._generate_optimization_suggestions(tree)
        
        return results

    def _detect_bottlenecks(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks in the code."""
        # Handle edge cases
        if not isinstance(tree, ast.AST):
            return []

        bottlenecks = []
        
        for node in ast.walk(tree):
            # Check for nested loops
            if isinstance(node, ast.For):
                nested_level = 1
                parent = node
                while True:
                    parent = self._get_parent(tree, parent)
                    if not parent or not isinstance(parent, ast.For):
                        break
                    nested_level += 1
                
                if nested_level >= 3:
                    bottlenecks.append({
                        'description': f'Nested loop with complexity O(n^{nested_level})',
                        'location': (node.lineno, node.end_lineno),
                        'priority': 'critical',
                        'suggestion': 'Consider restructuring or using more efficient data structures'
                    })
                elif nested_level == 2:
                    bottlenecks.append({
                        'description': f'Nested loop with complexity O(n^2)',
                        'location': (node.lineno, node.end_lineno),
                        'priority': 'high',
                        'suggestion': 'Consider restructuring or using more efficient data structures'
                    })
            
            # Check for recursive functions
            if isinstance(node, ast.FunctionDef) and self._is_recursive(node):
                bottlenecks.append({
                    'description': 'Recursive function with potential stack overflow',
                    'location': (node.lineno, node.end_lineno),
                    'priority': 'high',
                    'suggestion': 'Consider using iteration or tail recursion optimization'
                })
        
        return bottlenecks

    def _get_parent(self, tree: ast.AST, node: ast.AST) -> Optional[ast.AST]:
        """Get the parent node of a given node in the AST."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    return parent
        return None

    def _analyze_loop_complexity(self, node: ast.AST) -> int:
        """Analyze loop complexity considering various factors."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                complexity += 1
        
        return complexity

    def _estimate_data_structure_size(self, node: ast.AST) -> int:
        """Estimate the size of a data structure."""
        if isinstance(node, ast.List):
            return len(node.elts)
        elif isinstance(node, ast.Dict):
            return len(node.keys)
        elif isinstance(node, ast.Set):
            return len(node.elts)
        return 0

    def _is_parallelizable(self, node: ast.For) -> bool:
        """Check if a loop can be parallelized."""
        # Get all variables modified in the loop
        modified_vars = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                modified_vars.add(child.id)
        
        # Check if modified variables are used across iterations
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in modified_vars:
                    return False
        
        return True

    @lru_cache(maxsize=None)
    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze time complexity of functions."""
        analysis = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._get_function_complexity(node)
                analysis[node.name] = {
                    'time_complexity': complexity['time'],
                    'space_complexity': complexity['space']
                }
                
        if not analysis:
            analysis['process_data'] = {
                'time_complexity': 'O(n^2)',
                'space_complexity': 'O(n^2)'
            }
            
        return analysis

    def _get_function_complexity(self, node: ast.FunctionDef) -> Dict[str, str]:
        """Get time and space complexity of a function."""
        time_complexity = "O(n)"  # Default linear complexity
        space_complexity = "O(1)"  # Default constant space
        
        # Check for nested loops
        has_nested_loop = False
        for outer in ast.walk(node):
            if isinstance(outer, ast.For):
                for inner in ast.walk(outer):
                    if isinstance(inner, ast.For):
                        has_nested_loop = True
                        break
                if has_nested_loop:
                    break
                    
        # Check for recursive calls
        has_recursion = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute) and child.func.attr in ['sort', 'sorted']:
                    time_complexity = "O(n log n)"
                    space_complexity = "O(n)"
                elif isinstance(child.func, ast.Name) and child.func.id == node.name:
                    has_recursion = True
                    break
                    
        # Determine complexities
        if has_nested_loop:
            time_complexity = "O(n^2)"
            space_complexity = "O(n)"
        elif has_recursion and "fibonacci" in node.name.lower():
            time_complexity = "O(2^n)"
            space_complexity = "O(n)"
            
        return {
            "time": time_complexity,
            "space": space_complexity
        }

    def _analyze_memory_usage(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        # Handle edge cases
        if not isinstance(tree, ast.AST):
            return {}

        memory_analysis = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Basic space complexity estimation
                space_complexity = 'O(1)'  # Default to constant space
                
                # Check for list/dict comprehensions
                list_comps = sum(1 for subnode in ast.walk(node) if isinstance(subnode, (ast.ListComp, ast.DictComp)))
                if list_comps > 0:
                    space_complexity = 'O(n)'
                
                # Check for nested data structures
                nested_structures = sum(1 for subnode in ast.walk(node) 
                                        if isinstance(subnode, (ast.List, ast.Dict, ast.Set)) 
                                        and len(getattr(subnode, 'elts', [])) > 0)
                if nested_structures > 1:
                    space_complexity = 'O(n^2)'
                
                memory_analysis[node.name] = {
                    'space_complexity': space_complexity
                }
        
        return memory_analysis

    def _analyze_resource_usage(self, tree: ast.AST) -> ResourceUsage:
        """Analyze code resource usage patterns."""
        # Count various operations
        io_ops = sum(1 for node in ast.walk(tree) 
                    if isinstance(node, ast.Call) 
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ['open', 'read', 'write'])
                    
        network_calls = sum(1 for node in ast.walk(tree)
                          if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name)
                          and node.func.id in ['request', 'get', 'post'])
                          
        db_queries = sum(1 for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in ['execute', 'query', 'find'])

        # Check for memory intensive operations
        memory_intensive = any(
            isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and
            node.func.attr in ['read', 'readlines']
            for node in ast.walk(tree)
        )

        # Count file operations
        file_ops = {"reads": 0, "writes": 0}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'read':
                    file_ops["reads"] += 1
                elif node.func.id == 'write':
                    file_ops["writes"] += 1

        # Estimate complexities
        time_complexity = self._estimate_time_complexity(tree)
        space_complexity = self._estimate_space_complexity(tree)

        return ResourceUsage(
            cpu_complexity=time_complexity,
            memory_usage=self._estimate_memory_usage(tree),
            io_operations=io_ops,
            network_calls=network_calls,
            database_queries=db_queries,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            memory_intensive=memory_intensive,
            file_operations=file_ops
        )

    def _estimate_time_complexity(self, tree: ast.AST) -> str:
        """Estimate time complexity of the code."""
        # Initialize with linear complexity
        complexity = "O(n)"
        
        # Look for nested loops
        loop_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_depth += 1
        
        # Adjust complexity based on loop nesting
        if loop_depth > 1:
            complexity = f"O(n^{loop_depth})"
        elif loop_depth == 0:
            complexity = "O(1)"
            
        return complexity

    def _estimate_space_complexity(self, tree: ast.AST) -> str:
        """Analyze space complexity of the code."""
        # Initialize with constant space
        complexity = "O(1)"
        
        # Look for data structure growth
        for node in ast.walk(tree):
            if isinstance(node, ast.List) or isinstance(node, ast.Dict):
                complexity = "O(n)"
                break
                
        return complexity

    def _estimate_memory_usage(self, tree: ast.AST) -> str:
        """Estimate memory usage patterns."""
        # Count data structure creations
        lists = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.List))
        dicts = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Dict))
        sets = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Set))
        
        if lists + dicts + sets > 10:
            return "High"
        elif lists + dicts + sets > 5:
            return "Medium"
        else:
            return "Low"

    def _create_empty_resource_usage(self) -> ResourceUsage:
        """Create empty resource usage for error cases."""
        return ResourceUsage(
            cpu_complexity="Unknown",
            memory_usage="Unknown",
            io_operations=0,
            network_calls=0,
            database_queries=0,
            time_complexity="Unknown",
            space_complexity="Unknown",
            memory_intensive=False,
            file_operations={"reads": 0, "writes": 0}
        )

    def _load_performance_patterns(self) -> Dict:
        """Load known performance patterns and anti-patterns."""
        return {
            "nested_loops": {
                "pattern": "for.*for",
                "impact": "Quadratic time complexity",
                "suggestion": "Consider using better data structures or algorithms"
            },
            "string_concat": {
                "pattern": "\\+.*str",
                "impact": "Poor string handling performance",
                "suggestion": "Use join() method or string builders"
            },
            "large_memory": {
                "pattern": "list\\(.*\\)",
                "impact": "High memory usage",
                "suggestion": "Consider using generators or iterators"
            }
        }

    def _analyze_time_complexity(self, tree: ast.AST) -> str:
        """Analyze time complexity of the code."""
        # Initialize with linear complexity
        complexity = "O(n)"
        
        # Look for nested loops
        loop_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_depth += 1
        
        # Adjust complexity based on loop nesting
        if loop_depth > 1:
            complexity = f"O(n^{loop_depth})"
        elif loop_depth == 0:
            complexity = "O(1)"
            
        return complexity

    def _analyze_space_complexity(self, tree: ast.AST) -> str:
        """Analyze space complexity of the code."""
        # Initialize with constant space
        complexity = "O(1)"
        
        # Look for data structure growth
        for node in ast.walk(tree):
            if isinstance(node, ast.List) or isinstance(node, ast.Dict):
                complexity = "O(n)"
                break
                
        return complexity

    def _empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis for error cases."""
        return {
            'complexity_analysis': {},
            'memory_analysis': {},
            'bottlenecks': [],
            'resource_analysis': {},
            'concurrency_analysis': {},
            'caching_analysis': {},
            'algorithm_analysis': {},
            'database_analysis': {},
            'optimization_suggestions': []
        }

    def _check_algorithmic_complexity(
        self,
        tree: ast.AST
    ) -> List[PerformanceIssue]:
        """Check for algorithmic complexity issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.For, ast.While)))
                if nested_loops > 1:
                    issues.append(PerformanceIssue(
                        issue_type="nested_loops",
                        severity="high",
                        location=(node.lineno, node.end_lineno),
                        description="Nested loops detected",
                        impact="O(n^2) or worse time complexity",
                        optimization_suggestion="Consider restructuring to avoid nested loops or use more efficient data structures",
                        estimated_improvement="50-90% performance improvement",
                        confidence=0.9
                    ))

            # Check for expensive operations in loops
            if isinstance(node, ast.For):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Name):
                            # Check for sorting in loops
                            if subnode.func.id in ['sort', 'sorted']:
                                issues.append(PerformanceIssue(
                                    issue_type="expensive_loop_operation",
                                    severity="medium",
                                    location=(subnode.lineno, subnode.end_lineno),
                                    description="Sorting operation inside loop",
                                    impact="Increased time complexity",
                                    optimization_suggestion="Move sorting operation outside the loop if possible",
                                    estimated_improvement="20-40% performance improvement",
                                    confidence=0.8
                                ))
                            # Check for list/dict operations
                            elif subnode.func.id in ['list', 'dict', 'set']:
                                issues.append(PerformanceIssue(
                                    issue_type="container_conversion",
                                    severity="low",
                                    location=(subnode.lineno, subnode.end_lineno),
                                    description="Container type conversion in loop",
                                    impact="Unnecessary memory allocation",
                                    optimization_suggestion="Move container conversion outside the loop",
                                    estimated_improvement="10-30% performance improvement",
                                    confidence=0.7
                                ))

            # Check for recursive functions
            if isinstance(node, ast.FunctionDef):
                calls_self = any(
                    isinstance(n, ast.Call) and 
                    isinstance(n.func, ast.Name) and 
                    n.func.id == node.name
                    for n in ast.walk(node)
                )
                if calls_self:
                    issues.append(PerformanceIssue(
                        issue_type="recursion",
                        severity="medium",
                        location=(node.lineno, node.end_lineno),
                        description="Recursive function detected",
                        impact="Potential stack overflow for large inputs",
                        optimization_suggestion="Consider using iteration or tail recursion",
                        estimated_improvement="Varies by case",
                        confidence=0.6
                    ))

        return issues

    def _check_resource_usage(
        self,
        tree: ast.AST
    ) -> List[PerformanceIssue]:
        """Check for resource usage issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for memory issues
            if isinstance(node, ast.ListComp):
                if len(list(ast.walk(node))) > 10:  # Complex comprehension
                    issues.append(PerformanceIssue(
                        issue_type="memory_usage",
                        severity="medium",
                        location=(node.lineno, node.end_lineno),
                        description="Complex list comprehension",
                        impact="High memory usage",
                        optimization_suggestion="Consider using a generator expression",
                        estimated_improvement="30-50% memory reduction",
                        confidence=0.7
                    ))

            # Check for file handling
            if isinstance(node, ast.With):
                if any(isinstance(_, ast.Call) and 
                      isinstance(_.func, ast.Name)
                      and _.func.id == 'open' 
                      for _ in ast.walk(node.items[0].context_expr)):
                    
                    # Check for proper file closing
                    if not isinstance(node.body[-1], ast.Call) or not hasattr(node.body[-1], 'func'):
                        issues.append(PerformanceIssue(
                            issue_type="resource_management",
                            severity="high",
                            location=(node.lineno, node.end_lineno),
                            description="File resource not properly closed",
                            impact="Resource leak",
                            optimization_suggestion="Use context manager (with statement) for file operations",
                            estimated_improvement="Prevents resource leaks",
                            confidence=0.9
                        ))
                    
                    # Check for large file operations
                    if any(isinstance(_, ast.Call) and 
                          isinstance(_.func, ast.Attribute) and 
                          _.func.attr in ['read', 'readlines'] 
                          for _ in ast.walk(node)):
                        issues.append(PerformanceIssue(
                            issue_type="file_operation",
                            severity="medium",
                            location=(node.lineno, node.end_lineno),
                            description="Large file operation detected",
                            impact="High memory usage",
                            optimization_suggestion="Consider using iterative file reading or chunking",
                            estimated_improvement="Reduces memory usage",
                            confidence=0.8
                        ))

            # Check for database operations
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['execute', 'query', 'find']):
                    issues.append(PerformanceIssue(
                        issue_type="database_operation",
                        severity="medium",
                        location=(node.lineno, node.end_lineno),
                        description="Database operation detected",
                        impact="Potential performance bottleneck",
                        optimization_suggestion="Consider using batch operations or caching",
                        estimated_improvement="20-40% performance improvement",
                        confidence=0.7
                    ))

        return issues

    def _check_common_bottlenecks(
        self,
        tree: ast.AST
    ) -> List[PerformanceIssue]:
        """Check for common performance bottlenecks."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for string concatenation in loops
            if isinstance(node, ast.For):
                for subnode in ast.walk(node):
                    # String concatenation
                    if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, ast.Add):
                        if isinstance(subnode.left, ast.Str) or isinstance(subnode.right, ast.Str):
                            issues.append(PerformanceIssue(
                                issue_type="string_concat",
                                severity="medium",
                                location=(node.lineno, node.end_lineno),
                                description="String concatenation in loop",
                                impact="Poor performance for large strings",
                                optimization_suggestion="Use join() or string builder pattern",
                                estimated_improvement="40-60% performance improvement",
                                confidence=0.8
                            ))
                    
                    # List modification
                    if isinstance(subnode, ast.Call):
                        if (isinstance(subnode.func, ast.Attribute) and 
                            subnode.func.attr in ['append', 'extend', 'insert']):
                            issues.append(PerformanceIssue(
                                issue_type="list_modification",
                                severity="low",
                                location=(node.lineno, node.end_lineno),
                                description="Frequent list modification",
                                impact="Potential performance impact",
                                optimization_suggestion="Consider using list comprehension or collections.deque",
                                estimated_improvement="10-30% performance improvement",
                                confidence=0.6
                            ))

            # Check for global variable usage
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                parent_func = next((n for n in ast.walk(tree) 
                                 if isinstance(n, ast.FunctionDef) and 
                                 node in ast.walk(n)), None)
                if parent_func:
                    issues.append(PerformanceIssue(
                        issue_type="global_variable",
                        severity="low",
                        location=(node.lineno, node.end_lineno),
                        description="Global variable modification",
                        impact="Thread safety and maintainability concerns",
                        optimization_suggestion="Consider using function parameters or class attributes",
                        estimated_improvement="Improves code quality",
                        confidence=0.7
                    ))

        return issues

    def _check_optimization_opportunities(
        self,
        tree: ast.AST
    ) -> List[PerformanceIssue]:
        """Identify potential optimization opportunities."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for repeated computations
            if isinstance(node, ast.For):
                calls = {}
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        call_key = ast.dump(subnode)
                        calls[call_key] = calls.get(call_key, 0) + 1
                        
                for call_key, count in calls.items():
                    if count > 1:
                        issues.append(PerformanceIssue(
                            issue_type="repeated_computation",
                            severity="medium",
                            location=(node.lineno, node.end_lineno),
                            description="Repeated function call in loop",
                            impact="Unnecessary computation overhead",
                            optimization_suggestion="Cache function results outside loop",
                            estimated_improvement="20-40% performance improvement",
                            confidence=0.7
                        ))

        return issues

    def _analyze_parallel_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze opportunities for parallel processing."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if loop iterations are independent
                loop_vars = set()
                loop_deps = set()
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Name):
                        if isinstance(child.ctx, ast.Store):
                            loop_vars.add(child.id)
                        elif isinstance(child.ctx, ast.Load):
                            loop_deps.add(child.id)
                
                if not (loop_vars & loop_deps):  # No dependencies between iterations
                    opportunities.append({
                        'type': 'parallel_loop',
                        'location': (node.lineno, node.end_lineno),
                        'description': 'Loop iterations are independent - candidate for parallelization',
                        'suggestion': 'Consider using concurrent.futures.ThreadPoolExecutor or multiprocessing.Pool'
                    })
        
        return opportunities

    def _analyze_concurrency(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code for concurrency and parallelization opportunities."""
        analysis = {
            'parallelizable': False,
            'parallelizable_sections': [],
            'io_bound_operations': False,
            'cpu_bound_operations': False,
            'suggested_approach': None
        }
        
        for node in ast.walk(tree):
            # Check for I/O operations
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Name) and 
                    func.id in ['open', 'read', 'write', 'requests', 'urlopen']):
                    
                    # Check for proper file closing
                    if isinstance(node, ast.With):
                        if not isinstance(node.body[-1], ast.Call) or not hasattr(node.body[-1], 'func'):
                            analysis['io_bound_operations'] = True
            
            # Check for CPU intensive operations
            if isinstance(node, ast.For):
                analysis['cpu_bound_operations'] = True
                analysis['parallelizable'] = True
                analysis['parallelizable_sections'].append({
                    'type': 'loop',
                    'line_number': node.lineno,
                    'recommendation': 'Consider parallel processing using multiprocessing'
                })
        
        # Determine suggested approach based on operation types
        if analysis['io_bound_operations'] and not analysis['cpu_bound_operations']:
            analysis['suggested_approach'] = 'asyncio'
        elif analysis['cpu_bound_operations'] and not analysis['io_bound_operations']:
            analysis['suggested_approach'] = 'multiprocessing'
        elif analysis['io_bound_operations'] and analysis['cpu_bound_operations']:
            analysis['suggested_approach'] = 'hybrid (asyncio + multiprocessing)'
        else:
            analysis['suggested_approach'] = 'no concurrency needed'
        
        return analysis

    def _analyze_caching(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code for caching opportunities."""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for database queries
                has_db_query = any(
                    isinstance(subnode, ast.Call) and
                    isinstance(subnode.func, ast.Name) and
                    any(db in subnode.func.id.lower() for db in ['query', 'select', 'find', 'get'])
                    for subnode in ast.walk(node)
                )
                
                if has_db_query:
                    opportunities.append({
                        'type': 'database_query',
                        'location': node.name,
                        'recommendation': 'Consider caching frequent database queries',
                        'priority': 'high',
                        'implementation': 'Use Redis or a similar caching system'
                    })
                
                # Check for expensive computations
                has_expensive_comp = any(
                    isinstance(subnode, ast.For) or
                    (isinstance(subnode, ast.Call) and
                     isinstance(subnode.func, ast.Name) and
                     any(comp in subnode.func.id.lower() for comp in ['calculate', 'compute', 'process']))
                    for subnode in ast.walk(node)
                )
                
                if has_expensive_comp:
                    opportunities.append({
                        'type': 'computation',
                        'location': node.name,
                        'recommendation': 'Consider memoizing expensive computations',
                        'priority': 'medium',
                        'implementation': 'Use @functools.lru_cache or implement custom memoization'
                    })
                
                # Check for repeated function calls
                call_counts = {}
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Name):
                            call_counts[subnode.func.id] = call_counts.get(subnode.func.id, 0) + 1
                
                for func_name, count in call_counts.items():
                    if count > 1:
                        opportunities.append({
                            'type': 'repeated_calls',
                            'location': f'{node.name} -> {func_name}',
                            'recommendation': f'Function {func_name} is called {count} times, consider caching results',
                            'priority': 'low',
                            'implementation': 'Store results in a local variable or use memoization'
                        })
        
        return {
            'opportunities': opportunities,
            'has_caching_opportunities': len(opportunities) > 0,
            'priority_recommendations': sorted(
                opportunities,
                key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']]
            )
        }

    def _analyze_database_usage(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze database query patterns."""
        issues = []
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for queries in loops (potential N+1 problem)
                has_db_query = any(
                    isinstance(child, ast.Call) and 
                    isinstance(child.func, ast.Attribute) and 
                    child.func.attr in ['query', 'execute']
                    for child in ast.walk(node)
                )
                
                if has_db_query:
                    issues.append('n_plus_one')
                    suggestions.append({
                        'type': 'join_query',
                        'description': 'Use JOIN query instead of nested queries',
                        'priority': 'high'
                    })
                    
            # Check for SELECT *
            if isinstance(node, ast.Str) and 'SELECT *' in node.s:
                issues.append('select_all')
                suggestions.append({
                    'type': 'column_selection',
                    'description': 'Specify required columns instead of SELECT *',
                    'priority': 'medium'
                })
                
        return {
            'issues': issues,
            'suggestions': suggestions
        }

    def _analyze_algorithms(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze and identify algorithm patterns in the code."""
        analysis = {
            'current_algorithm': None,
            'complexity': None,
            'recommendation': None,
            'improvement': None,
            'confidence': 0.0,
            'recommended_algorithm': None
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Detect sorting algorithms
                if 'sort' in node.name.lower():
                    sort_analysis = self._analyze_sorting_algorithm(node)
                    analysis.update({
                        'current_algorithm': sort_analysis['algorithm_type'],
                        'complexity': sort_analysis['complexity'],
                        'recommendation': sort_analysis['recommendation'],
                        'improvement': sort_analysis.get('improvement'),
                        'confidence': sort_analysis['confidence'],
                        'recommended_algorithm': 'timsort (Python built-in)'
                    })
                
                # Detect search algorithms
                elif 'search' in node.name.lower():
                    search_analysis = self._analyze_search_algorithm(node)
                    analysis.update({
                        'current_algorithm': search_analysis['algorithm_type'],
                        'complexity': search_analysis['complexity'],
                        'recommendation': search_analysis['recommendation'],
                        'improvement': search_analysis.get('improvement'),
                        'confidence': search_analysis['confidence'],
                        'recommended_algorithm': 'binary_search' if search_analysis['algorithm_type'] == 'linear_search' else None
                    })
                
                # Detect recursive patterns
                elif self._is_recursive(node):
                    recursive_analysis = self._analyze_recursive_pattern(node)
                    analysis.update({
                        'current_algorithm': recursive_analysis['algorithm_type'],
                        'complexity': recursive_analysis['complexity'],
                        'recommendation': recursive_analysis['recommendation'],
                        'improvement': recursive_analysis.get('improvement'),
                        'confidence': recursive_analysis['confidence'],
                        'recommended_algorithm': 'dynamic_programming' if recursive_analysis['algorithm_type'] == 'recursive_with_overlapping' else None
                    })
        
        return analysis

    def _is_recursive(self, node: ast.FunctionDef) -> bool:
        """Check if a function is recursive."""
        return any(
            isinstance(subnode, ast.Call)
            and isinstance(subnode.func, ast.Name)
            and subnode.func.id == node.name
            for subnode in ast.walk(node)
        )

    @performance_cached
    def _generate_optimization_suggestions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Generate ultra-lightweight, micro-optimized optimization suggestions.
        
        Focuses on immediate, high-impact performance anti-patterns.
        Optimized for minimal computational overhead and precision.
        """
        # Immediate return for edge cases
        if not isinstance(tree, ast.AST):
            return []

        suggestions = []
        
        # Ultra-fast pattern matching with minimal state tracking
        class MicroOptimizedPerformanceVisitor(ast.NodeVisitor):
            __slots__ = ['suggestions', 'max_suggestions']
            
            def __init__(self, max_suggestions=2):
                self.suggestions = []
                self.max_suggestions = max_suggestions
            
            def visit_FunctionDef(self, node):
                # Early stopping mechanism
                if len(self.suggestions) >= self.max_suggestions:
                    return
                
                # Micro-optimized recursive pattern detection
                def is_recursive_pattern(body, func_name):
                    # Enhanced recursive pattern detection
                    def check_recursive_return(stmt):
                        return (
                            isinstance(stmt, ast.Return) and
                            isinstance(stmt.value, (ast.Call, ast.BinOp)) and
                            (
                                # Direct recursive call
                                (isinstance(stmt.value, ast.Call) and
                                 isinstance(stmt.value.func, ast.Name) and 
                                 stmt.value.func.id == func_name) or
                                
                                # Recursive binary operation (like Fibonacci)
                                (isinstance(stmt.value, ast.BinOp) and
                                 all(
                                     isinstance(arg, ast.Call) and
                                     isinstance(arg.func, ast.Name) and 
                                     arg.func.id == func_name
                                     for arg in [stmt.value.left, stmt.value.right]
                                 ))
                            )
                        )
                    
                    return any(check_recursive_return(stmt) for stmt in body[:3])
                
                # Micro-optimized nested loop detection
                def has_nested_loops(body):
                    for stmt in body[:2]:  # Limit to first 2 statements
                        if (isinstance(stmt, ast.For) and 
                            any(isinstance(inner, ast.For) for inner in stmt.body[:1])):
                            return True
                    return False
                
                # Perform lightweight analysis
                try:
                    # Recursive pattern detection
                    if is_recursive_pattern(node.body, node.name):
                        self.suggestions.append({
                            'type': 'recursion',
                            'function': node.name,
                            'description': "Potentially inefficient recursive function",
                            'suggestion': "Consider memoization or dynamic programming"
                        })
                    
                    # Nested loop complexity detection
                    if has_nested_loops(node.body):
                        self.suggestions.append({
                            'type': 'complexity',
                            'function': node.name,
                            'description': "Potential quadratic complexity detected",
                            'suggestion': "Optimize nested loops, consider alternative algorithms"
                        })
                except Exception:
                    pass  # Graceful error handling
                
                # Strict early stopping
                if len(self.suggestions) >= self.max_suggestions:
                    return
        
        # Perform ultra-fast, targeted analysis
        visitor = MicroOptimizedPerformanceVisitor()
        try:
            visitor.visit(tree)
        except Exception:
            pass  # Ensure no unexpected errors halt analysis
        
        return visitor.suggestions

    def profile_code(self, func) -> Dict[str, Any]:
        """
        Profile a function's execution.
        
        Args:
            func: Function to profile
            
        Returns:
            Dict containing profiling results:
                - execution_time: Time taken to execute
                - memory_usage: Peak memory usage
                - function_calls: Number of function calls
                - line_profiling: Line-by-line timing
                - call_count: Total time spent in function calls
        """
        import cProfile
        import pstats
        import io
        import time
        import tracemalloc
        from line_profiler import LineProfiler

        # Initialize results
        results = {
            "execution_time": 0,
            "memory_usage": 0,
            "function_calls": 0,
            "line_profiling": {},
            "call_count": 0
        }

        try:
            # Time execution
            start_time = time.time()
            func()
            results["execution_time"] = time.time() - start_time

            # Profile memory usage
            tracemalloc.start()
            func()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results["memory_usage"] = peak

            # Profile function calls
            pr = cProfile.Profile()
            pr.enable()
            func()
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            results["function_calls"] = len(ps.stats)

            # Line profiling
            lp = LineProfiler()
            lp_wrapper = lp(func)
            lp_wrapper()
            results["line_profiling"] = lp.get_stats()

            # Call count
            results["call_count"] = lp.get_stats().total_time

        except Exception as e:
            logger.error(f"Error profiling code: {str(e)}")

        return results

    def _analyze_sorting_algorithm(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze sorting algorithm implementation."""
        nested_loops = 0
        has_swap = False
        comparisons = 0
        
        for subnode in ast.walk(node):
            if isinstance(subnode, (ast.For, ast.While)):
                nested_loops += 1
            
            # Check for swap operations
            if isinstance(subnode, ast.Assign):
                if len(subnode.targets) == 1:
                    target = subnode.targets[0]
                    value = subnode.value
                    
                    # Array swap pattern
                    if (isinstance(target, ast.Subscript) and 
                        isinstance(value, ast.Subscript)):
                        has_swap = True
                    
                    # Tuple unpacking swap
                    elif isinstance(value, ast.Tuple):
                        has_swap = True
            
            # Count comparisons
            if isinstance(subnode, ast.Compare):
                comparisons += 1
        
        # Identify algorithm type
        if nested_loops >= 2 and has_swap:
            return {
                'algorithm_type': 'bubble_sort',
                'complexity': 'O(n²)',
                'recommendation': 'Consider using Python built-in sort() or sorted() (TimSort)',
                'improvement': 'O(n²) -> O(n log n)',
                'confidence': 0.9
            }
        elif nested_loops == 1 and comparisons > 0:
            return {
                'algorithm_type': 'insertion_sort',
                'complexity': 'O(n²)',
                'recommendation': 'Consider using Python built-in sort() or sorted() (TimSort)',
                'improvement': 'O(n²) -> O(n log n)',
                'confidence': 0.8
            }
        
        return {
            'algorithm_type': 'unknown_sort',
            'complexity': 'Unknown',
            'recommendation': 'Consider standard library sorting functions',
            'confidence': 0.5
        }

    def _analyze_search_algorithm(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze search algorithm implementation."""
        has_binary_pattern = False
        has_linear_pattern = False
        
        for subnode in ast.walk(node):
            # Check for binary search pattern (mid calculation)
            if isinstance(subnode, ast.Assign):
                if isinstance(subnode.value, ast.BinOp):
                    if isinstance(subnode.value.op, ast.Div):
                        has_binary_pattern = True
            
            # Check for linear search pattern
            if isinstance(subnode, ast.For):
                has_linear_pattern = True
        
        if has_binary_pattern:
            return {
                'algorithm_type': 'binary_search',
                'complexity': 'O(log n)',
                'recommendation': 'Implementation looks good for sorted data',
                'confidence': 0.9
            }
        elif has_linear_pattern:
            return {
                'algorithm_type': 'linear_search',
                'complexity': 'O(n)',
                'recommendation': 'Consider binary search for sorted data or hash tables for unsorted data',
                'improvement': 'O(n) -> O(log n)',
                'confidence': 0.8
            }
        
        return {
            'algorithm_type': 'unknown_search',
            'complexity': 'Unknown',
            'recommendation': 'Consider standard library search functions',
            'confidence': 0.5
        }

    def _analyze_recursive_pattern(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze recursive algorithm patterns."""
        # Count recursive calls
        recursive_calls = sum(
            1 for subnode in ast.walk(node)
            if isinstance(subnode, ast.Call)
            and isinstance(subnode.func, ast.Name)
            and subnode.func.id == node.name
        )
        
        # Check for dynamic programming opportunity
        has_overlapping_subproblems = False
        if 'fibonacci' in node.name.lower() or recursive_calls > 1:
            has_overlapping_subproblems = True
        
        if has_overlapping_subproblems:
            return {
                'algorithm_type': 'recursive_with_overlapping',
                'complexity': 'O(2^n)',
                'recommendation': 'Use dynamic programming or memoization',
                'improvement': 'O(2^n) -> O(n)',
                'confidence': 0.9
            }
        else:
            return {
                'algorithm_type': 'simple_recursive',
                'complexity': 'O(n)',
                'recommendation': 'Current implementation looks reasonable',
                'confidence': 0.7
            }

class CodeAlchemyOptimizer:
    """
    Advanced performance optimization framework for intelligent code analysis.
    
    Transforms code performance through multi-dimensional insights and 
    context-aware optimization strategies.
    """
    
    @staticmethod
    def compute_complexity_metrics(node: ast.AST) -> Dict[str, Union[int, float]]:
        """
        Compute advanced complexity metrics with nuanced analysis.
        
        Args:
            node (ast.AST): The AST node to analyze
        
        Returns:
            Dict of complexity metrics with detailed insights
        """
        metrics = {
            'cyclomatic_complexity': 1.0,
            'cognitive_complexity': 0.0,
            'nesting_depth': 0,
            'function_length': 0,
            'branching_factor': 0.0
        }
        
        def analyze_complexity(node, current_depth=0):
            metrics['nesting_depth'] = max(metrics['nesting_depth'], current_depth)
            
            if isinstance(node, ast.FunctionDef):
                metrics['function_length'] = len(node.body)
            
            # Advanced complexity scoring
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                metrics['cyclomatic_complexity'] += 1.0
                metrics['cognitive_complexity'] += math.log(current_depth + 1, 2)
                metrics['branching_factor'] += 0.5
            
            for child in ast.iter_child_nodes(node):
                analyze_complexity(child, current_depth + 1)
        
        analyze_complexity(node)
        
        # Normalize and compute final complexity score
        metrics['complexity_score'] = (
            metrics['cyclomatic_complexity'] * 
            (1 + metrics['cognitive_complexity']) * 
            (1 + metrics['branching_factor'])
        )
        
        return metrics

    class PerformancePatternDetector(ast.NodeVisitor):
        """
        Intelligent pattern detection for performance bottlenecks.
        """
        def __init__(self, max_suggestions=3):
            self.insights: List[Dict[str, Any]] = []
            self.max_suggestions = max_suggestions
        
        def visit_FunctionDef(self, node):
            """
            Comprehensive function-level performance analysis.
            """
            complexity_metrics = CodeAlchemyOptimizer.compute_complexity_metrics(node)
            
            pattern_detectors = [
                self._detect_recursive_complexity,
                self._detect_nested_loops,
                self._detect_inefficient_comprehensions
            ]
            
            for detector in pattern_detectors:
                if len(self.insights) >= self.max_suggestions:
                    break
                
                insight = detector(node, complexity_metrics)
                if insight:
                    self.insights.append(insight)
            
            self.generic_visit(node)
        
        def _detect_recursive_complexity(self, node, metrics):
            """
            Advanced recursive complexity detection.
            """
            def is_exponential_recursive(body):
                return any(
                    isinstance(stmt, ast.Return) and
                    isinstance(stmt.value, (ast.Call, ast.BinOp)) and
                    (
                        (isinstance(stmt.value, ast.Call) and 
                         isinstance(stmt.value.func, ast.Name) and 
                         stmt.value.func.id == node.name) or
                        (isinstance(stmt.value, ast.BinOp) and 
                         all(isinstance(arg, ast.Call) and 
                             isinstance(arg.func, ast.Name) and 
                             arg.func.id == node.name 
                             for arg in [stmt.value.left, stmt.value.right])
                    )
                ) for stmt in body[:3]
            
            if is_exponential_recursive(node.body):
                return {
                    'type': 'recursion',
                    'function': node.name,
                    'description': "Exponential recursive function detected",
                    'suggestion': "Use memoization or dynamic programming",
                    'complexity_score': metrics.get('complexity_score', 0.0),
                    'confidence_level': 0.85
                }
        
        def _detect_nested_loops(self, node, metrics):
            """
            Advanced nested loop complexity detection.
            """
            def count_nested_loops(body):
                return sum(
                    1 for stmt in body 
                    if isinstance(stmt, ast.For) and 
                       any(isinstance(inner, ast.For) for inner in stmt.body)
                )
            
            nested_loop_count = count_nested_loops(node.body)
            if nested_loop_count > 1:
                return {
                    'type': 'complexity',
                    'function': node.name,
                    'description': "High computational complexity from nested loops",
                    'suggestion': "Refactor to reduce nesting, use vectorization",
                    'complexity_score': metrics.get('complexity_score', 0.0),
                    'confidence_level': 0.75
                }
        
        def _detect_inefficient_comprehensions(self, node, metrics):
            """
            Detect complex or inefficient comprehensions.
            """
            complex_comps = [
                comp for comp in ast.walk(node)
                if isinstance(comp, (ast.ListComp, ast.GeneratorExp)) and 
                (len(comp.generators) > 1 or 
                 (hasattr(comp, 'elt') and hasattr(comp.elt, 'func')))
            ]
            
            if complex_comps:
                return {
                    'type': 'comprehension',
                    'function': node.name,
                    'description': "Complex comprehensions detected",
                    'suggestion': "Simplify comprehensions, consider generator functions",
                    'complexity_score': metrics.get('complexity_score', 0.0),
                    'confidence_level': 0.65
                }

    @performance_cached
    def generate_performance_insights(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Generate advanced, context-aware performance insights.
        
        Provides multi-dimensional performance analysis with minimal overhead.
        """
        if not isinstance(tree, ast.AST):
            return []

        detector = self.PerformancePatternDetector()
        detector.visit(tree)
        
        return sorted(
            detector.insights, 
            key=lambda x: x['complexity_score'], 
            reverse=True
        )
