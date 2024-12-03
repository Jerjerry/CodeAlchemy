"""
Tests for the Performance Optimization component.
"""

import pytest
import time
from unittest.mock import Mock, patch

from ai.models.performance_optimization import PerformanceOptimizer

# Test data
INEFFICIENT_CODE = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_data(data):
    results = []
    for i in range(len(data)):
        for j in range(len(data)):
            results.append(data[i] * data[j])
    return results
"""

EFFICIENT_CODE = """
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def process_data(data):
    seen = set()
    result = []
    for num in data:
        if -num in seen:
            result.append((num, -num))
        seen.add(num)
    return result

def search_items(items, target):
    return target in set(items)

class DataProcessor:
    def __init__(self):
        self.data = []
        
    def process_all(self):
        self.data = [item * 2 for item in self.data]
"""

@pytest.fixture
def performance_model():
    """Create a PerformanceOptimizer instance for testing."""
    return PerformanceOptimizer()

def test_analyze_time_complexity(performance_model):
    """Test time complexity analysis."""
    # Analyze inefficient code
    results = performance_model.analyze(INEFFICIENT_CODE)
    
    assert 'complexity_analysis' in results
    analysis = results['complexity_analysis']
    
    assert analysis['fibonacci']['time_complexity'] == 'O(2^n)'
    assert analysis['process_data']['time_complexity'] == 'O(n^2)'

def test_detect_performance_bottlenecks(performance_model):
    """Test performance bottleneck detection."""
    results = performance_model.analyze(INEFFICIENT_CODE)
    
    assert 'bottlenecks' in results
    bottlenecks = results['bottlenecks']
    
    assert len(bottlenecks) >= 3
    assert any('exponential complexity' in b['description'].lower() 
              for b in bottlenecks)
    assert any('nested loops' in b['description'].lower() 
              for b in bottlenecks)

def test_memory_usage_analysis(performance_model):
    """Test memory usage analysis."""
    # Analyze inefficient code
    results = performance_model.analyze(INEFFICIENT_CODE)
    
    assert 'memory_analysis' in results
    analysis = results['memory_analysis']
    
    assert analysis['fibonacci']['space_complexity'] == 'O(n)'  # recursion stack
    assert analysis['process_data']['space_complexity'] == 'O(n^2)'  # result list

def test_optimization_suggestions(performance_model):
    """Test optimization suggestion generation."""
    results = performance_model.analyze(INEFFICIENT_CODE)
    
    assert 'optimization_suggestions' in results
    suggestions = results['optimization_suggestions']
    
    assert len(suggestions) >= 3
    assert any('dynamic programming' in s['description'].lower() 
              for s in suggestions)

def test_resource_utilization(performance_model):
    """Test resource utilization analysis."""
    code = """
def process_large_file(filename):
    with open(filename, 'r') as f:
        data = f.read()  # Loads entire file into memory
    return len(data)
"""
    
    results = performance_model.analyze(code)
    
    assert 'resource_analysis' in results
    analysis = results['resource_analysis']
    
    assert 'memory_intensive' in analysis
    assert 'file_operations' in analysis

def test_concurrency_analysis(performance_model):
    """Test concurrency and threading analysis."""
    code = """
def process_items(items):
    results = []
    for item in items:
        results.append(expensive_operation(item))
    return results
"""
    
    results = performance_model.analyze(code)
    
    assert 'concurrency_analysis' in results
    analysis = results['concurrency_analysis']
    
    assert 'parallelizable' in analysis
    assert 'suggested_approach' in analysis

def test_caching_opportunities(performance_model):
    """Test detection of caching opportunities."""
    code = """
def get_user_data(user_id):
    return database.query(f"SELECT * FROM users WHERE id = {user_id}")
"""
    
    results = performance_model.analyze(code)
    
    assert 'caching_analysis' in results
    analysis = results['caching_analysis']
    
    assert len(analysis['opportunities']) > 0
    assert any('memoization' in opp['suggestion'].lower() 
              for opp in analysis['opportunities'])

def test_database_query_analysis(performance_model):
    """Test database query performance analysis."""
    code = """
def get_user_posts(user_id):
    posts = db.query("SELECT * FROM posts WHERE user_id = %s", user_id)
    for post in posts:
        post.comments = db.query("SELECT * FROM comments WHERE post_id = %s", post.id)
    return posts
"""
    
    results = performance_model.analyze(code)
    
    assert 'database_analysis' in results
    analysis = results['database_analysis']
    
    assert 'n_plus_one' in analysis['issues']
    assert analysis['suggestions'][0]['type'] == 'join_query'

def test_algorithm_selection(performance_model):
    """Test algorithm selection recommendations."""
    code = """
def sort_data(data):
    # Bubble sort implementation
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
    return data
"""
    
    results = performance_model.analyze(code)
    
    assert 'algorithm_analysis' in results
    analysis = results['algorithm_analysis']
    
    assert analysis['current_algorithm'] == 'bubble_sort'
    assert 'timsort' in analysis['recommended_algorithm'].lower()

def test_code_profiling(performance_model):
    """Test code profiling functionality."""
    def slow_function():
        time.sleep(0.1)
        return True
    
    with patch('time.sleep'):  # Mock sleep for faster tests
        profile_results = performance_model.profile_code(slow_function)
    
    assert 'execution_time' in profile_results
    assert 'call_count' in profile_results
    assert 'memory_usage' in profile_results
