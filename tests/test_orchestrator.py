"""
Tests for the AI Orchestrator component.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from ai.orchestrator import AIOrchestrator
from core.parser.python_parser import PythonParser

# Test data
SAMPLE_CODE = """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data: list) -> dict:
    '''Process data with potential security issues.'''
    result = {}
    query = f"SELECT * FROM users WHERE id = {data[0]}"  # SQL injection vulnerability
    result['data'] = expensive_operation(data)  # Performance issue
    return result
"""

@pytest.fixture
def orchestrator():
    """Create an AIOrchestrator instance for testing."""
    return AIOrchestrator()

@pytest.fixture
def mock_models():
    """Mock all AI models."""
    with patch('ai.models.code_embedding.CodeEmbeddingModel') as embedding_mock, \
         patch('ai.models.code_understanding.CodeUnderstandingModel') as understanding_mock, \
         patch('ai.models.pattern_detection.PatternDetectionModel') as pattern_mock, \
         patch('ai.models.security_analysis.SecurityAnalysisModel') as security_mock, \
         patch('ai.models.performance_optimization.PerformanceOptimizationModel') as performance_mock:
        
        yield {
            'embedding': embedding_mock,
            'understanding': understanding_mock,
            'pattern': pattern_mock,
            'security': security_mock,
            'performance': performance_mock
        }

@pytest.mark.asyncio
async def test_analyze_code_basic(orchestrator, mock_models):
    """Test basic code analysis functionality."""
    # Configure mocks
    mock_models['understanding'].return_value.analyze.return_value = {
        'purpose': 'Calculate Fibonacci numbers',
        'complexity': 'medium'
    }
    
    # Run analysis
    results = await orchestrator.analyze_code(
        code=SAMPLE_CODE,
        language='python'
    )
    
    # Verify results
    assert results is not None
    assert 'understanding' in results
    assert results.understanding['purpose'] == 'Calculate Fibonacci numbers'
    
    # Verify model calls
    mock_models['understanding'].return_value.analyze.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_code_security(orchestrator, mock_models):
    """Test security analysis functionality."""
    # Configure mocks
    mock_models['security'].return_value.analyze.return_value = {
        'vulnerabilities': [{
            'type': 'SQL Injection',
            'severity': 'high',
            'location': (8, 8)
        }]
    }
    
    # Run analysis
    results = await orchestrator.analyze_code(
        code=SAMPLE_CODE,
        language='python',
        analysis_types=['security']
    )
    
    # Verify results
    assert results is not None
    assert 'vulnerabilities' in results
    assert len(results.vulnerabilities) == 1
    assert results.vulnerabilities[0]['type'] == 'SQL Injection'
    
    # Verify model calls
    mock_models['security'].return_value.analyze.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_code_performance(orchestrator, mock_models):
    """Test performance analysis functionality."""
    # Configure mocks
    mock_models['performance'].return_value.analyze.return_value = {
        'time_complexity': 'O(2^n)',
        'bottlenecks': ['recursive_fibonacci'],
        'suggestions': ['Use dynamic programming']
    }
    
    # Run analysis
    results = await orchestrator.analyze_code(
        code=SAMPLE_CODE,
        language='python',
        analysis_types=['performance']
    )
    
    # Verify results
    assert results is not None
    assert 'time_complexity' in results
    assert results.time_complexity == 'O(2^n)'
    assert len(results.bottlenecks) == 1
    
    # Verify model calls
    mock_models['performance'].return_value.analyze.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_code_patterns(orchestrator, mock_models):
    """Test pattern detection functionality."""
    # Configure mocks
    mock_models['pattern'].return_value.analyze.return_value = {
        'patterns': [{
            'name': 'Recursive Algorithm',
            'confidence': 0.95,
            'location': (1, 5)
        }],
        'anti_patterns': [{
            'name': 'String Formatting SQL',
            'severity': 'high',
            'location': (8, 8)
        }]
    }
    
    # Run analysis
    results = await orchestrator.analyze_code(
        code=SAMPLE_CODE,
        language='python',
        analysis_types=['patterns']
    )
    
    # Verify results
    assert results is not None
    assert 'patterns' in results
    assert len(results.patterns) == 1
    assert results.patterns[0]['name'] == 'Recursive Algorithm'
    assert len(results.anti_patterns) == 1
    
    # Verify model calls
    mock_models['pattern'].return_value.analyze.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_code_all_types(orchestrator, mock_models):
    """Test comprehensive analysis with all types."""
    # Run analysis
    results = await orchestrator.analyze_code(
        code=SAMPLE_CODE,
        language='python',
        analysis_types=['understanding', 'security', 'performance', 'patterns']
    )
    
    # Verify all models were called
    for mock in mock_models.values():
        mock.return_value.analyze.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_code_invalid_language(orchestrator):
    """Test handling of invalid programming language."""
    with pytest.raises(ValueError):
        await orchestrator.analyze_code(
            code=SAMPLE_CODE,
            language='invalid_lang'
        )

@pytest.mark.asyncio
async def test_analyze_code_empty(orchestrator):
    """Test handling of empty code."""
    with pytest.raises(ValueError):
        await orchestrator.analyze_code(
            code='',
            language='python'
        )

def test_get_improvement_suggestions(orchestrator, mock_models):
    """Test improvement suggestion generation."""
    # Configure mock results
    results = Mock()
    results.vulnerabilities = [{
        'type': 'SQL Injection',
        'severity': 'high',
        'location': (8, 8)
    }]
    results.performance_issues = [{
        'type': 'Exponential Complexity',
        'severity': 'high',
        'location': (1, 5)
    }]
    
    # Get suggestions
    suggestions = orchestrator.get_improvement_suggestions(results)
    
    # Verify suggestions
    assert len(suggestions) > 0
    assert any('SQL Injection' in s for s in suggestions)
    assert any('complexity' in s.lower() for s in suggestions)

def test_orchestrator_initialization():
    """Test AIOrchestrator initialization."""
    orchestrator = AIOrchestrator()
    assert orchestrator.models is not None
    assert len(orchestrator.models) > 0
    assert all(model is not None for model in orchestrator.models.values())
