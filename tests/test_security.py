"""
Tests for the Security Analysis component.
"""

import pytest
from unittest.mock import Mock, patch

from ai.models.security_analysis import SecurityAnalysisModel

# Test data
VULNERABLE_CODE = """
def process_user_input(user_data):
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
    
    # Command injection vulnerability
    import os
    os.system(f"process {user_data['filename']}")
    
    # Unsafe deserialization
    import pickle
    data = pickle.loads(user_data['data'])
    
    # Hardcoded credentials
    password = "super_secret_123"
    api_key = "1234567890abcdef"
"""

SECURE_CODE = """
def process_user_input(user_data):
    # Parameterized query
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_data['id'],))
    
    # Safe command execution
    import subprocess
    subprocess.run(['process', user_data['filename']], check=True)
    
    # Safe deserialization
    import json
    data = json.loads(user_data['data'])
    
    # Environment variables
    import os
    password = os.getenv('DB_PASSWORD')
    api_key = os.getenv('API_KEY')
"""

@pytest.fixture
def security_model():
    """Create a SecurityAnalysisModel instance for testing."""
    return SecurityAnalysisModel()

def test_detect_sql_injection(security_model):
    """Test SQL injection vulnerability detection."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'SQL Injection']
    
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0]['severity'] == 'high'
    
    # Analyze secure code
    results = security_model.analyze(SECURE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'SQL Injection']
    
    assert len(vulnerabilities) == 0

def test_detect_command_injection(security_model):
    """Test command injection vulnerability detection."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Command Injection']
    
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0]['severity'] == 'high'
    
    # Analyze secure code
    results = security_model.analyze(SECURE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Command Injection']
    
    assert len(vulnerabilities) == 0

def test_detect_unsafe_deserialization(security_model):
    """Test unsafe deserialization vulnerability detection."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Unsafe Deserialization']
    
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0]['severity'] == 'high'
    
    # Analyze secure code
    results = security_model.analyze(SECURE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Unsafe Deserialization']
    
    assert len(vulnerabilities) == 0

def test_detect_hardcoded_credentials(security_model):
    """Test hardcoded credentials detection."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Hardcoded Credentials']
    
    assert len(vulnerabilities) == 2  # password and api_key
    assert all(v['severity'] == 'medium' for v in vulnerabilities)
    
    # Analyze secure code
    results = security_model.analyze(SECURE_CODE)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Hardcoded Credentials']
    
    assert len(vulnerabilities) == 0

def test_security_risk_assessment(security_model):
    """Test security risk assessment functionality."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    
    assert 'risk_assessment' in results
    assessment = results['risk_assessment']
    
    assert assessment['overall_risk'] == 'high'
    assert len(assessment['risk_factors']) > 0
    assert assessment['compliance_issues'] > 0

def test_owasp_compliance(security_model):
    """Test OWASP Top 10 compliance checking."""
    # Analyze vulnerable code
    results = security_model.analyze(VULNERABLE_CODE)
    
    assert 'owasp_compliance' in results
    compliance = results['owasp_compliance']
    
    assert 'A1:2021' in compliance  # Broken Access Control
    assert 'A3:2021' in compliance  # Injection
    assert any(issue['status'] == 'failed' for issue in compliance.values())

def test_security_best_practices(security_model):
    """Test security best practices validation."""
    # Analyze secure code
    results = security_model.analyze(SECURE_CODE)
    
    assert 'best_practices' in results
    practices = results['best_practices']
    
    assert practices['input_validation'] == 'passed'
    assert practices['secure_configuration'] == 'passed'
    assert practices['data_protection'] == 'passed'

def test_vulnerability_description(security_model):
    """Test vulnerability description generation."""
    results = security_model.analyze(VULNERABLE_CODE)
    
    for vulnerability in results['vulnerabilities']:
        assert 'description' in vulnerability
        assert 'remediation' in vulnerability
        assert len(vulnerability['description']) > 0
        assert len(vulnerability['remediation']) > 0

def test_false_positives(security_model):
    """Test handling of potential false positives."""
    # Code that might trigger false positives
    code = """
    def safe_function():
        # This is a comment about SQL injection
        safe_query = "SELECT * FROM users"
        # This is a password in a comment: password123
        return safe_query
    """
    
    results = security_model.analyze(code)
    
    assert len(results['vulnerabilities']) == 0

def test_custom_rules(security_model):
    """Test custom security rule integration."""
    # Add custom rule
    custom_rule = {
        'pattern': r'debug\s*=\s*True',
        'type': 'Debug Mode Enabled',
        'severity': 'medium'
    }
    security_model.add_custom_rule(custom_rule)
    
    # Test code with custom vulnerability
    code = """
    app.config['DEBUG'] = True
    debug = True
    """
    
    results = security_model.analyze(code)
    vulnerabilities = [v for v in results['vulnerabilities'] 
                      if v['type'] == 'Debug Mode Enabled']
    
    assert len(vulnerabilities) == 2
