"""
CodeInsight Security Analysis Model

This module provides advanced security analysis capabilities for code,
including:
1. Vulnerability detection
2. Security risk assessment
3. OWASP compliance checking
4. Security best practices validation
"""

import torch
import torch.nn as nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    T5EncoderModel,
    T5Tokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger
import ast
from typing import Any

from .code_embedding import CodeEmbedding, CodeEmbeddingManager


@dataclass
class SecurityVulnerability:
    """Represents a detected security vulnerability."""
    vulnerability_type: str
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    cvss_score: float  # Common Vulnerability Scoring System (0.0-10.0)
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    affected_lines: List[Tuple[int, int]]
    impact: str
    remediation: str
    confidence: float
    references: List[str]


@dataclass
class SecurityRisk:
    """Represents an identified security risk."""
    risk_type: str
    likelihood: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    description: str
    mitigation_strategy: str
    affected_components: List[str]
    compliance_impact: List[str]
    priority: str  # 'critical', 'high', 'medium', 'low'


class SecurityAnalysisModel(nn.Module):
    """
    Neural model for security analysis and vulnerability detection in code.
    Uses specialized transformer models and security-focused architectures.
    """

    def __init__(
        self,
        security_model: str = "microsoft/codebert-base",
        vulnerability_db: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        
        # Initialize embedding manager
        self.embedding_manager = CodeEmbeddingManager()
        
        # Initialize security analysis model
        self.security_tokenizer = RobertaTokenizer.from_pretrained(security_model)
        self.security_model = RobertaModel.from_pretrained(security_model)
        self.security_model = self.security_model.to(device)
        
        # Vulnerability detection heads
        self.vulnerability_detector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.get_vulnerability_types()))
        )
        
        # Risk assessment heads
        self.risk_assessor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.get_risk_types()))
        )
        
        # Load vulnerability database
        self.vulnerability_db = self._load_vulnerability_db(vulnerability_db)
        
        logger.info(f"Initialized SecurityAnalysisModel on {device}")

    def analyze_security(
        self,
        code: str,
        language: str = "python",
        context: Optional[List[str]] = None
    ) -> Tuple[List[SecurityVulnerability], List[SecurityRisk]]:
        """
        Perform comprehensive security analysis of code.
        
        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional surrounding code context
            
        Returns:
            Tuple[List[SecurityVulnerability], List[SecurityRisk]]:
                Detected vulnerabilities and identified risks
        """
        # Get code embedding
        embedding = self.embedding_manager.get_embedding(code, language, context)
        
        # Detect vulnerabilities
        vulnerabilities = self.detect_vulnerabilities(code, embedding)
        
        # Assess risks
        risks = self.assess_risks(code, embedding, vulnerabilities)
        
        return vulnerabilities, risks

    def detect_vulnerabilities(
        self,
        code: str,
        embedding: Optional[CodeEmbedding] = None,
        min_confidence: float = 0.7
    ) -> List[SecurityVulnerability]:
        """
        Detect security vulnerabilities in code.
        
        Args:
            code: Source code to analyze
            embedding: Optional pre-computed embedding
            min_confidence: Minimum confidence threshold
            
        Returns:
            List[SecurityVulnerability]: Detected vulnerabilities
        """
        # Get or compute embedding
        if embedding is None:
            embedding = self.embedding_manager.get_embedding(code)
            
        # Detect vulnerabilities using transformer
        inputs = self.security_tokenizer(
            code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.security_model(**inputs)
            vuln_logits = self.vulnerability_detector(outputs.last_hidden_state[:, 0, :])
            vuln_probs = torch.sigmoid(vuln_logits)
        
        # Convert to vulnerability objects
        vulnerabilities = []
        for idx, prob in enumerate(vuln_probs[0]):
            if prob.item() >= min_confidence:
                vuln_type = self.get_vulnerability_types()[idx]
                vuln_info = self.vulnerability_db.get(vuln_type, {})
                
                vulnerability = SecurityVulnerability(
                    vulnerability_type=vuln_type,
                    cwe_id=vuln_info.get("cwe_id"),
                    cvss_score=self._calculate_cvss_score(vuln_type, code),
                    severity=self._determine_severity(prob.item()),
                    description=vuln_info.get("description", ""),
                    affected_lines=self._find_vulnerable_lines(code, vuln_type),
                    impact=vuln_info.get("impact", ""),
                    remediation=vuln_info.get("remediation", ""),
                    confidence=prob.item(),
                    references=vuln_info.get("references", [])
                )
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities

    def assess_risks(
        self,
        code: str,
        embedding: Optional[CodeEmbedding] = None,
        vulnerabilities: Optional[List[SecurityVulnerability]] = None
    ) -> List[SecurityRisk]:
        """
        Assess security risks in code.
        
        Args:
            code: Source code to analyze
            embedding: Optional pre-computed embedding
            vulnerabilities: Optional pre-detected vulnerabilities
            
        Returns:
            List[SecurityRisk]: Identified security risks
        """
        # Get or compute embedding
        if embedding is None:
            embedding = self.embedding_manager.get_embedding(code)
            
        # Get or detect vulnerabilities
        if vulnerabilities is None:
            vulnerabilities = self.detect_vulnerabilities(code, embedding)
            
        # Assess risks using transformer
        inputs = self.security_tokenizer(
            code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.security_model(**inputs)
            risk_logits = self.risk_assessor(outputs.last_hidden_state[:, 0, :])
            risk_probs = torch.sigmoid(risk_logits)
        
        # Convert to risk objects
        risks = []
        for idx, prob in enumerate(risk_probs[0]):
            risk_type = self.get_risk_types()[idx]
            
            # Calculate risk metrics
            likelihood = prob.item()
            impact = self._calculate_risk_impact(risk_type, vulnerabilities)
            
            risk = SecurityRisk(
                risk_type=risk_type,
                likelihood=likelihood,
                impact=impact,
                description=self._generate_risk_description(risk_type, likelihood, impact),
                mitigation_strategy=self._generate_mitigation_strategy(risk_type),
                affected_components=self._identify_affected_components(code, risk_type),
                compliance_impact=self._assess_compliance_impact(risk_type),
                priority=self._determine_risk_priority(likelihood, impact)
            )
            risks.append(risk)
        
        return risks

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        try:
            tree = ast.parse(code)
        except Exception as e:
            logger.error(f"Failed to parse code: {e}")
            return {
                "vulnerabilities": [],
                "risk_level": "unknown",
                "confidence": 0.0
            }

        vulnerabilities = []
        
        # Check for SQL injection
        sql_vulnerabilities = self._check_sql_injection(tree)
        vulnerabilities.extend(sql_vulnerabilities)
        
        # Check for command injection
        cmd_vulnerabilities = self._check_command_injection(tree)
        vulnerabilities.extend(cmd_vulnerabilities)
        
        # Calculate risk level and confidence
        risk_level = "high" if vulnerabilities else "low"
        confidence = 0.9 if vulnerabilities else 0.7
        
        return {
            "vulnerabilities": vulnerabilities,
            "risk_level": risk_level,
            "confidence": confidence
        }

    def _check_sql_injection(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for SQL injection vulnerabilities."""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for string formatting in SQL queries
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr in ['execute', 'query']):
                
                for arg in node.args:
                    if isinstance(arg, (ast.JoinedStr, ast.BinOp)):
                        vulnerabilities.append({
                            "type": "SQL Injection",
                            "severity": "high",
                            "location": f"Line {node.lineno}",
                            "description": "Potential SQL injection vulnerability - use parameterized queries"
                        })
                        
        return vulnerabilities

    def _check_command_injection(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for command injection vulnerabilities."""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for shell command execution
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id in ['system', 'popen', 'exec', 'eval']):
                
                for arg in node.args:
                    if isinstance(arg, (ast.JoinedStr, ast.BinOp, ast.Name)):
                        vulnerabilities.append({
                            "type": "Command Injection",
                            "severity": "high",
                            "location": f"Line {node.lineno}",
                            "description": "Potential command injection vulnerability - sanitize user input"
                        })
                        
        return vulnerabilities

    @staticmethod
    def get_vulnerability_types() -> List[str]:
        """Get list of known vulnerability types."""
        return [
            "sql_injection",
            "xss",
            "csrf",
            "command_injection",
            "path_traversal",
            "insecure_deserialization",
            "broken_authentication",
            "sensitive_data_exposure",
            "security_misconfiguration",
            "insufficient_logging"
        ]

    @staticmethod
    def get_risk_types() -> List[str]:
        """Get list of known risk types."""
        return [
            "data_breach",
            "unauthorized_access",
            "denial_of_service",
            "data_corruption",
            "privilege_escalation",
            "information_disclosure",
            "compliance_violation",
            "reputation_damage"
        ]

    def _load_vulnerability_db(
        self,
        db_path: Optional[str]
    ) -> Dict[str, Dict]:
        """Load vulnerability database."""
        # In practice, would load from a comprehensive database
        return {
            "sql_injection": {
                "cwe_id": "CWE-89",
                "description": "SQL injection vulnerability",
                "impact": "Critical data breach risk",
                "remediation": "Use parameterized queries",
                "references": ["https://owasp.org/www-community/attacks/SQL_Injection"]
            },
            # Add more vulnerability definitions...
        }

    def _calculate_cvss_score(
        self,
        vuln_type: str,
        code: str
    ) -> float:
        """Calculate CVSS score for vulnerability."""
        # In practice, would implement CVSS calculation
        base_scores = {
            "sql_injection": 9.8,
            "xss": 6.1,
            "csrf": 8.8
        }
        return base_scores.get(vuln_type, 5.0)

    def _determine_severity(self, confidence: float) -> str:
        """Determine severity level based on confidence."""
        if confidence > 0.9:
            return "critical"
        elif confidence > 0.7:
            return "high"
        elif confidence > 0.5:
            return "medium"
        return "low"

    def _find_vulnerable_lines(
        self,
        code: str,
        vuln_type: str
    ) -> List[Tuple[int, int]]:
        """Find lines containing vulnerability."""
        # In practice, would implement precise vulnerability localization
        return [(1, 5)]  # Placeholder

    def _calculate_risk_impact(
        self,
        risk_type: str,
        vulnerabilities: List[SecurityVulnerability]
    ) -> float:
        """Calculate potential impact of risk."""
        # Consider vulnerability severity
        severity_scores = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        
        impact = 0.0
        for vuln in vulnerabilities:
            impact = max(impact, severity_scores.get(vuln.severity, 0.0))
            
        return impact

    def _generate_risk_description(
        self,
        risk_type: str,
        likelihood: float,
        impact: float
    ) -> str:
        """Generate detailed risk description."""
        return f"Risk of {risk_type} with likelihood {likelihood:.2f} and impact {impact:.2f}"

    def _generate_mitigation_strategy(
        self,
        risk_type: str
    ) -> str:
        """Generate risk mitigation strategy."""
        strategies = {
            "data_breach": "Implement encryption and access controls",
            "unauthorized_access": "Strengthen authentication mechanisms",
            # Add more strategies...
        }
        return strategies.get(risk_type, "Implement security best practices")

    def _identify_affected_components(
        self,
        code: str,
        risk_type: str
    ) -> List[str]:
        """Identify components affected by risk."""
        # In practice, would implement component analysis
        return ["database", "authentication"]  # Placeholder

    def _assess_compliance_impact(
        self,
        risk_type: str
    ) -> List[str]:
        """Assess impact on compliance requirements."""
        compliance_mapping = {
            "data_breach": ["GDPR", "CCPA"],
            "unauthorized_access": ["HIPAA", "PCI-DSS"],
            # Add more mappings...
        }
        return compliance_mapping.get(risk_type, [])

    def _determine_risk_priority(
        self,
        likelihood: float,
        impact: float
    ) -> str:
        """Determine risk priority level."""
        risk_score = likelihood * impact
        
        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.3:
            return "medium"
        return "low"

    def explain_vulnerability(
        self,
        vulnerability: SecurityVulnerability
    ) -> str:
        """Generate detailed explanation of vulnerability."""
        explanation = f"Vulnerability Type: {vulnerability.vulnerability_type}\n"
        if vulnerability.cwe_id:
            explanation += f"CWE ID: {vulnerability.cwe_id}\n"
        explanation += f"CVSS Score: {vulnerability.cvss_score:.1f}\n"
        explanation += f"Severity: {vulnerability.severity}\n\n"
        explanation += f"Description: {vulnerability.description}\n\n"
        explanation += f"Impact: {vulnerability.impact}\n\n"
        explanation += "Remediation:\n"
        explanation += f"{vulnerability.remediation}\n\n"
        explanation += f"Confidence: {vulnerability.confidence:.2f}\n\n"
        explanation += "References:\n"
        for ref in vulnerability.references:
            explanation += f"- {ref}\n"
        return explanation

    def explain_risk(self, risk: SecurityRisk) -> str:
        """Generate detailed explanation of security risk."""
        explanation = f"Risk Type: {risk.risk_type}\n"
        explanation += f"Priority: {risk.priority}\n"
        explanation += f"Likelihood: {risk.likelihood:.2f}\n"
        explanation += f"Impact: {risk.impact:.2f}\n\n"
        explanation += f"Description: {risk.description}\n\n"
        explanation += "Mitigation Strategy:\n"
        explanation += f"{risk.mitigation_strategy}\n\n"
        explanation += "Affected Components:\n"
        for component in risk.affected_components:
            explanation += f"- {component}\n"
        explanation += "\nCompliance Impact:\n"
        for compliance in risk.compliance_impact:
            explanation += f"- {compliance}\n"
        return explanation

    def _detect_sql_injection(self, tree: ast.AST) -> List[SecurityVulnerability]:
        # In practice, would implement SQL injection detection
        return []  # Placeholder

    def _detect_command_injection(self, tree: ast.AST) -> List[SecurityVulnerability]:
        # In practice, would implement command injection detection
        return []  # Placeholder

    def _detect_unsafe_deserialization(self, tree: ast.AST) -> List[SecurityVulnerability]:
        # In practice, would implement unsafe deserialization detection
        return []  # Placeholder

    def _detect_hardcoded_credentials(self, tree: ast.AST) -> List[SecurityVulnerability]:
        # In practice, would implement hardcoded credentials detection
        return []  # Placeholder

    def _check_owasp_compliance(self, tree: ast.AST) -> Dict[str, bool]:
        # In practice, would implement OWASP Top 10 compliance checking
        return {}  # Placeholder

    def _generate_security_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        # In practice, would implement security recommendation generation
        return []  # Placeholder

    def _calculate_risk_level(self, vulnerabilities: List[SecurityVulnerability]) -> str:
        # In practice, would implement risk level calculation
        return "low"  # Placeholder
