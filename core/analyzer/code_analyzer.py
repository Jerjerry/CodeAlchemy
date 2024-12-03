"""
CodeInsight Code Analyzer Module

This module provides advanced code analysis capabilities, including:
- Code quality assessment
- Pattern detection
- Complexity analysis
- Security vulnerability detection
- Performance optimization suggestions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import ast
import re

from loguru import logger
from pydantic import BaseModel

from ..parser.base_parser import AbstractSyntaxTree, ParsedEntity, SymbolReference


@dataclass
class CodeIssue:
    """Represents a detected code issue."""
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str  # 'security', 'performance', 'maintainability', etc.
    message: str
    location: Tuple[int, int]  # line, column
    suggestion: Optional[str]
    confidence: float  # 0.0 to 1.0


class CodeMetrics(BaseModel):
    """Represents various code quality metrics."""
    cyclomatic_complexity: float
    cognitive_complexity: float
    maintainability_index: float
    lines_of_code: int
    comment_ratio: float
    test_coverage: Optional[float]
    documentation_completeness: float
    dependency_count: int
    security_score: float
    performance_score: float


class OptimizationSuggestion(BaseModel):
    """Represents a code optimization suggestion."""
    category: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    effort: str  # 'high', 'medium', 'low'
    before_code: str
    after_code: str
    estimated_improvement: str
    rationale: str


class SecurityVulnerability(BaseModel):
    """Represents a detected security vulnerability."""
    severity: str
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    description: str
    affected_code: str
    mitigation: str
    false_positive_probability: float
    references: List[str]


class CodeAnalyzer:
    """Main code analysis engine."""

    def __init__(self):
        self.patterns = self._load_patterns()
        self.security_rules = self._load_security_rules()
        self.performance_patterns = self._load_performance_patterns()
        logger.info("Code analyzer initialized")

    async def analyze_code(self, ast: AbstractSyntaxTree) -> Tuple[
        List[CodeIssue],
        CodeMetrics,
        List[OptimizationSuggestion],
        List[SecurityVulnerability]
    ]:
        """Perform comprehensive code analysis."""
        try:
            # Parallel analysis for better performance
            issues = await self._detect_issues(ast)
            metrics = await self._calculate_metrics(ast)
            optimizations = await self._suggest_optimizations(ast)
            vulnerabilities = await self._detect_vulnerabilities(ast)

            return issues, metrics, optimizations, vulnerabilities
        except Exception as e:
            logger.error(f"Error during code analysis: {str(e)}")
            raise

    async def _detect_issues(self, ast: AbstractSyntaxTree) -> List[CodeIssue]:
        """Detect code quality issues and anti-patterns."""
        issues = []
        
        for entity in ast.entities:
            # Check for common code smells
            issues.extend(self._check_code_smells(entity))
            
            # Check for complexity issues
            if entity.complexity > 10:  # McCabe complexity threshold
                issues.append(CodeIssue(
                    severity="high",
                    category="maintainability",
                    message=f"High cyclomatic complexity ({entity.complexity}) in {entity.name}",
                    location=(entity.location.line, entity.location.column),
                    suggestion="Consider breaking down this code into smaller functions",
                    confidence=0.9
                ))

            # Check for documentation issues
            if not entity.docstring and entity.kind in ["class", "function"]:
                issues.append(CodeIssue(
                    severity="medium",
                    category="documentation",
                    message=f"Missing documentation for {entity.kind} {entity.name}",
                    location=(entity.location.line, entity.location.column),
                    suggestion="Add a docstring describing the purpose and usage",
                    confidence=1.0
                ))

        return issues

    async def _calculate_metrics(self, ast: AbstractSyntaxTree) -> CodeMetrics:
        """Calculate various code quality metrics."""
        total_complexity = sum(entity.complexity for entity in ast.entities)
        loc = self._count_lines_of_code(ast)
        
        return CodeMetrics(
            cyclomatic_complexity=total_complexity,
            cognitive_complexity=self._calculate_cognitive_complexity(ast),
            maintainability_index=self._calculate_maintainability_index(ast),
            lines_of_code=loc,
            comment_ratio=self._calculate_comment_ratio(ast),
            test_coverage=None,  # Requires runtime analysis
            documentation_completeness=self._calculate_doc_completeness(ast),
            dependency_count=len(ast.imports),
            security_score=self._calculate_security_score(ast),
            performance_score=self._calculate_performance_score(ast)
        )

    async def _suggest_optimizations(self, ast: AbstractSyntaxTree) -> List[OptimizationSuggestion]:
        """Suggest code optimizations."""
        suggestions = []
        
        for entity in ast.entities:
            # Check for inefficient patterns
            for pattern in self.performance_patterns:
                if self._matches_pattern(entity, pattern):
                    suggestions.append(self._create_optimization_suggestion(entity, pattern))

            # Check for potential algorithmic improvements
            if complexity_suggestion := self._check_algorithmic_complexity(entity):
                suggestions.append(complexity_suggestion)

        return suggestions

    async def _detect_vulnerabilities(self, ast: AbstractSyntaxTree) -> List[SecurityVulnerability]:
        """Detect security vulnerabilities."""
        vulnerabilities = []
        
        for entity in ast.entities:
            # Check against security rules
            for rule in self.security_rules:
                if self._matches_security_rule(entity, rule):
                    vulnerabilities.append(self._create_vulnerability_report(entity, rule))

            # Check for common vulnerability patterns
            vulnerabilities.extend(self._check_common_vulnerabilities(entity))

        return vulnerabilities

    def _load_patterns(self) -> Dict[str, Any]:
        """Load code pattern definitions."""
        # Placeholder - would load from configuration
        return {}

    def _load_security_rules(self) -> List[Dict[str, Any]]:
        """Load security rule definitions."""
        # Placeholder - would load from configuration
        return []

    def _load_performance_patterns(self) -> List[Dict[str, Any]]:
        """Load performance pattern definitions."""
        # Placeholder - would load from configuration
        return []

    def _check_code_smells(self, entity: ParsedEntity) -> List[CodeIssue]:
        """Check for code smells in an entity."""
        issues = []
        
        # Check for long method
        if entity.kind == "function" and len(entity.code_snippet.splitlines()) > 50:
            issues.append(CodeIssue(
                severity="medium",
                category="maintainability",
                message="Method is too long",
                location=(entity.location.line, entity.location.column),
                suggestion="Consider breaking this method into smaller methods",
                confidence=0.8
            ))

        # Check for too many parameters
        if entity.kind == "function" and len(self._get_parameters(entity)) > 5:
            issues.append(CodeIssue(
                severity="medium",
                category="maintainability",
                message="Too many parameters",
                location=(entity.location.line, entity.location.column),
                suggestion="Consider using a parameter object",
                confidence=0.7
            ))

        return issues

    def _calculate_cognitive_complexity(self, ast: AbstractSyntaxTree) -> float:
        """Calculate cognitive complexity score."""
        # Placeholder - would implement detailed cognitive complexity calculation
        return sum(entity.complexity * 1.5 for entity in ast.entities)

    def _calculate_maintainability_index(self, ast: AbstractSyntaxTree) -> float:
        """Calculate maintainability index."""
        # Simplified MI calculation
        loc = self._count_lines_of_code(ast)
        complexity = sum(entity.complexity for entity in ast.entities)
        
        if loc == 0:
            return 100.0
            
        return 100.0 - (complexity * math.log(loc))

    def _count_lines_of_code(self, ast: AbstractSyntaxTree) -> int:
        """Count effective lines of code."""
        return sum(
            len(entity.code_snippet.splitlines())
            for entity in ast.entities
        )

    def _calculate_comment_ratio(self, ast: AbstractSyntaxTree) -> float:
        """Calculate comment to code ratio."""
        total_lines = 0
        comment_lines = 0
        
        for entity in ast.entities:
            lines = entity.code_snippet.splitlines()
            total_lines += len(lines)
            comment_lines += sum(1 for line in lines if line.strip().startswith('#'))
            
        return comment_lines / total_lines if total_lines > 0 else 0.0

    def _calculate_doc_completeness(self, ast: AbstractSyntaxTree) -> float:
        """Calculate documentation completeness score."""
        total_entities = len(ast.entities)
        documented_entities = sum(
            1 for entity in ast.entities
            if entity.docstring and len(entity.docstring) > 10
        )
        
        return documented_entities / total_entities if total_entities > 0 else 0.0

    def _calculate_security_score(self, ast: AbstractSyntaxTree) -> float:
        """Calculate security score."""
        # Placeholder - would implement detailed security scoring
        return 0.8

    def _calculate_performance_score(self, ast: AbstractSyntaxTree) -> float:
        """Calculate performance score."""
        # Placeholder - would implement detailed performance scoring
        return 0.7

    def _get_parameters(self, entity: ParsedEntity) -> List[str]:
        """Extract parameters from a function entity."""
        if not entity.kind == "function":
            return []
            
        # Basic parameter extraction from AST
        if isinstance(entity.ast_node, ast.FunctionDef):
            return [arg.arg for arg in entity.ast_node.args.args]
        return []

    def _matches_pattern(self, entity: ParsedEntity, pattern: Dict[str, Any]) -> bool:
        """Check if entity matches a code pattern."""
        # Placeholder - would implement pattern matching
        return False

    def _matches_security_rule(self, entity: ParsedEntity, rule: Dict[str, Any]) -> bool:
        """Check if entity matches a security rule."""
        # Placeholder - would implement security rule matching
        return False

    def _create_optimization_suggestion(
        self,
        entity: ParsedEntity,
        pattern: Dict[str, Any]
    ) -> OptimizationSuggestion:
        """Create an optimization suggestion."""
        # Placeholder - would create detailed optimization suggestions
        return OptimizationSuggestion(
            category="performance",
            description="Generic optimization suggestion",
            impact="medium",
            effort="medium",
            before_code="# Original code",
            after_code="# Optimized code",
            estimated_improvement="~20% performance improvement",
            rationale="Generic optimization rationale"
        )

    def _create_vulnerability_report(
        self,
        entity: ParsedEntity,
        rule: Dict[str, Any]
    ) -> SecurityVulnerability:
        """Create a security vulnerability report."""
        # Placeholder - would create detailed vulnerability reports
        return SecurityVulnerability(
            severity="medium",
            cwe_id=None,
            description="Generic vulnerability description",
            affected_code=entity.code_snippet,
            mitigation="Generic mitigation steps",
            false_positive_probability=0.2,
            references=[]
        )

    def _check_algorithmic_complexity(self, entity: ParsedEntity) -> Optional[OptimizationSuggestion]:
        """Check for potential algorithmic improvements."""
        # Placeholder - would implement algorithmic analysis
        return None

    def _check_common_vulnerabilities(self, entity: ParsedEntity) -> List[SecurityVulnerability]:
        """Check for common security vulnerabilities."""
        # Placeholder - would implement common vulnerability checks
        return []
