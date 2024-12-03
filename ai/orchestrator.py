"""
CodeInsight AI Orchestrator

This module orchestrates all AI models to provide comprehensive code analysis.
It coordinates:
1. Code understanding and documentation
2. Pattern and anomaly detection
3. Security vulnerability analysis
4. Performance optimization
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .models.code_embedding import CodeEmbedding, CodeEmbeddingManager
from .models.code_understanding import (
    CodeUnderstanding,
    DocumentationGeneration,
    CodeUnderstandingModel
)
from .models.pattern_detection import (
    PatternMatch,
    CodeAnomaly,
    PatternDetectionModel
)
from .models.security_analysis import (
    SecurityVulnerability,
    SecurityRisk,
    SecurityAnalysisModel
)
from .models.performance_optimization import (
    PerformanceIssue,
    ResourceUsage,
    PerformanceOptimizer
)


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis results from all models."""
    # Code Understanding
    understanding: CodeUnderstanding
    documentation: DocumentationGeneration
    
    # Pattern Detection
    patterns: List[PatternMatch]
    anomalies: List[CodeAnomaly]
    
    # Security
    vulnerabilities: List[SecurityVulnerability]
    security_risks: List[SecurityRisk]
    
    # Performance
    performance_issues: List[PerformanceIssue]
    resource_usage: ResourceUsage
    
    # Metadata
    analysis_time: float
    confidence_scores: Dict[str, float]


class AIOrchestrator:
    """
    Orchestrates all AI models to provide comprehensive code analysis.
    """

    def __init__(self, models_path: Optional[str] = None):
        """Initialize all AI models."""
        logger.info("Initializing AI Orchestrator")
        
        # Initialize individual models
        self.embedding_manager = CodeEmbeddingManager()
        self.understanding_model = CodeUnderstandingModel()
        self.pattern_model = PatternDetectionModel()
        self.security_model = SecurityAnalysisModel()
        self.performance_optimizer = PerformanceOptimizer()
        
        logger.info("AI Orchestrator initialization complete")

    async def analyze_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[List[str]] = None,
        analysis_types: Optional[List[str]] = None
    ) -> ComprehensiveAnalysis:
        """
        Perform comprehensive code analysis using all models.
        
        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional surrounding code context
            analysis_types: Optional list of specific analyses to run
                          ['understanding', 'patterns', 'security', 'performance']
            
        Returns:
            ComprehensiveAnalysis: Complete analysis results
        """
        import time
        start_time = time.time()
        
        # Default to all analyses if none specified
        if analysis_types is None:
            analysis_types = [
                'understanding',
                'patterns',
                'security',
                'performance'
            ]
        
        # Get code embedding (shared across models)
        embedding = self.embedding_manager.get_embedding(code, language, context)
        
        results = ComprehensiveAnalysis(
            understanding=None,
            documentation=None,
            patterns=[],
            anomalies=[],
            vulnerabilities=[],
            security_risks=[],
            performance_issues=[],
            resource_usage=None,
            analysis_time=0.0,
            confidence_scores={}
        )
        
        # Run requested analyses
        if 'understanding' in analysis_types:
            results.understanding = self.understanding_model.understand_code(
                code, language, context
            )
            results.documentation = self.understanding_model.generate_documentation(
                code, results.understanding, language
            )
            results.confidence_scores['understanding'] = results.understanding.confidence

        if 'patterns' in analysis_types:
            results.patterns = self.pattern_model.detect_patterns(
                code, language
            )
            results.anomalies = self.pattern_model.detect_anomalies(
                code, language, context
            )
            results.confidence_scores['patterns'] = sum(
                p.confidence for p in results.patterns
            ) / max(len(results.patterns), 1)

        if 'security' in analysis_types:
            results.vulnerabilities, results.security_risks = (
                self.security_model.analyze_security(code, language, context)
            )
            results.confidence_scores['security'] = sum(
                v.confidence for v in results.vulnerabilities
            ) / max(len(results.vulnerabilities), 1)

        if 'performance' in analysis_types:
            results.performance_issues, results.resource_usage = (
                self.performance_optimizer.analyze_performance(code, language)
            )
            results.confidence_scores['performance'] = sum(
                i.confidence for i in results.performance_issues
            ) / max(len(results.performance_issues), 1)

        results.analysis_time = time.time() - start_time
        
        return results

    def generate_report(
        self,
        analysis: ComprehensiveAnalysis,
        format: str = "markdown"
    ) -> str:
        """
        Generate a comprehensive report from analysis results.
        
        Args:
            analysis: Analysis results
            format: Output format ('markdown' or 'html')
            
        Returns:
            str: Formatted report
        """
        if format == "markdown":
            return self._generate_markdown_report(analysis)
        elif format == "html":
            return self._generate_html_report(analysis)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown_report(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Generate markdown report."""
        report = [
            "# Code Analysis Report\n",
            f"Analysis Time: {analysis.analysis_time:.2f} seconds\n",
            
            "## Code Understanding\n",
            f"Purpose: {analysis.understanding.purpose}\n",
            f"Description: {analysis.understanding.description}\n",
            
            "### Key Components\n",
            *[f"- {comp}\n" for comp in analysis.understanding.key_components],
            
            "## Detected Patterns\n",
            *[f"- {p.pattern_name} (Confidence: {p.confidence:.2f})\n"
              for p in analysis.patterns],
            
            "## Security Analysis\n",
            "### Vulnerabilities\n",
            *[f"- {v.vulnerability_type} (Severity: {v.severity})\n"
              for v in analysis.vulnerabilities],
            
            "### Security Risks\n",
            *[f"- {r.risk_type} (Priority: {r.priority})\n"
              for r in analysis.security_risks],
            
            "## Performance Analysis\n",
            "### Issues\n",
            *[f"- {i.issue_type} (Severity: {i.severity})\n"
              for i in analysis.performance_issues],
            
            "### Resource Usage\n",
            f"Time Complexity: {analysis.resource_usage.time_complexity}\n",
            f"Space Complexity: {analysis.resource_usage.space_complexity}\n",
            
            "## Confidence Scores\n",
            *[f"- {k}: {v:.2f}\n" for k, v in analysis.confidence_scores.items()]
        ]
        
        return "".join(report)

    def _generate_html_report(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML
        import markdown
        return markdown.markdown(self._generate_markdown_report(analysis))

    def get_improvement_suggestions(
        self,
        analysis: ComprehensiveAnalysis
    ) -> List[str]:
        """
        Generate prioritized list of improvement suggestions.
        
        Args:
            analysis: Analysis results
            
        Returns:
            List[str]: Prioritized improvement suggestions
        """
        suggestions = []
        
        # Security improvements (highest priority)
        for vuln in analysis.vulnerabilities:
            if vuln.severity in ['critical', 'high']:
                suggestions.append(
                    f"[SECURITY] {vuln.description}: {vuln.remediation}"
                )

        # Performance improvements
        for issue in analysis.performance_issues:
            if issue.severity in ['critical', 'high']:
                suggestions.append(
                    f"[PERFORMANCE] {issue.description}: {issue.optimization_suggestion}"
                )

        # Pattern improvements
        for pattern in analysis.patterns:
            if pattern.confidence > 0.8:
                suggestions.append(
                    f"[PATTERN] {pattern.pattern_name}: {pattern.suggestions[0]}"
                )

        # Sort by severity and confidence
        suggestions.sort(key=lambda x: (
            'SECURITY' in x,
            'PERFORMANCE' in x,
            'PATTERN' in x
        ), reverse=True)
        
        return suggestions

    def explain_analysis(
        self,
        analysis: ComprehensiveAnalysis,
        aspect: Optional[str] = None
    ) -> str:
        """
        Generate natural language explanation of analysis results.
        
        Args:
            analysis: Analysis results
            aspect: Optional specific aspect to explain
                   ('understanding', 'patterns', 'security', 'performance')
            
        Returns:
            str: Natural language explanation
        """
        if aspect == 'understanding':
            return self._explain_understanding(analysis)
        elif aspect == 'patterns':
            return self._explain_patterns(analysis)
        elif aspect == 'security':
            return self._explain_security(analysis)
        elif aspect == 'performance':
            return self._explain_performance(analysis)
        else:
            # Generate complete explanation
            explanations = [
                self._explain_understanding(analysis),
                self._explain_patterns(analysis),
                self._explain_security(analysis),
                self._explain_performance(analysis)
            ]
            return "\n\n".join(explanations)

    def _explain_understanding(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Explain code understanding results."""
        return (
            f"This code appears to be {analysis.understanding.purpose}. "
            f"{analysis.understanding.description}\n\n"
            f"It consists of {len(analysis.understanding.key_components)} "
            f"key components: {', '.join(analysis.understanding.key_components)}."
        )

    def _explain_patterns(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Explain pattern detection results."""
        pattern_count = len(analysis.patterns)
        anomaly_count = len(analysis.anomalies)
        
        return (
            f"Analysis detected {pattern_count} design patterns and "
            f"{anomaly_count} potential code anomalies. "
            + (f"The main patterns found are: "
               f"{', '.join(p.pattern_name for p in analysis.patterns[:3])}..."
               if pattern_count > 0 else "")
        )

    def _explain_security(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Explain security analysis results."""
        vuln_count = len(analysis.vulnerabilities)
        risk_count = len(analysis.security_risks)
        
        critical_vulns = sum(
            1 for v in analysis.vulnerabilities
            if v.severity == 'critical'
        )
        
        return (
            f"Security analysis revealed {vuln_count} potential vulnerabilities "
            f"({critical_vulns} critical) and {risk_count} security risks. "
            + (f"Critical issues include: "
               f"{', '.join(v.vulnerability_type for v in analysis.vulnerabilities if v.severity == 'critical')}"
               if critical_vulns > 0 else "No critical issues found.")
        )

    def _explain_performance(
        self,
        analysis: ComprehensiveAnalysis
    ) -> str:
        """Explain performance analysis results."""
        issue_count = len(analysis.performance_issues)
        
        return (
            f"Performance analysis identified {issue_count} potential issues. "
            f"The code has {analysis.resource_usage.time_complexity} time complexity "
            f"and {analysis.resource_usage.space_complexity} space complexity. "
            + (f"Key performance issues: "
               f"{', '.join(i.issue_type for i in analysis.performance_issues[:3])}..."
               if issue_count > 0 else "No significant performance issues found.")
        )
