"""
Example script demonstrating detailed single file analysis using CodeInsight.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ai.orchestrator import AIOrchestrator

# Initialize rich console
console = Console()

async def analyze_file(
    file_path: str,
    language: str = "python",
    analysis_types: list = None
):
    """Perform detailed analysis of a single file."""
    console.print("[bold blue]CodeInsight File Analysis[/bold blue]\n")
    
    # Initialize AI Orchestrator
    console.print("Initializing AI models...")
    orchestrator = AIOrchestrator()
    
    # Read file
    file = Path(file_path)
    if not file.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return
        
    code = file.read_text(encoding='utf-8')
    
    # Display code
    console.print("\n[bold blue]Analyzing Code:[/bold blue]")
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)
    
    # Run analysis
    console.print("\n[bold blue]Running Analysis...[/bold blue]")
    results = await orchestrator.analyze_code(
        code,
        language=language,
        analysis_types=analysis_types
    )
    
    # Display results by category
    _display_understanding_results(results)
    _display_pattern_results(results)
    _display_security_results(results)
    _display_performance_results(results)
    
    # Display improvement suggestions
    _display_suggestions(results, orchestrator)


def _display_understanding_results(results):
    """Display code understanding results."""
    if not results.understanding:
        return
        
    console.print("\n[bold blue]Code Understanding[/bold blue]")
    
    # Create understanding table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Aspect")
    table.add_column("Details")
    
    table.add_row("Purpose", results.understanding.purpose)
    table.add_row("Description", results.understanding.description)
    table.add_row(
        "Key Components",
        "\n".join(f"- {comp}" for comp in results.understanding.key_components)
    )
    
    console.print(table)


def _display_pattern_results(results):
    """Display pattern detection results."""
    if not results.patterns and not results.anomalies:
        return
        
    console.print("\n[bold blue]Pattern Analysis[/bold blue]")
    
    # Display patterns
    if results.patterns:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Pattern")
        table.add_column("Type")
        table.add_column("Confidence")
        table.add_column("Location")
        
        for pattern in results.patterns:
            table.add_row(
                pattern.pattern_name,
                pattern.pattern_type,
                f"{pattern.confidence:.2f}",
                f"Lines {pattern.locations[0][0]}-{pattern.locations[0][1]}"
            )
            
        console.print(table)
    
    # Display anomalies
    if results.anomalies:
        console.print("\n[bold red]Detected Anomalies:[/bold red]")
        for anomaly in results.anomalies:
            console.print(Panel(
                f"Type: {anomaly.anomaly_type}\n"
                f"Severity: {anomaly.severity:.2f}\n"
                f"Description: {anomaly.description}\n"
                f"Location: Lines {anomaly.location[0]}-{anomaly.location[1]}",
                title=f"Anomaly (Confidence: {anomaly.confidence:.2f})",
                expand=False
            ))


def _display_security_results(results):
    """Display security analysis results."""
    if not results.vulnerabilities and not results.security_risks:
        return
        
    console.print("\n[bold blue]Security Analysis[/bold blue]")
    
    # Display vulnerabilities
    if results.vulnerabilities:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Vulnerability")
        table.add_column("Severity")
        table.add_column("CVSS")
        table.add_column("Location")
        
        for vuln in results.vulnerabilities:
            table.add_row(
                vuln.vulnerability_type,
                vuln.severity,
                f"{vuln.cvss_score:.1f}",
                f"Lines {vuln.affected_lines[0][0]}-{vuln.affected_lines[0][1]}"
            )
            
        console.print(table)
    
    # Display security risks
    if results.security_risks:
        console.print("\n[bold red]Security Risks:[/bold red]")
        for risk in results.security_risks:
            console.print(Panel(
                f"Type: {risk.risk_type}\n"
                f"Likelihood: {risk.likelihood:.2f}\n"
                f"Impact: {risk.impact:.2f}\n"
                f"Priority: {risk.priority}\n"
                f"Description: {risk.description}",
                title="Security Risk",
                expand=False
            ))


def _display_performance_results(results):
    """Display performance analysis results."""
    if not results.performance_issues:
        return
        
    console.print("\n[bold blue]Performance Analysis[/bold blue]")
    
    # Display resource usage
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    
    table.add_row("Time Complexity", results.resource_usage.time_complexity)
    table.add_row("Space Complexity", results.resource_usage.space_complexity)
    table.add_row("Memory Usage", results.resource_usage.memory_usage)
    table.add_row("I/O Operations", str(results.resource_usage.io_operations))
    
    console.print(table)
    
    # Display performance issues
    if results.performance_issues:
        console.print("\n[bold yellow]Performance Issues:[/bold yellow]")
        for issue in results.performance_issues:
            console.print(Panel(
                f"Type: {issue.issue_type}\n"
                f"Severity: {issue.severity}\n"
                f"Description: {issue.description}\n"
                f"Impact: {issue.impact}\n"
                f"Suggestion: {issue.optimization_suggestion}",
                title=f"Performance Issue (Est. Improvement: {issue.estimated_improvement})",
                expand=False
            ))


def _display_suggestions(results, orchestrator):
    """Display prioritized improvement suggestions."""
    suggestions = orchestrator.get_improvement_suggestions(results)
    
    if not suggestions:
        return
        
    console.print("\n[bold blue]Improvement Suggestions[/bold blue]")
    
    for i, suggestion in enumerate(suggestions, 1):
        console.print(f"{i}. {suggestion}")


if __name__ == "__main__":
    # Example usage
    file_path = "../test_file.py"  # Path to file to analyze
    asyncio.run(analyze_file(file_path))
