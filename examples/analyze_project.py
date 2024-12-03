"""
Example script demonstrating comprehensive project analysis using CodeInsight.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time

from ai.orchestrator import AIOrchestrator

# Initialize rich console
console = Console()

async def analyze_project(
    project_path: str,
    language: str = "python",
    min_confidence: float = 0.7
):
    """Analyze an entire project directory."""
    console.print("[bold blue]CodeInsight Project Analysis[/bold blue]\n")
    
    # Initialize AI Orchestrator
    console.print("Initializing AI models...")
    orchestrator = AIOrchestrator()
    
    # Get all code files
    project_dir = Path(project_path)
    code_files = list(project_dir.rglob(f"*.{language}"))
    
    if not code_files:
        console.print(f"[red]No {language} files found in {project_path}[/red]")
        return
    
    console.print(f"\nFound {len(code_files)} {language} files to analyze.\n")
    
    # Analyze each file
    all_results = []
    for file in track(code_files, description="Analyzing files..."):
        try:
            # Read file
            code = file.read_text(encoding='utf-8')
            
            # Run analysis
            results = await orchestrator.analyze_code(
                code,
                language=language
            )
            
            # Store results
            all_results.append({
                'file': file,
                'results': results
            })
            
        except Exception as e:
            console.print(f"[red]Error analyzing {file}: {str(e)}[/red]")
    
    # Display summary
    _display_project_summary(all_results, min_confidence)
    
    # Generate recommendations
    _display_project_recommendations(all_results, orchestrator)


def _display_project_summary(results: list, min_confidence: float):
    """Display project-wide analysis summary."""
    console.print("\n[bold blue]Project Analysis Summary[/bold blue]")
    
    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Count")
    
    # Calculate totals
    total_vulnerabilities = sum(
        len([v for v in r['results'].vulnerabilities if v.confidence >= min_confidence])
        for r in results
    )
    total_patterns = sum(
        len([p for p in r['results'].patterns if p.confidence >= min_confidence])
        for r in results
    )
    total_performance_issues = sum(
        len([i for i in r['results'].performance_issues if i.confidence >= min_confidence])
        for r in results
    )
    
    # Add rows
    table.add_row("Files Analyzed", str(len(results)))
    table.add_row("Security Vulnerabilities", str(total_vulnerabilities))
    table.add_row("Code Patterns", str(total_patterns))
    table.add_row("Performance Issues", str(total_performance_issues))
    
    console.print(table)


def _display_project_recommendations(results: list, orchestrator: AIOrchestrator):
    """Display project-wide recommendations."""
    console.print("\n[bold blue]Project Recommendations[/bold blue]")
    
    # Get all suggestions
    all_suggestions = []
    for r in results:
        suggestions = orchestrator.get_improvement_suggestions(r['results'])
        all_suggestions.extend([
            (str(r['file']), suggestion)
            for suggestion in suggestions
        ])
    
    # Sort by priority
    all_suggestions.sort(key=lambda x: (
        'SECURITY' in x[1],
        'PERFORMANCE' in x[1],
        'PATTERN' in x[1]
    ), reverse=True)
    
    # Display top suggestions
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File")
    table.add_column("Suggestion")
    
    for file, suggestion in all_suggestions[:10]:  # Show top 10
        table.add_row(file, suggestion)
    
    console.print(table)


if __name__ == "__main__":
    # Example usage
    project_path = "../test_project"  # Path to project to analyze
    asyncio.run(analyze_project(project_path))
