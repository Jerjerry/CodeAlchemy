"""
CodeInsight CLI Interface

A professional command-line interface for the CodeInsight code analysis system.
Provides rich formatting, detailed logging, and comprehensive analysis options.
"""

import typer
from typing import List, Optional
from pathlib import Path
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import time
from loguru import logger
import json
import yaml

from ai.orchestrator import AIOrchestrator
from core.parser.python_parser import PythonParser

# Initialize Typer app with rich formatting
app = typer.Typer(
    name="codeinsight",
    help="Advanced AI-powered code analysis system",
    add_completion=False
)
console = Console()

# Configure logging
logger.add(
    "logs/codeinsight.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


@app.command()
def analyze(
    path: Path = typer.Argument(
        ...,
        help="Path to file or directory to analyze",
        exists=True
    ),
    language: str = typer.Option(
        "python",
        "--language", "-l",
        help="Programming language of the code"
    ),
    output: str = typer.Option(
        "markdown",
        "--output", "-o",
        help="Output format (markdown, html, json)"
    ),
    analysis_types: Optional[List[str]] = typer.Option(
        None,
        "--types", "-t",
        help="Specific analysis types to run (understanding,patterns,security,performance)"
    ),
    context_lines: int = typer.Option(
        5,
        "--context", "-c",
        help="Number of context lines for analysis"
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--confidence",
        help="Minimum confidence threshold for results"
    ),
    save_to: Optional[Path] = typer.Option(
        None,
        "--save",
        help="Save analysis results to file"
    )
):
    """
    Perform comprehensive code analysis using AI models.
    """
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing AI models...", total=None)
            orchestrator = AIOrchestrator()

        # Read and parse code
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob(f"*.{language}"))

        if not files:
            console.print(f"[red]No {language} files found in {path}[/red]")
            raise typer.Exit(1)

        # Analyze each file
        all_results = []
        for file in files:
            console.print(f"\n[bold blue]Analyzing {file}...[/bold blue]")
            
            # Read file
            code = file.read_text(encoding='utf-8')
            
            # Get context
            context = _get_file_context(file, context_lines)
            
            # Run analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task("Running AI analysis...", total=None)
                results = asyncio.run(orchestrator.analyze_code(
                    code,
                    language=language,
                    context=context,
                    analysis_types=analysis_types
                ))
            
            # Filter by confidence
            results = _filter_by_confidence(results, min_confidence)
            
            # Generate report
            report = orchestrator.generate_report(results, format=output)
            
            # Get improvement suggestions
            suggestions = orchestrator.get_improvement_suggestions(results)
            
            # Store results
            all_results.append({
                'file': str(file),
                'results': results,
                'report': report,
                'suggestions': suggestions
            })
            
            # Display results
            _display_results(results, report, suggestions)

        # Save results if requested
        if save_to:
            _save_results(all_results, save_to, output)
            console.print(f"\n[green]Results saved to {save_to}[/green]")

    except Exception as e:
        logger.exception("Analysis failed")
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def explain(
    path: Path = typer.Argument(
        ...,
        help="Path to file to explain",
        exists=True
    ),
    aspect: Optional[str] = typer.Option(
        None,
        "--aspect", "-a",
        help="Specific aspect to explain (understanding,patterns,security,performance)"
    )
):
    """
    Generate natural language explanation of code.
    """
    try:
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing AI models...", total=None)
            orchestrator = AIOrchestrator()

        # Read code
        code = path.read_text(encoding='utf-8')
        
        # Run analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Analyzing code...", total=None)
            results = asyncio.run(orchestrator.analyze_code(code))
        
        # Generate explanation
        explanation = orchestrator.explain_analysis(results, aspect)
        
        # Display explanation
        console.print("\n[bold blue]Code Explanation[/bold blue]")
        console.print(Panel(explanation, expand=False))

    except Exception as e:
        logger.exception("Explanation failed")
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _get_file_context(file: Path, context_lines: int) -> List[str]:
    """Get surrounding context lines from file."""
    try:
        with file.open('r', encoding='utf-8') as f:
            return f.readlines()[-context_lines:]
    except Exception:
        return []


def _filter_by_confidence(results: any, min_confidence: float) -> any:
    """Filter analysis results by confidence threshold."""
    # Filter vulnerabilities
    results.vulnerabilities = [
        v for v in results.vulnerabilities
        if v.confidence >= min_confidence
    ]
    
    # Filter patterns
    results.patterns = [
        p for p in results.patterns
        if p.confidence >= min_confidence
    ]
    
    # Filter performance issues
    results.performance_issues = [
        i for i in results.performance_issues
        if i.confidence >= min_confidence
    ]
    
    return results


def _display_results(results: any, report: str, suggestions: List[str]):
    """Display analysis results with rich formatting."""
    # Display report
    console.print("\n[bold blue]Analysis Report[/bold blue]")
    console.print(Panel(report, expand=False))
    
    # Display suggestions table
    if suggestions:
        console.print("\n[bold blue]Improvement Suggestions[/bold blue]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Priority")
        table.add_column("Suggestion")
        
        for i, suggestion in enumerate(suggestions, 1):
            table.add_row(f"#{i}", suggestion)
        
        console.print(table)


def _save_results(results: List[dict], save_path: Path, format: str):
    """Save analysis results to file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert results to JSON-serializable format
        json_results = []
        for r in results:
            json_results.append({
                'file': r['file'],
                'report': r['report'],
                'suggestions': r['suggestions']
            })
        save_path.write_text(json.dumps(json_results, indent=2))
        
    elif format == 'yaml':
        # Convert results to YAML format
        yaml_results = []
        for r in results:
            yaml_results.append({
                'file': r['file'],
                'report': r['report'],
                'suggestions': r['suggestions']
            })
        save_path.write_text(yaml.dump(yaml_results))
        
    else:
        # Save as markdown
        with save_path.open('w') as f:
            for r in results:
                f.write(f"\n# Analysis Results for {r['file']}\n\n")
                f.write(r['report'])
                f.write("\n## Improvement Suggestions\n\n")
                for i, suggestion in enumerate(r['suggestions'], 1):
                    f.write(f"{i}. {suggestion}\n")


if __name__ == "__main__":
    app()
