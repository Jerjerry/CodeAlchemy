"""
CodeInsight Base Parser Module

This module defines the core parsing infrastructure for the CodeInsight system.
It provides abstract base classes and interfaces for language-specific parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import libcst as cst
from tree_sitter import Language, Parser, Tree, Node
from loguru import logger
from pydantic import BaseModel


@dataclass
class SourceLocation:
    """Represents a location in source code."""
    line: int
    column: int
    end_line: int
    end_column: int
    file_path: Path


class SymbolReference(BaseModel):
    """Represents a symbol reference in the code."""
    name: str
    kind: str  # function, class, variable, etc.
    location: SourceLocation
    scope: str
    is_definition: bool
    documentation: Optional[str]
    type_hints: Optional[str]
    references: List[SourceLocation]


class ParsedEntity(BaseModel):
    """Represents a parsed code entity with its metadata."""
    name: str
    kind: str
    location: SourceLocation
    docstring: Optional[str]
    dependencies: Set[str]
    complexity: float
    code_snippet: str
    symbols: List[SymbolReference]
    ast_node: Any  # Store the AST node for further analysis


class AbstractSyntaxTree(BaseModel):
    """Represents the complete AST of a source file."""
    file_path: Path
    language: str
    entities: List[ParsedEntity]
    imports: List[str]
    global_scope: Dict[str, SymbolReference]
    tree_root: Any  # The root AST node


class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""

    def __init__(self, language: str):
        self.language = language
        self.symbol_table: Dict[str, SymbolReference] = {}
        logger.info(f"Initializing parser for {language}")

    @abstractmethod
    async def parse_file(self, file_path: Path) -> AbstractSyntaxTree:
        """Parse a source file and return its AST."""
        pass

    @abstractmethod
    async def parse_string(self, source: str) -> AbstractSyntaxTree:
        """Parse source code string and return its AST."""
        pass

    @abstractmethod
    def get_symbol_references(self, ast: AbstractSyntaxTree) -> Dict[str, List[SourceLocation]]:
        """Extract all symbol references from the AST."""
        pass

    @abstractmethod
    def get_complexity_metrics(self, ast: AbstractSyntaxTree) -> Dict[str, float]:
        """Calculate complexity metrics for the parsed code."""
        pass

    @abstractmethod
    def extract_documentation(self, ast: AbstractSyntaxTree) -> Dict[str, str]:
        """Extract documentation strings and comments."""
        pass

    def validate_syntax(self, source: str) -> Tuple[bool, Optional[str]]:
        """Validate source code syntax."""
        try:
            ast = self.parse_string(source)
            return True, None
        except Exception as e:
            logger.error(f"Syntax validation failed: {str(e)}")
            return False, str(e)

    async def analyze_dependencies(self, ast: AbstractSyntaxTree) -> Dict[str, Set[str]]:
        """Analyze code dependencies from imports and usage."""
        dependencies = {}
        for entity in ast.entities:
            dependencies[entity.name] = entity.dependencies
        return dependencies

    def get_symbol_info(self, name: str) -> Optional[SymbolReference]:
        """Get information about a specific symbol."""
        return self.symbol_table.get(name)

    def register_symbol(self, symbol: SymbolReference) -> None:
        """Register a new symbol in the symbol table."""
        self.symbol_table[symbol.name] = symbol
        logger.debug(f"Registered symbol: {symbol.name} ({symbol.kind})")


class ParserFactory:
    """Factory class for creating language-specific parsers."""
    
    _parsers: Dict[str, type] = {}

    @classmethod
    def register_parser(cls, language: str, parser_class: type) -> None:
        """Register a parser class for a specific language."""
        cls._parsers[language.lower()] = parser_class
        logger.info(f"Registered parser for {language}")

    @classmethod
    def create_parser(cls, language: str) -> BaseParser:
        """Create a parser instance for the specified language."""
        parser_class = cls._parsers.get(language.lower())
        if not parser_class:
            raise ValueError(f"No parser registered for language: {language}")
        return parser_class(language)

    @classmethod
    def supported_languages(cls) -> Set[str]:
        """Get the set of supported languages."""
        return set(cls._parsers.keys())
