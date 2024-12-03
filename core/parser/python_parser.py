"""
CodeInsight Python Parser Module

This module implements Python-specific parsing functionality using multiple parsing
strategies for maximum accuracy and information extraction.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import symtable
from dataclasses import dataclass

import libcst as cst
from libcst.metadata import MetadataWrapper
import astroid
from typed_ast import ast3
from loguru import logger

from .base_parser import (
    BaseParser,
    AbstractSyntaxTree,
    ParsedEntity,
    SourceLocation,
    SymbolReference,
    ParserFactory
)


class PythonSymbolCollector(cst.CSTVisitor):
    """Collects symbol information from Python code using LibCST."""
    
    def __init__(self):
        self.symbols: List[SymbolReference] = []
        self.current_scope: List[str] = ["global"]
        
    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        """Process function definitions."""
        location = SourceLocation(
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
            file_path=Path()  # Will be set later
        )
        
        symbol = SymbolReference(
            name=node.name.value,
            kind="function",
            location=location,
            scope="::".join(self.current_scope),
            is_definition=True,
            documentation=ast.get_docstring(node),
            type_hints=self._extract_type_hints(node),
            references=[]
        )
        self.symbols.append(symbol)
        self.current_scope.append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Leave function scope."""
        self.current_scope.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        """Process class definitions."""
        location = SourceLocation(
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
            file_path=Path()
        )
        
        symbol = SymbolReference(
            name=node.name.value,
            kind="class",
            location=location,
            scope="::".join(self.current_scope),
            is_definition=True,
            documentation=ast.get_docstring(node),
            type_hints=None,
            references=[]
        )
        self.symbols.append(symbol)
        self.current_scope.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Leave class scope."""
        self.current_scope.pop()

    def _extract_type_hints(self, node: cst.FunctionDef) -> Optional[str]:
        """Extract type hints from function definition."""
        try:
            returns = node.returns.annotation.value if node.returns else None
            params = []
            for param in node.params:
                if param.annotation:
                    params.append(f"{param.name.value}: {param.annotation.annotation.value}")
                else:
                    params.append(param.name.value)
            
            signature = f"({', '.join(params)})"
            if returns:
                signature += f" -> {returns}"
            return signature
        except Exception:
            return None


class PythonParser(BaseParser):
    """Python-specific parser implementation."""

    def __init__(self):
        super().__init__("Python")
        self.ast_parser = ast.parse
        self.libcst_parser = cst.parse_module
        self.astroid_parser = astroid.parse

    async def parse_file(self, file_path: Path) -> AbstractSyntaxTree:
        """Parse a Python source file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            return await self.parse_string(source, file_path)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise

    async def parse_string(self, source: str, file_path: Optional[Path] = None) -> AbstractSyntaxTree:
        """Parse Python source code string."""
        try:
            # Parse with multiple parsers for different aspects
            ast_tree = self.ast_parser(source)
            cst_tree = self.libcst_parser(source)
            astroid_tree = self.astroid_parser(source)

            # Collect symbols
            symbol_collector = PythonSymbolCollector()
            cst_tree.visit(symbol_collector)

            # Extract entities
            entities = self._extract_entities(ast_tree, cst_tree, astroid_tree)

            # Create AST representation
            return AbstractSyntaxTree(
                file_path=file_path or Path(),
                language="Python",
                entities=entities,
                imports=self._extract_imports(ast_tree),
                global_scope=self._build_global_scope(symbol_collector.symbols),
                tree_root=ast_tree
            )
        except Exception as e:
            logger.error(f"Error parsing Python code: {str(e)}")
            raise

    def _extract_entities(
        self,
        ast_tree: ast.AST,
        cst_tree: cst.Module,
        astroid_tree: astroid.Module
    ) -> List[ParsedEntity]:
        """Extract code entities from multiple AST representations."""
        entities = []
        
        # Process functions
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entity = self._process_function(node, cst_tree, astroid_tree)
                entities.append(entity)
            elif isinstance(node, ast.ClassDef):
                entity = self._process_class(node, cst_tree, astroid_tree)
                entities.append(entity)

        return entities

    def _process_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        cst_tree: cst.Module,
        astroid_tree: astroid.Module
    ) -> ParsedEntity:
        """Process a function definition node."""
        location = SourceLocation(
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno or node.lineno,
            end_column=node.end_col_offset or 0,
            file_path=Path()
        )

        return ParsedEntity(
            name=node.name,
            kind="function",
            location=location,
            docstring=ast.get_docstring(node),
            dependencies=self._get_function_dependencies(node),
            complexity=self._calculate_complexity(node),
            code_snippet=self._extract_code_snippet(node),
            symbols=self._extract_function_symbols(node),
            ast_node=node
        )

    def _process_class(
        self,
        node: ast.ClassDef,
        cst_tree: cst.Module,
        astroid_tree: astroid.Module
    ) -> ParsedEntity:
        """Process a class definition node."""
        location = SourceLocation(
            line=node.lineno,
            column=node.col_offset,
            end_line=node.end_lineno or node.lineno,
            end_column=node.end_col_offset or 0,
            file_path=Path()
        )

        return ParsedEntity(
            name=node.name,
            kind="class",
            location=location,
            docstring=ast.get_docstring(node),
            dependencies=self._get_class_dependencies(node),
            complexity=self._calculate_complexity(node),
            code_snippet=self._extract_code_snippet(node),
            symbols=self._extract_class_symbols(node),
            ast_node=node
        )

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _extract_code_snippet(self, node: ast.AST) -> str:
        """Extract the source code for an AST node."""
        # This is a placeholder - in real implementation, we'd use the source code
        return ast.unparse(node)

    def _get_function_dependencies(self, node: ast.FunctionDef) -> Set[str]:
        """Extract function dependencies."""
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                deps.add(child.id)
        return deps

    def _get_class_dependencies(self, node: ast.ClassDef) -> Set[str]:
        """Extract class dependencies."""
        deps = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                deps.add(base.id)
        return deps

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from the AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        return imports

    def _build_global_scope(self, symbols: List[SymbolReference]) -> Dict[str, SymbolReference]:
        """Build global scope symbol table."""
        return {
            symbol.name: symbol
            for symbol in symbols
            if symbol.scope == "global"
        }

    def _extract_function_symbols(self, node: ast.FunctionDef) -> List[SymbolReference]:
        """Extract symbols from a function definition."""
        symbols = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                location = SourceLocation(
                    line=child.lineno,
                    column=child.col_offset,
                    end_line=child.lineno,
                    end_column=child.col_offset + len(child.id),
                    file_path=Path()
                )
                symbols.append(SymbolReference(
                    name=child.id,
                    kind="variable",
                    location=location,
                    scope=node.name,
                    is_definition=isinstance(child.ctx, ast.Store),
                    documentation=None,
                    type_hints=None,
                    references=[]
                ))
        return symbols

    def _extract_class_symbols(self, node: ast.ClassDef) -> List[SymbolReference]:
        """Extract symbols from a class definition."""
        symbols = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                location = SourceLocation(
                    line=child.lineno,
                    column=child.col_offset,
                    end_line=child.lineno,
                    end_column=child.col_offset + len(child.id),
                    file_path=Path()
                )
                symbols.append(SymbolReference(
                    name=child.id,
                    kind="attribute" if isinstance(child.ctx, ast.Store) else "reference",
                    location=location,
                    scope=node.name,
                    is_definition=isinstance(child.ctx, ast.Store),
                    documentation=None,
                    type_hints=None,
                    references=[]
                ))
        return symbols


# Register the Python parser
ParserFactory.register_parser("Python", PythonParser)
