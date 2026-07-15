"""Code chunking for indexing and semantic search.

Extracts functions, methods, and classes from source files into
discrete CodeChunk objects suitable for embedding and retrieval.
"""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeChunk:
    """A semantic unit of code extracted from a source file.

    Attributes:
        chunk_id: Unique identifier (sha256 hash of file_path:name:start_line)
        file_path: Absolute or relative path to the source file
        name: Function, method, class, or module name
        chunk_type: "function" | "method" | "class" | "module"
        start_line: First line number (1-indexed)
        end_line: Last line number (1-indexed)
        source_code: The actual code text
        docstring: Extracted docstring or None
        calls: List of function/method names this chunk calls
        complexity: Cyclomatic complexity score
        file_hash: sha256 hash of the entire source file
    """

    chunk_id: str
    file_path: str
    name: str
    chunk_type: str
    start_line: int
    end_line: int
    source_code: str
    docstring: str | None
    calls: list[str]
    complexity: int
    file_hash: str


class PythonChunker:
    """Chunks Python source code into semantic units using AST parsing."""

    def chunk(self, file_path: Path, source: str, file_hash: str) -> list[CodeChunk]:
        """Parse Python source and extract chunks for indexing.

        Args:
            file_path: Path to the source file
            source: Python source code text
            file_hash: sha256 hash of the source file

        Returns:
            List of CodeChunk objects representing functions, methods, classes,
            and module-level code. Returns empty list on syntax errors.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        chunks: list[CodeChunk] = []
        file_path_str = str(file_path)

        # Track module-level assignments for module chunk
        module_level_lines: list[int] = []

        for node in ast.walk(tree):
            # Handle functions and async functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private functions unless it's __init__
                if node.name.startswith("_") and node.name != "__init__":
                    continue

                # Determine if this is a method (inside a class)
                chunk_type = self._get_function_type(node, tree)

                # Extract docstring
                docstring = ast.get_docstring(node)

                # Extract calls
                calls = self._extract_calls(node)

                # Compute complexity
                complexity = self._compute_complexity(node)

                # Extract source code
                source_lines = source.splitlines()
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                source_code = "\n".join(source_lines[start_line - 1 : end_line])

                # Generate chunk ID
                chunk_id = self._generate_chunk_id(
                    file_path_str, node.name, start_line
                )

                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=file_path_str,
                        name=node.name,
                        chunk_type=chunk_type,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=source_code,
                        docstring=docstring,
                        calls=calls,
                        complexity=complexity,
                        file_hash=file_hash,
                    )
                )

            # Handle classes
            elif isinstance(node, ast.ClassDef):
                # Extract docstring
                docstring = ast.get_docstring(node)

                # For classes, only include header + docstring, not methods
                source_lines = source.splitlines()
                start_line = node.lineno
                
                # Find the end of the class header and docstring
                # Class header is the line with "class ClassName..."
                # If there's a docstring, include it
                if docstring:
                    # Find where docstring ends
                    docstring_node = node.body[0] if node.body and isinstance(node.body[0], ast.Expr) else None
                    end_line = docstring_node.end_lineno if docstring_node else start_line
                else:
                    # Just the class header line
                    end_line = start_line

                source_code = "\n".join(source_lines[start_line - 1 : end_line])

                # Generate chunk ID
                chunk_id = self._generate_chunk_id(
                    file_path_str, node.name, start_line
                )

                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=file_path_str,
                        name=node.name,
                        chunk_type="class",
                        start_line=start_line,
                        end_line=end_line,
                        source_code=source_code,
                        docstring=docstring,
                        calls=[],  # Classes don't have calls
                        complexity=1,  # Base complexity
                        file_hash=file_hash,
                    )
                )

            # Track module-level assignments
            elif isinstance(node, ast.Assign) and self._is_module_level(node, tree):
                if hasattr(node, "lineno"):
                    module_level_lines.append(node.lineno)

        # Create module chunk if there are module-level assignments
        if module_level_lines:
            module_chunk = self._create_module_chunk(
                file_path_str, source, file_hash, module_level_lines
            )
            if module_chunk:
                chunks.append(module_chunk)

        return chunks

    def _get_function_type(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.AST
    ) -> str:
        """Determine if a function is a method or standalone function."""
        # Walk up to see if this function is inside a class
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in ast.walk(parent):
                    if child is node:
                        return "method"
        return "function"

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract function and method call names from an AST node."""
        calls: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Simple function calls: func()
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                # Method calls: obj.method()
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))  # Deduplicate

    def _compute_complexity(self, node: ast.AST) -> int:
        """Compute cyclomatic complexity for an AST node.

        Counts decision points: If, For, While, Try, ExceptHandler,
        BoolOp, IfExp, and comprehension guards. Base complexity is 1.
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.BoolOp,
                    ast.IfExp,
                ),
            ):
                complexity += 1
            # Count comprehension guards
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for generator in child.generators:
                    complexity += len(generator.ifs)
        return complexity

    def _is_module_level(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is at module level (not inside a function or class)."""
        # Simple heuristic: if node is a direct child of Module, it's module-level
        if isinstance(tree, ast.Module):
            return node in tree.body
        return False

    def _create_module_chunk(
        self, file_path: str, source: str, file_hash: str, assignment_lines: list[int]
    ) -> CodeChunk | None:
        """Create a module-level chunk for imports and top-level constants."""
        if not assignment_lines:
            return None

        source_lines = source.splitlines()
        
        # Find all imports and module-level code
        # Typically at the top of the file
        start_line = 1
        end_line = max(assignment_lines)

        # Extract module-level code
        module_code_lines = []
        for i, line in enumerate(source_lines[:end_line], start=1):
            if line.strip() and not line.strip().startswith("#"):
                # Include imports and assignments
                if any(
                    line.strip().startswith(kw)
                    for kw in ["import ", "from ", "class ", "def ", "async def"]
                ):
                    if not line.strip().startswith(("class ", "def ", "async def")):
                        module_code_lines.append((i, line))
                elif i in assignment_lines:
                    module_code_lines.append((i, line))

        if not module_code_lines:
            return None

        # Build source code from selected lines
        source_code = "\n".join(line for _, line in module_code_lines)
        start_line = module_code_lines[0][0]
        end_line = module_code_lines[-1][0]

        chunk_id = self._generate_chunk_id(file_path, "__module__", start_line)

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            name="__module__",
            chunk_type="module",
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            docstring=None,
            calls=[],
            complexity=1,
            file_hash=file_hash,
        )

    def _generate_chunk_id(self, file_path: str, name: str, start_line: int) -> str:
        """Generate a unique chunk ID from file path, name, and line number."""
        key = f"{file_path}:{name}:{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
