"""AST-based deterministic static analysis for Python files."""

import ast
from collections import deque
from typing import List
from code_reviewer.core.models import Finding
from code_reviewer.config import Settings

# When cyclomatic complexity exceeds max_cc by this factor, severity is HIGH.
_HIGH_CC_MULTIPLIER = 1.5


class ASTAnalyzer(ast.NodeVisitor):
    """Analyzes Python AST to find deterministic code issues."""

    def __init__(self, file_path: str, settings: Settings):
        self.file_path = file_path
        self.settings = settings
        self.findings: List[Finding] = []

        # State tracking for rules
        self.current_depth = 0
        self.current_function = None
        self.in_assign = False
        self.assign_targets = []

    def report_finding(self, node: ast.AST, severity: str, category: str, message: str, suggestion: str):
        self.findings.append(Finding(
            file_path=self.file_path,
            line_number=getattr(node, 'lineno', 1),
            severity=severity,
            category=category,
            message=message,
            suggestion=suggestion,
            source="ast"
        ))

    def analyze(self, source_code: str) -> List[Finding]:
        """Parses the source code and walks the AST."""
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
        except SyntaxError as e:
            # Report parse failures as high severity issues
            self.findings.append(Finding(
                file_path=self.file_path,
                line_number=e.lineno or 1,
                severity="HIGH",
                category="logic",
                message=f"Syntax error prevents AST analysis: {e.msg}",
                suggestion="Fix syntax error.",
                source="ast"
            ))
        return self.findings

    # --- Node Visitors for specific rules ---

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._analyze_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._analyze_function(node)

    def _walk_no_nested_funcs(self, node):
        """BFS walk that does not descend into nested function definitions."""
        todo = deque([node])
        while todo:
            curr = todo.popleft()
            yield curr
            for child in ast.iter_child_nodes(curr):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    todo.append(child)

    # ------------------------------------------------------------------
    # Per-rule check helpers (called from _analyze_function)
    # ------------------------------------------------------------------

    def _check_function_length(self, node: ast.AST) -> None:
        """Report if the function body exceeds the configured line limit."""
        if not self.settings.rules.complexity.enabled:
            return
        if not (hasattr(node, 'end_lineno') and hasattr(node, 'lineno')):
            return
        length = node.end_lineno - node.lineno
        if length > self.settings.rules.complexity.max_function_length:
            self.report_finding(
                node,
                severity="MEDIUM",
                category="complexity",
                message=f"Function '{node.name}' is too long ({length} lines).",
                suggestion="Extract logic into smaller helper functions.",
            )

    def _check_missing_docstring(self, node: ast.AST) -> None:
        """Report missing docstrings on public functions."""
        if not self.settings.rules.docs.enabled:
            return
        if not node.name.startswith("_") or node.name == "__init__":
            if not ast.get_docstring(node):
                self.report_finding(
                    node,
                    severity="LOW",
                    category="docs",
                    message=f"Missing docstring in public function '{node.name}'.",
                    suggestion="Add a docstring explaining the function's purpose.",
                )

    def _check_mutable_defaults(self, node: ast.AST) -> None:
        """Report mutable default argument values (list/dict/set literals)."""
        if not self.settings.rules.mutable_defaults.enabled:
            return
        if not hasattr(node, 'args'):
            return
        for default in node.args.defaults + getattr(node.args, 'kw_defaults', []):
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.report_finding(
                    default,
                    severity="HIGH",
                    category="logic",
                    message=f"Mutable default argument found in '{node.name}'.",
                    suggestion=(
                        "Use None as default and initialize the mutable object "
                        "inside the function body."
                    ),
                )

    def _check_cyclomatic_complexity(self, node: ast.AST) -> None:
        """Report functions whose cyclomatic complexity exceeds configured thresholds."""
        if not self.settings.rules.complexity.enabled:
            return

        complexity = 1
        for child in self._walk_no_nested_funcs(node):
            if isinstance(child, (
                ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler,
                ast.With, ast.AsyncFor, ast.AsyncWith, ast.IfExp,
            )):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                # +1 for the implicit loop, +1 for each 'if' condition
                complexity += 1 + len(child.ifs)

        max_cc = self.settings.rules.complexity.max_cyclomatic_complexity
        high_cc_threshold = max(int(max_cc * _HIGH_CC_MULTIPLIER), 15)

        if complexity > high_cc_threshold:
            self.report_finding(
                node,
                severity="HIGH",
                category="complexity",
                message=f"Function '{node.name}' has very high cyclomatic complexity ({complexity}).",
                suggestion="Refactor to simplify logic and reduce branching.",
            )
        elif complexity > max_cc:
            self.report_finding(
                node,
                severity="MEDIUM",
                category="complexity",
                message=f"Function '{node.name}' has high cyclomatic complexity ({complexity}).",
                suggestion="Refactor to simplify logic and reduce branching.",
            )

    # ------------------------------------------------------------------
    # Main function dispatcher
    # ------------------------------------------------------------------

    def _analyze_function(self, node: ast.AST):
        prev_function = self.current_function
        prev_depth = self.current_depth
        self.current_function = node
        self.current_depth = 0

        self._check_function_length(node)
        self._check_missing_docstring(node)
        self._check_mutable_defaults(node)
        self._check_cyclomatic_complexity(node)

        # Walk body to evaluate internal nodes (nesting depth, magic numbers)
        self.generic_visit(node)
        self.current_function = prev_function
        self.current_depth = prev_depth

    def visit_ClassDef(self, node: ast.ClassDef):
        # Missing Docstring Check for classes
        if self.settings.rules.docs.enabled:
            if not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    self.report_finding(
                        node,
                        severity="LOW",
                        category="docs",
                        message=f"Missing docstring in public class '{node.name}'.",
                        suggestion="Add a docstring explaining the class's purpose.",
                    )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Track assignments so we can ignore Magic Numbers bound to ALL_CAPS constants
        self.in_assign = True
        self.assign_targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assign_targets.append(target.id)
        self.generic_visit(node)
        self.in_assign = False
        self.assign_targets = []

    def visit_Constant(self, node: ast.Constant):
        # Magic Numbers — excludes 0, 1, and booleans
        if self.settings.rules.magic_numbers.enabled:
            if isinstance(node.value, (int, float)) and type(node.value) is not bool:
                if node.value not in [0, 1]:
                    is_constant_assign = (
                        self.in_assign
                        and any(t.isupper() for t in self.assign_targets)
                    )
                    if not is_constant_assign:
                        self.report_finding(
                            node,
                            severity="INFO",
                            category="style",
                            message=f"Magic number {node.value} found.",
                            suggestion="Extract this number into a named constant (e.g., MAX_ITEMS = ...).",
                        )
        self.generic_visit(node)

    # --- Depth calculations for Blocks ---
    def _handle_block(self, node: ast.AST):
        self.current_depth += 1

        if self.settings.rules.nesting.enabled:
            max_depth = self.settings.rules.nesting.max_nesting_depth
            if self.current_depth > 6:
                self.report_finding(
                    node,
                    severity="HIGH",
                    category="complexity",
                    message=f"Extremely deep nesting ({self.current_depth} levels).",
                    suggestion="Extract nested blocks into separate functions.",
                )
            elif self.current_depth > max_depth:
                self.report_finding(
                    node,
                    severity="MEDIUM",
                    category="complexity",
                    message=f"Deep nesting ({self.current_depth} levels).",
                    suggestion="Extract nested blocks into separate functions or return early.",
                )

        self.generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node: ast.If): self._handle_block(node)
    def visit_For(self, node: ast.For): self._handle_block(node)
    def visit_While(self, node: ast.While): self._handle_block(node)
    def visit_Try(self, node: ast.Try): self._handle_block(node)
    def visit_With(self, node: ast.With): self._handle_block(node)
    def visit_AsyncFor(self, node: ast.AsyncFor): self._handle_block(node)
    def visit_AsyncWith(self, node: ast.AsyncWith): self._handle_block(node)
