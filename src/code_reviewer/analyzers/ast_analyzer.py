"""AST-based deterministic static analysis for Python files."""

import ast
from typing import List, Any
from code_reviewer.core.models import Finding
from code_reviewer.config import Settings

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
        
    def _analyze_function(self, node: ast.AST):
        prev_function = self.current_function
        self.current_function = node
        
        # 1. Function Length Check
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            length = node.end_lineno - node.lineno
            if length > self.settings.rules.complexity.max_function_length:
                self.report_finding(
                    node,
                    severity="MEDIUM",
                    category="complexity",
                    message=f"Function '{node.name}' is too long ({length} lines).",
                    suggestion="Extract logic into smaller helper functions."
                )
                
        # 2. Missing Docstring Check
        if not node.name.startswith("_") or node.name == "__init__":
            if not ast.get_docstring(node):
                self.report_finding(
                    node,
                    severity="LOW",
                    category="docs",
                    message=f"Missing docstring in public function '{node.name}'.",
                    suggestion="Add a docstring explaining the function's purpose."
                )
                
        # 3. Mutable Defaults Check
        if hasattr(node, 'args'):
            for default in node.args.defaults + getattr(node.args, 'kw_defaults', []):
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    self.report_finding(
                        default,
                        severity="HIGH",
                        category="logic",
                        message=f"Mutable default argument found in '{node.name}'.",
                        suggestion="Use None as default and initialize the mutable object inside the function body."
                    )
                    
        # 4. Cyclomatic Complexity Check (Approximation)
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.AsyncFor, ast.AsyncWith, ast.IfExp)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                # +1 for the implicit loop, +1 for each 'if' condition
                complexity += 1 + len(child.ifs)
                
        max_cc = self.settings.rules.complexity.max_cyclomatic_complexity
        if complexity > 15:
            self.report_finding(
                node,
                severity="HIGH",
                category="complexity",
                message=f"Function '{node.name}' has very high cyclomatic complexity ({complexity}).",
                suggestion="Refactor to simplify logic and reduce branching."
            )
        elif complexity > max_cc:
            self.report_finding(
                node,
                severity="MEDIUM",
                category="complexity",
                message=f"Function '{node.name}' has high cyclomatic complexity ({complexity}).",
                suggestion="Refactor to simplify logic and reduce branching."
            )

        # Walk body to evaluate internal nodes (like blocks for nesting, and magic numbers)
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_ClassDef(self, node: ast.ClassDef):
        # 5. Missing Docstring Check for classes
        if not node.name.startswith("_"):
            if not ast.get_docstring(node):
                self.report_finding(
                    node,
                    severity="LOW",
                    category="docs",
                    message=f"Missing docstring in public class '{node.name}'.",
                    suggestion="Add a docstring explaining the class's purpose."
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
        # 6. Magic Numbers
        # Excludes 0, 1, and booleans
        if isinstance(node.value, (int, float)) and type(node.value) is not bool:
            if node.value not in [0, 1]:
                is_constant_assign = False
                if self.in_assign:
                    is_constant_assign = any(t.isupper() for t in self.assign_targets)
                
                # We also might want to check if it's outside a function (like module level constant).
                # But the rule applies to any naked magic number not assigned to an uppercase var.
                if not is_constant_assign:
                    self.report_finding(
                        node,
                        severity="INFO",
                        category="style",
                        message=f"Magic number {node.value} found.",
                        suggestion="Extract this number into a named constant (e.g., MAX_ITEMS = ...)."
                    )
        self.generic_visit(node)

    # --- Depth calculations for Blocks ---
    def _handle_block(self, node: ast.AST):
        self.current_depth += 1
        
        max_depth = self.settings.rules.complexity.max_nesting_depth
        if self.current_depth > 6:
            self.report_finding(
                node,
                severity="HIGH",
                category="complexity",
                message=f"Extremely deep nesting ({self.current_depth} levels).",
                suggestion="Extract nested blocks into separate functions."
            )
        elif self.current_depth > max_depth:
            self.report_finding(
                node,
                severity="MEDIUM",
                category="complexity",
                message=f"Deep nesting ({self.current_depth} levels).",
                suggestion="Extract nested blocks into separate functions or return early."
            )
            
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node: ast.If): self._handle_block(node)
    def visit_For(self, node: ast.For): self._handle_block(node)
    def visit_While(self, node: ast.While): self._handle_block(node)
    def visit_Try(self, node: ast.Try): self._handle_block(node)
    def visit_With(self, node: ast.With): self._handle_block(node)
