"""Multi-language static analysis using Tree-sitter."""

import os
from typing import List
import tree_sitter

try:
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_java
    HAVE_TREESITTER_LANGS = True
except ImportError:
    HAVE_TREESITTER_LANGS = False

from code_reviewer.core.models import Finding
from code_reviewer.config import Settings
from code_reviewer.analyzers.ast_analyzer import ASTAnalyzer, _HIGH_CC_MULTIPLIER

# Tree-sitter node types that count as branch points for cyclomatic complexity.
_CC_BRANCH_TYPES = frozenset([
    "if_statement", "for_statement", "while_statement",
    "catch_clause", "ternary_expression", "&&", "||",
])


def _get_ts_line(node) -> int:
    """Safely extract 1-indexed line number from a tree-sitter node."""
    if hasattr(node, "start_point") and node.start_point is not None:
        return node.start_point[0] + 1
    return 1


def _get_ts_length(node) -> int:
    """Safely extract length in lines from a tree-sitter node."""
    if hasattr(node, "start_point") and node.start_point is not None and hasattr(node, "end_point") and node.end_point is not None:
        return node.end_point[0] - node.start_point[0]
    return 0



class StaticAnalyzer:
    """Wrapper that delegates to ASTAnalyzer (Python) or Tree-sitter (Multi-language)."""

    def __init__(self, settings: Settings):
        self.settings = settings

        self.parsers = {}
        if HAVE_TREESITTER_LANGS:
            self.parsers["python"] = tree_sitter.Language(tree_sitter_python.language())
            self.parsers["javascript"] = tree_sitter.Language(tree_sitter_javascript.language())
            self.parsers["java"] = tree_sitter.Language(tree_sitter_java.language())

    def analyze_file(self, file_path: str, source_code: str) -> List[Finding]:
        """Analyse a single source file and return all findings.

        Args:
            file_path: Absolute or repo-relative path used for labelling findings.
            source_code: Full source text of the file.

        Returns:
            List of Finding objects produced by static analysis.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # 1. Prefer pure Python AST for .py files due to higher precision
        if ext == ".py":
            analyzer = ASTAnalyzer(file_path=file_path, settings=self.settings)
            return analyzer.analyze(source_code)

        # 2. Fall back to Tree-sitter for non-Python languages
        lang = None
        if ext in [".js", ".jsx"]:
            lang = "javascript"
        elif ext == ".java":
            lang = "java"

        if not lang or lang not in self.parsers:
            return []  # Unsupported language fallback

        return self._analyze_with_treesitter(file_path, source_code, lang)

    def _analyze_with_treesitter(self, file_path: str, source_code: str, lang_name: str) -> List[Finding]:
        """Run Tree-sitter analysis on a non-Python file.

        Args:
            file_path: Path used for labelling.
            source_code: Full source text.
            lang_name: Key into self.parsers ('javascript' or 'java').

        Returns:
            List of Finding objects.
        """
        parser = tree_sitter.Parser(self.parsers[lang_name])
        tree = parser.parse(bytes(source_code, "utf8"))

        findings: List[Finding] = []

        # First pass: nesting depth across the whole tree
        self._check_ts_nesting(file_path, tree.root_node, findings)

        # Second pass: function-scoped rules
        self._analyze_functions_treesitter(file_path, tree.root_node, source_code, findings)

        return findings

    # ------------------------------------------------------------------
    # Tree-sitter check helpers
    # ------------------------------------------------------------------

    def _check_ts_nesting(self, file_path: str, root_node, findings: List[Finding]) -> None:
        """Walk the tree and report nodes that exceed the configured nesting depth.

        Args:
            file_path: Path used for labelling.
            root_node: The tree-sitter root node.
            findings: List to append new Finding objects to.
        """
        if not self.settings.rules.nesting.enabled:
            return

        max_depth = self.settings.rules.nesting.max_nesting_depth

        def walk(node, depth: int) -> None:
            is_block = "block" in node.type or "statement_block" in node.type
            if is_block:
                depth += 1
                if depth > 6:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=_get_ts_line(node),
                        severity="HIGH",
                        category="complexity",
                        message=f"Extremely deep nesting ({depth} levels).",
                        suggestion="Extract nested blocks into separate functions.",
                        source="ast",
                    ))
                elif depth > max_depth:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=_get_ts_line(node),
                        severity="MEDIUM",
                        category="complexity",
                        message=f"Deep nesting ({depth} levels).",
                        suggestion="Extract nested blocks into separate functions.",
                        source="ast",
                    ))

            for child in node.children:
                walk(child, depth)

        walk(root_node, 0)

    def _check_ts_length(
        self,
        file_path: str,
        func,
        findings: List[Finding],
    ) -> None:
        """Report if a tree-sitter function node exceeds the configured length limit.

        Args:
            file_path: Path used for labelling.
            func: Tree-sitter function node.
            findings: List to append new Finding objects to.
        """
        if not self.settings.rules.complexity.enabled:
            return
        length = _get_ts_length(func)
        if length > self.settings.rules.complexity.max_function_length:
            findings.append(Finding(
                file_path=file_path,
                line_number=_get_ts_line(func),
                severity="MEDIUM",
                category="complexity",
                message=f"Function is too long ({length} lines).",
                suggestion="Extract logic into smaller helper functions.",
                source="ast",
            ))

    def _check_ts_complexity(
        self,
        file_path: str,
        func,
        findings: List[Finding],
    ) -> None:
        """Report cyclomatic complexity for a tree-sitter function node.

        Args:
            file_path: Path used for labelling.
            func: Tree-sitter function node.
            findings: List to append new Finding objects to.
        """
        if not self.settings.rules.complexity.enabled:
            return

        cc = 1

        def count_cc(node) -> None:
            nonlocal cc
            if node.type in _CC_BRANCH_TYPES:
                cc += 1
            if node.type == "binary_expression":
                op = node.child_by_field_name("operator")
                if op and op.type in {"&&", "||"}:
                    cc += 1
            for child in node.children:
                # Do not leak complexity into inner functions
                if "function" not in child.type and "method" not in child.type:
                    count_cc(child)

        for child in func.children:
            count_cc(child)

        max_cc = self.settings.rules.complexity.max_cyclomatic_complexity
        high_cc_threshold = max(int(max_cc * _HIGH_CC_MULTIPLIER), 15)

        if cc > high_cc_threshold:
            findings.append(Finding(
                file_path=file_path,
                line_number=_get_ts_line(func),
                severity="HIGH",
                category="complexity",
                message=f"Function has very high cyclomatic complexity ({cc}).",
                suggestion="Refactor to simplify logic and reduce branching.",
                source="ast",
            ))
        elif cc > max_cc:
            findings.append(Finding(
                file_path=file_path,
                line_number=_get_ts_line(func),
                severity="MEDIUM",
                category="complexity",
                message=f"Function has high cyclomatic complexity ({cc}).",
                suggestion="Refactor to simplify logic and reduce branching.",
                source="ast",
            ))

    def _check_ts_docs(
        self,
        file_path: str,
        func,
        source_code: str,
        findings: List[Finding],
    ) -> None:
        """Report missing JSDoc/JavaDoc comments on public tree-sitter functions.

        Args:
            file_path: Path used for labelling.
            func: Tree-sitter function node.
            source_code: Full source text (used to extract comment text).
            findings: List to append new Finding objects to.
        """
        if not self.settings.rules.docs.enabled:
            return

        prev = func.prev_sibling
        has_doc = False
        while prev and prev.type == "comment":
            text = source_code[prev.start_byte:prev.end_byte]
            if text.startswith("/**"):
                has_doc = True
                break
            prev = prev.prev_sibling

        if not has_doc:
            name_node = func.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte]
                if not name.startswith("_"):
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=_get_ts_line(func),
                        severity="LOW",
                        category="docs",
                        message=f"Missing docstring in public function '{name}'.",
                        suggestion="Add a docstring explaining the function's purpose.",
                        source="ast",
                    ))

    def _check_ts_magic_numbers(
        self,
        file_path: str,
        func,
        source_code: str,
        findings: List[Finding],
    ) -> None:
        """Report bare numeric literals that should be named constants.

        Args:
            file_path: Path used for labelling.
            func: Tree-sitter function node.
            source_code: Full source text (used to extract literal values).
            findings: List to append new Finding objects to.
        """
        if not self.settings.rules.magic_numbers.enabled:
            return

        def get_target_name(node) -> str | None:
            for field in ("name", "id", "left"):
                target = node.child_by_field_name(field)
                if target:
                    return source_code[target.start_byte:target.end_byte]
            if node.children:
                first = node.children[0]
                return source_code[first.start_byte:first.end_byte]
            return None

        def walk(node, parent_is_constant_assign: bool = False) -> None:
            is_constant_assign = parent_is_constant_assign

            if node.type in {"variable_declarator", "assignment_expression"}:
                name = get_target_name(node)
                if name and name.isupper():
                    is_constant_assign = True
                else:
                    is_constant_assign = False

            if node.type == "number":
                val_str = source_code[node.start_byte:node.end_byte]
                if val_str not in {"0", "1"} and not is_constant_assign:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=_get_ts_line(node),
                        severity="INFO",
                        category="style",
                        message=f"Magic number {val_str} found.",
                        suggestion="Extract this number into a named constant.",
                        source="ast",
                    ))

            for child in node.children:
                walk(child, is_constant_assign)

        walk(func, False)

    # ------------------------------------------------------------------
    # Function-level orchestrator
    # ------------------------------------------------------------------

    def _analyze_functions_treesitter(
        self,
        file_path: str,
        root_node,
        source_code: str,
        findings: List[Finding],
    ) -> None:
        """Collect all function nodes and run per-function rule checks.

        Args:
            file_path: Path used for labelling.
            root_node: Tree-sitter root node.
            source_code: Full source text.
            findings: List to append new Finding objects to.
        """
        funcs: list = []

        def find_funcs(node) -> None:
            if (
                "function" in node.type
                or "method" in node.type
                or node.type in {
                    "method_declaration",
                    "constructor_declaration",
                    "function_declaration",
                    "arrow_function",
                }
            ):
                funcs.append(node)
            for child in node.children:
                find_funcs(child)

        find_funcs(root_node)

        for func in funcs:
            self._check_ts_length(file_path, func, findings)
            self._check_ts_complexity(file_path, func, findings)
            self._check_ts_docs(file_path, func, source_code, findings)
            self._check_ts_magic_numbers(file_path, func, source_code, findings)
