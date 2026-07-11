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
from code_reviewer.analyzers.ast_analyzer import ASTAnalyzer


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
        ext = os.path.splitext(file_path)[1].lower()
        
        # 1. Prefer pure Python AST for .py files due to higher precision
        if ext == ".py":
            analyzer = ASTAnalyzer(file_path=file_path, settings=self.settings)
            return analyzer.analyze(source_code)
            
        # 2. Fall back to Tree-sitter for non-Python languages
        lang = None
        if ext in [".js", ".ts", ".jsx", ".tsx"]:
            lang = "javascript"
        elif ext == ".java":
            lang = "java"
            
        if not lang or lang not in self.parsers:
            return [] # Unsupported language fallback
            
        return self._analyze_with_treesitter(file_path, source_code, lang)

    def _analyze_with_treesitter(self, file_path: str, source_code: str, lang_name: str) -> List[Finding]:
        parser = tree_sitter.Parser(self.parsers[lang_name])
        
        # Parse the raw bytes as required by tree-sitter
        tree = parser.parse(bytes(source_code, "utf8"))
        
        findings: List[Finding] = []
        
        # Generic recursive tree walk
        def walk(node, depth):
            # 1. Nesting Depth Check
            is_block = "block" in node.type or "statement_block" in node.type
            if is_block:
                depth += 1
                if self.settings.rules.nesting.enabled:
                    max_depth = self.settings.rules.nesting.max_nesting_depth
                    if depth > 6:
                        findings.append(Finding(
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            severity="HIGH",
                            category="complexity",
                            message=f"Extremely deep nesting ({depth} levels).",
                            suggestion="Extract nested blocks into separate functions.",
                            source="ast"
                        ))
                    elif depth > max_depth:
                        findings.append(Finding(
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            severity="MEDIUM",
                            category="complexity",
                            message=f"Deep nesting ({depth} levels).",
                            suggestion="Extract nested blocks into separate functions.",
                            source="ast"
                        ))

            for child in node.children:
                walk(child, depth)

        walk(tree.root_node, 0)
        
        # 2. Second pass to compute function-scoped rules safely
        self._analyze_functions_treesitter(file_path, tree.root_node, source_code, findings)
        
        return findings

    def _analyze_functions_treesitter(self, file_path: str, root_node, source_code: str, findings: List[Finding]):
        """Runs checks that are isolated within a function boundary."""
        funcs = []
        def find_funcs(node):
            if "function" in node.type or "method" in node.type or node.type in ["method_declaration", "constructor_declaration", "function_declaration", "arrow_function"]:
                funcs.append(node)
            for child in node.children:
                find_funcs(child)
                
        find_funcs(root_node)
        
        for func in funcs:
            # Check Function Length
            if self.settings.rules.complexity.enabled:
                length = func.end_point[0] - func.start_point[0]
                if length > self.settings.rules.complexity.max_function_length:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=func.start_point[0] + 1,
                        severity="MEDIUM",
                        category="complexity",
                        message=f"Function is too long ({length} lines).",
                        suggestion="Extract logic into smaller helper functions.",
                        source="ast"
                    ))

            # Check Cyclomatic Complexity
            if self.settings.rules.complexity.enabled:
                cc = 1
                def count_cc(node):
                    nonlocal cc
                    branch_nodes = ["if_statement", "for_statement", "while_statement", "catch_clause", "ternary_expression", "&&", "||"]
                    if node.type in branch_nodes:
                        cc += 1
                    if node.type == "binary_expression":
                        op = node.child_by_field_name("operator")
                        if op and op.type in ["&&", "||"]:
                            cc += 1
                            
                    for child in node.children:
                        # Do not leak complexity calculations into inner functions
                        if "function" not in child.type and "method" not in child.type:
                            count_cc(child)
                
                for child in func.children:
                    count_cc(child)
                    
                max_cc = self.settings.rules.complexity.max_cyclomatic_complexity
                if cc > 15:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=func.start_point[0] + 1,
                        severity="HIGH",
                        category="complexity",
                        message=f"Function has very high cyclomatic complexity ({cc}).",
                        suggestion="Refactor to simplify logic and reduce branching.",
                        source="ast"
                    ))
                elif cc > max_cc:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=func.start_point[0] + 1,
                        severity="MEDIUM",
                        category="complexity",
                        message=f"Function has high cyclomatic complexity ({cc}).",
                        suggestion="Refactor to simplify logic and reduce branching.",
                        source="ast"
                    ))
                
            # Check Missing Docstrings (JavaDoc/JSDoc)
            if self.settings.rules.docs.enabled:
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
                                line_number=func.start_point[0] + 1,
                                severity="LOW",
                                category="docs",
                                message=f"Missing docstring in public function '{name}'.",
                                suggestion="Add a docstring explaining the function's purpose.",
                                source="ast"
                            ))

            # Check Magic Numbers
            if self.settings.rules.magic_numbers.enabled:
                def find_bad_nodes(node, in_assign=False):
                    if node.type in ["assignment_expression", "variable_declarator"]:
                        in_assign = True
                    
                    if node.type == "number":
                        val_str = source_code[node.start_byte:node.end_byte]
                        if val_str not in ["0", "1"]:
                            if not in_assign:
                                findings.append(Finding(
                                    file_path=file_path,
                                    line_number=node.start_point[0] + 1,
                                    severity="INFO",
                                    category="style",
                                    message=f"Magic number {val_str} found.",
                                    suggestion="Extract this number into a named constant.",
                                    source="ast"
                                ))

                    for child in node.children:
                        find_bad_nodes(child, in_assign)
                
                find_bad_nodes(func)
