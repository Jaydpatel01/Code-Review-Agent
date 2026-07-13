"""Unit tests for StaticAnalyzer — tree-sitter paths (JS and Java).

All tests use real fixture files and real tree-sitter parsing — no mocking.
Tree-sitter is deterministic, so the findings are stable across runs.
"""

import os
import pytest
from code_reviewer.analyzers.static_checks import StaticAnalyzer, HAVE_TREESITTER_LANGS
from code_reviewer.config import Settings

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_code")


def _read(filename: str) -> str:
    with open(os.path.join(FIXTURES, filename), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Skip entire module if tree-sitter language packages are not installed
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not HAVE_TREESITTER_LANGS,
    reason="tree-sitter language packages not installed",
)


# ---------------------------------------------------------------------------
# Shared settings factories
# ---------------------------------------------------------------------------

def _settings_all_enabled(max_cc: int = 10, max_len: int = 50, max_depth: int = 4) -> Settings:
    s = Settings()
    s.rules.complexity.enabled = True
    s.rules.nesting.enabled = True
    s.rules.docs.enabled = True
    s.rules.magic_numbers.enabled = True
    s.rules.complexity.max_cyclomatic_complexity = max_cc
    s.rules.complexity.max_function_length = max_len
    s.rules.nesting.max_nesting_depth = max_depth
    return s


def _settings_complexity_only() -> Settings:
    s = Settings()
    s.rules.complexity.enabled = True
    s.rules.nesting.enabled = True
    s.rules.docs.enabled = False
    s.rules.magic_numbers.enabled = False
    s.rules.complexity.max_cyclomatic_complexity = 10
    s.rules.complexity.max_function_length = 50
    s.rules.nesting.max_nesting_depth = 4
    return s


# ===========================================================================
# StaticAnalyzer routing
# ===========================================================================

class TestAnalyzeFileRouting:
    """Verify file-extension routing to the right analysis path."""

    def test_python_file_uses_ast_analyzer(self):
        """Python files must go through ASTAnalyzer, not tree-sitter."""
        s = _settings_all_enabled()
        analyzer = StaticAnalyzer(s)
        code = "def f(): pass\n"
        findings = analyzer.analyze_file("app.py", code)
        # All findings should have source='ast' (from ASTAnalyzer)
        for f in findings:
            assert f.source == "ast"

    def test_unsupported_extension_returns_empty(self):
        """Ruby, Go, and other unsupported extensions must return empty list."""
        s = _settings_all_enabled()
        analyzer = StaticAnalyzer(s)
        assert analyzer.analyze_file("script.rb", "puts 'hello'") == []
        assert analyzer.analyze_file("main.go", "package main") == []
        assert analyzer.analyze_file("style.css", "body { color: red; }") == []

    def test_js_file_routed_to_treesitter(self):
        """A .js file must hit the tree-sitter path and return findings."""
        s = _settings_complexity_only()
        analyzer = StaticAnalyzer(s)
        code = _read("bad_code.js")
        findings = analyzer.analyze_file("bad_code.js", code)
        assert len(findings) > 0
        for f in findings:
            assert f.source == "ast"  # tree-sitter also sets source='ast'

    def test_java_file_routed_to_treesitter(self):
        """A .java file must hit the tree-sitter path and return findings."""
        s = _settings_complexity_only()
        analyzer = StaticAnalyzer(s)
        code = _read("bad_code.java")
        findings = analyzer.analyze_file("bad_code.java", code)
        assert len(findings) > 0


# ===========================================================================
# JavaScript — bad_code.js
# ===========================================================================

class TestJavaScriptBadCode:
    """Verify all expected findings from bad_code.js."""

    def setup_method(self):
        self.code = _read("bad_code.js")
        self.settings = _settings_all_enabled()
        self.analyzer = StaticAnalyzer(self.settings)
        self.findings = self.analyzer.analyze_file("bad_code.js", self.code)

    def _by_category(self, category: str):
        return [f for f in self.findings if f.category == category]

    def _by_severity(self, severity: str):
        return [f for f in self.findings if f.severity == severity]

    # --- Nesting depth -------------------------------------------------------

    def test_deep_nesting_detected(self):
        """bad_code.js has 5+ levels of nesting — must produce complexity finding."""
        nesting = [f for f in self.findings
                   if f.category == "complexity" and "nesting" in f.message.lower()]
        assert len(nesting) >= 1

    def test_nesting_finding_has_medium_severity(self):
        """Depth 5 (> max 4, <= 6) → MEDIUM."""
        medium_nesting = [f for f in self.findings
                          if f.category == "complexity"
                          and "nesting" in f.message.lower()
                          and f.severity == "MEDIUM"]
        assert len(medium_nesting) >= 1

    # --- Cyclomatic complexity -----------------------------------------------

    def test_high_cc_detected(self):
        """highComplexity function has CC > 15 → HIGH severity."""
        high_cc = [f for f in self.findings
                   if f.category == "complexity"
                   and "cyclomatic" in f.message.lower()
                   and f.severity == "HIGH"]
        assert len(high_cc) >= 1

    def test_high_cc_message_contains_value(self):
        """CC finding message must include the numeric complexity value."""
        cc_findings = [f for f in self.findings
                       if f.category == "complexity" and "cyclomatic" in f.message.lower()]
        assert any(any(c.isdigit() for c in f.message) for f in cc_findings)

    # --- Function length -----------------------------------------------------

    def test_long_function_detected(self):
        """longFunction is > 50 lines → MEDIUM complexity finding."""
        long_fn = [f for f in self.findings
                   if f.category == "complexity" and "too long" in f.message.lower()]
        assert len(long_fn) >= 1
        assert long_fn[0].severity == "MEDIUM"

    # --- Magic numbers -------------------------------------------------------

    def test_magic_numbers_detected(self):
        """42 in useMagicNumber → INFO style finding."""
        magic = [f for f in self.findings if f.category == "style"]
        assert len(magic) >= 1
        assert all(f.severity == "INFO" for f in magic)

    # --- Docs ----------------------------------------------------------------

    def test_missing_docs_detected(self):
        """Functions without JSDoc → LOW docs finding."""
        docs = self._by_category("docs")
        assert len(docs) >= 1
        assert all(f.severity == "LOW" for f in docs)

    # --- Source field --------------------------------------------------------

    def test_all_findings_have_ast_source(self):
        """Every tree-sitter finding must have source='ast'."""
        for f in self.findings:
            assert f.source == "ast", f"Expected source='ast', got {f.source!r} for: {f.message}"

    # --- File path -----------------------------------------------------------

    def test_all_findings_have_correct_file_path(self):
        for f in self.findings:
            assert f.file_path == "bad_code.js"


# ===========================================================================
# JavaScript — rules disabled
# ===========================================================================

class TestJavaScriptRulesDisabled:
    """Disabling rules must suppress their corresponding findings."""

    def test_complexity_disabled_suppresses_cc_and_length(self):
        s = Settings()
        s.rules.complexity.enabled = False
        s.rules.nesting.enabled = False
        s.rules.docs.enabled = False
        s.rules.magic_numbers.enabled = False
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("bad_code.js", _read("bad_code.js"))
        assert findings == []

    def test_magic_numbers_disabled_suppresses_style_findings(self):
        s = _settings_all_enabled()
        s.rules.magic_numbers.enabled = False
        s.rules.docs.enabled = False
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("bad_code.js", _read("bad_code.js"))
        style = [f for f in findings if f.category == "style"]
        assert style == []

    def test_docs_disabled_suppresses_docs_findings(self):
        s = _settings_all_enabled()
        s.rules.docs.enabled = False
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("bad_code.js", _read("bad_code.js"))
        docs = [f for f in findings if f.category == "docs"]
        assert docs == []


# ===========================================================================
# Java — bad_code.java
# ===========================================================================

class TestJavaBadCode:
    """Verify all expected findings from bad_code.java."""

    def setup_method(self):
        self.code = _read("bad_code.java")
        self.settings = _settings_all_enabled()
        self.analyzer = StaticAnalyzer(self.settings)
        self.findings = self.analyzer.analyze_file("bad_code.java", self.code)

    def test_deep_nesting_detected(self):
        """bad_code.java has 5+ levels of nesting."""
        nesting = [f for f in self.findings
                   if f.category == "complexity" and "nesting" in f.message.lower()]
        assert len(nesting) >= 1

    def test_high_cc_detected(self):
        """highComplexity method has CC > 15 → HIGH."""
        high_cc = [f for f in self.findings
                   if f.category == "complexity"
                   and "cyclomatic" in f.message.lower()
                   and f.severity == "HIGH"]
        assert len(high_cc) >= 1

    def test_long_function_detected(self):
        """longMethod is > 50 lines → MEDIUM."""
        long_fn = [f for f in self.findings
                   if f.category == "complexity" and "too long" in f.message.lower()]
        assert len(long_fn) >= 1

    def test_missing_docs_detected(self):
        """Methods without JavaDoc → LOW docs."""
        docs = [f for f in self.findings if f.category == "docs"]
        assert len(docs) >= 1

    def test_all_findings_have_ast_source(self):
        for f in self.findings:
            assert f.source == "ast"

    def test_all_findings_have_correct_file_path(self):
        for f in self.findings:
            assert f.file_path == "bad_code.java"

    def test_line_numbers_are_positive(self):
        """All findings must have a positive line number."""
        for f in self.findings:
            assert f.line_number is not None
            assert f.line_number >= 1


# ===========================================================================
# Java — rules disabled
# ===========================================================================

class TestJavaRulesDisabled:
    def test_all_rules_disabled_returns_empty(self):
        s = Settings()
        s.rules.complexity.enabled = False
        s.rules.nesting.enabled = False
        s.rules.docs.enabled = False
        s.rules.magic_numbers.enabled = False
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("bad_code.java", _read("bad_code.java"))
        assert findings == []


# ===========================================================================
# Nesting threshold — extremely deep (> 6 levels) → HIGH
# ===========================================================================

class TestMediumCyclomaticComplexity:
    """Cover the MEDIUM-severity CC branch (cc > max_cc but <= 15)."""

    def test_js_medium_cc_detected(self):
        # CC of ~7 (> threshold 5, <= 15) → MEDIUM
        code = """\
function moderate(a, b, c) {
    if (a > 0) { }
    if (b > 0) { }
    if (c > 0) { }
    if (a && b) { }
    while (a > 0) { a--; }
    return a ? b : c;
}
"""
        s = _settings_all_enabled(max_cc=5)  # threshold=5, CC ~7 > 5 and <= 15 → MEDIUM
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("moderate.js", code)
        medium_cc = [f for f in findings
                     if f.category == "complexity"
                     and "cyclomatic" in f.message.lower()
                     and f.severity == "MEDIUM"]
        assert len(medium_cc) >= 1

    def test_java_medium_cc_detected(self):
        # CC of ~7 (> threshold 5, <= 15) → MEDIUM
        code = """\
public class M {
    public int moderate(int a, int b, int c) {
        if (a > 0) { }
        if (b > 0) { }
        if (c > 0) { }
        if (a > 1 && b > 1) { }
        while (a > 0) { a--; }
        return a > 0 ? b : c;
    }
}
"""
        s = _settings_all_enabled(max_cc=5)
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("moderate.java", code)
        medium_cc = [f for f in findings
                     if f.category == "complexity"
                     and "cyclomatic" in f.message.lower()
                     and f.severity == "MEDIUM"]
        assert len(medium_cc) >= 1

class TestExtremeNestingThreshold:
    """Verify the HIGH-severity path fires when depth > 6."""

    def test_js_extreme_nesting_produces_high_severity(self):
        # 7-level nesting: if inside if inside if... (7 deep)
        code = """\
function f(a) {
    if (a) {
        if (a) {
            if (a) {
                if (a) {
                    if (a) {
                        if (a) {
                            if (a) {
                                console.log("7 deep");
                            }
                        }
                    }
                }
            }
        }
    }
}
"""
        s = _settings_all_enabled(max_depth=4)
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("deep.js", code)
        high_nesting = [f for f in findings
                        if f.category == "complexity"
                        and f.severity == "HIGH"
                        and "nesting" in f.message.lower()]
        assert len(high_nesting) >= 1

    def test_java_extreme_nesting_produces_high_severity(self):
        code = """\
public class D {
    public void f(boolean a) {
        if (a) {
            if (a) {
                if (a) {
                    if (a) {
                        if (a) {
                            if (a) {
                                if (a) {
                                    System.out.println("7 deep");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""
        s = _settings_all_enabled(max_depth=4)
        analyzer = StaticAnalyzer(s)
        findings = analyzer.analyze_file("deep.java", code)
        high_nesting = [f for f in findings
                        if f.category == "complexity"
                        and f.severity == "HIGH"
                        and "nesting" in f.message.lower()]
        assert len(high_nesting) >= 1
