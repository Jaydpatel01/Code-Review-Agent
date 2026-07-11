"""Unit tests for ASTAnalyzer."""

import os
import pytest
from code_reviewer.analyzers.ast_analyzer import ASTAnalyzer
from code_reviewer.config import Settings

@pytest.fixture
def base_settings():
    """Returns Settings with all AST rules enabled for testing."""
    settings = Settings()
    settings.rules.complexity.enabled = True
    settings.rules.nesting.enabled = True
    settings.rules.mutable_defaults.enabled = True
    settings.rules.docs.enabled = True
    settings.rules.magic_numbers.enabled = True
    
    settings.rules.complexity.max_cyclomatic_complexity = 10
    settings.rules.complexity.max_function_length = 50
    settings.rules.nesting.max_nesting_depth = 4
    return settings

def test_ast_analyzer_bad_code(base_settings):
    """Test that ASTAnalyzer finds all expected issues in bad_code.py."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "sample_code", "bad_code.py"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        source_code = f.read()
        
    analyzer = ASTAnalyzer(fixture_path, base_settings)
    findings = analyzer.analyze(source_code)
    
    # We should find multiple violations
    assert len(findings) > 0
    
    categories = {f.category for f in findings}
    assert "complexity" in categories
    assert "docs" in categories
    assert "logic" in categories
    assert "style" in categories
    
    # Specific assertions based on known lines in bad_code.py
    
    # 1. Missing docstring in complex_function (line 1)
    doc_finding = next((f for f in findings if f.category == "docs" and "complex_function" in f.message), None)
    assert doc_finding is not None
    assert doc_finding.line_number == 1
    
    # 2. Deep nesting (line 7 is the print statement inside 5 ifs)
    # The AST node for 'if a < 10:' is at line 6, but the block inside starts there.
    nesting_finding = next((f for f in findings if "nesting" in f.message.lower()), None)
    assert nesting_finding is not None
    assert nesting_finding.severity == "MEDIUM" or nesting_finding.severity == "HIGH"
    
    # 3. Cyclomatic complexity > 10 (line 1)
    cc_finding = next((f for f in findings if "cyclomatic complexity" in f.message.lower()), None)
    assert cc_finding is not None
    assert cc_finding.line_number == 1
    
    # 4. Mutable default in mutable_defaults_func (line 26)
    mutable_finding = next((f for f in findings if "Mutable default" in f.message), None)
    assert mutable_finding is not None
    assert mutable_finding.line_number == 26
    
    # 5. Long function > 50 lines (line 31)
    long_finding = next((f for f in findings if "too long" in f.message.lower()), None)
    assert long_finding is not None
    assert long_finding.line_number == 31
    
    # 6. Magic number (line 87)
    magic_finding = next((f for f in findings if "Magic number 42" in f.message), None)
    assert magic_finding is not None
    assert magic_finding.line_number == 87
    
    # 7. Class missing docstring (line 93)
    class_doc_finding = next((f for f in findings if f.category == "docs" and "BadClass" in f.message), None)
    assert class_doc_finding is not None
    assert class_doc_finding.line_number == 93


def test_ast_analyzer_good_code(base_settings):
    """Test that ASTAnalyzer finds zero issues in good_code.py."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "sample_code", "good_code.py"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        source_code = f.read()
        
    analyzer = ASTAnalyzer(fixture_path, base_settings)
    findings = analyzer.analyze(source_code)
    
    assert len(findings) == 0
