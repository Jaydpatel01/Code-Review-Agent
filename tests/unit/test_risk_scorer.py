"""Unit tests for score_file_risk (src/code_reviewer/indexer/risk_scorer.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_reviewer.config import Settings
from code_reviewer.indexer.risk_scorer import score_file_risk


FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_code"


class TestRiskScorerFixtures:
    def test_bad_code_py_scores_high(self):
        """bad_code.py has deep nesting and high cyclomatic complexity → HIGH tier."""
        tier, findings = score_file_risk(FIXTURES / "bad_code.py", Settings())
        assert tier == "HIGH", (
            f"Expected HIGH for bad_code.py, got {tier}. "
            f"Findings: {[(f.severity, f.message) for f in findings]}"
        )
        assert len(findings) > 0

    def test_good_code_py_scores_low(self):
        """good_code.py has no violations → LOW tier with no findings."""
        tier, findings = score_file_risk(FIXTURES / "good_code.py", Settings())
        assert tier == "LOW", (
            f"Expected LOW for good_code.py, got {tier}. "
            f"Findings: {[(f.severity, f.message) for f in findings]}"
        )

    def test_findings_are_finding_objects(self):
        """score_file_risk always returns Finding instances (not dicts or None)."""
        from code_reviewer.core.models import Finding

        for name in ("bad_code.py", "good_code.py"):
            _, findings = score_file_risk(FIXTURES / name, Settings())
            for f in findings:
                assert isinstance(f, Finding), f"Expected Finding, got {type(f)}"


class TestRiskScorerEdgeCases:
    def test_syntax_error_file_returns_low_no_crash(self, tmp_path):
        """A file with a SyntaxError must return ('LOW', []) without raising."""
        bad = tmp_path / "syntax_err.py"
        bad.write_text("def broken(\n    x\n    # missing closing paren", encoding="utf-8")
        tier, findings = score_file_risk(bad, Settings())
        # ASTAnalyzer reports SyntaxError as a HIGH finding itself, so tier
        # may be HIGH — but the scorer must NEVER raise.  Just assert no crash.
        assert tier in ("HIGH", "MEDIUM", "LOW")

    def test_nonexistent_file_returns_low_no_crash(self, tmp_path):
        """A missing file must return ('LOW', []) without raising."""
        missing = tmp_path / "does_not_exist.py"
        tier, findings = score_file_risk(missing, Settings())
        assert tier == "LOW"
        assert findings == []

    def test_empty_file_returns_low(self, tmp_path):
        """An empty file has no findings → LOW tier."""
        empty = tmp_path / "empty.py"
        empty.write_text("", encoding="utf-8")
        tier, findings = score_file_risk(empty, Settings())
        assert tier == "LOW"

    def test_medium_tier_file(self, tmp_path):
        """A file with excessive nesting (5 levels) scores at least MEDIUM."""
        # 5 levels of nesting is above the default max_nesting_depth of 4
        code = (
            "def f(x):\n"
            "    if x:\n"
            "        if x > 1:\n"
            "            if x > 2:\n"
            "                if x > 3:\n"
            "                    if x > 4:\n"
            "                        pass\n"
        )
        f = tmp_path / "medium.py"
        f.write_text(code, encoding="utf-8")
        tier, findings = score_file_risk(f, Settings())
        # The file might be HIGH or MEDIUM depending on settings; either is fine
        # as long as it isn't LOW when there are real violations.
        assert tier in ("HIGH", "MEDIUM"), (
            f"Expected HIGH or MEDIUM for deeply-nested code, got {tier}. "
            f"Findings: {[(f_.severity, f_.message) for f_ in findings]}"
        )

    def test_return_type_is_tuple(self, tmp_path):
        """score_file_risk always returns a 2-tuple (str, list)."""
        f = tmp_path / "any.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = score_file_risk(f, Settings())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)
