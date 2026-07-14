"""AST-based risk scoring for individual source files.

Classifies each file as HIGH / MEDIUM / LOW based on the severity of
findings produced by the ASTAnalyzer.  Used by the repo-level review
command to decide which files need LLM review.
"""

from __future__ import annotations

from pathlib import Path

from code_reviewer.analyzers.ast_analyzer import ASTAnalyzer
from code_reviewer.config import Settings
from code_reviewer.core.models import Finding


def score_file_risk(
    file_path: Path,
    settings: Settings,
) -> tuple[str, list[Finding]]:
    """Run AST analysis on *file_path* and return a (tier, findings) pair.

    Tier values
    -----------
    ``"HIGH"``
        At least one HIGH-severity finding.
    ``"MEDIUM"``
        At least one MEDIUM-severity finding (and no HIGH findings).
    ``"LOW"``
        No HIGH or MEDIUM findings, or analysis raised an exception.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the Python source file.
    settings:
        Project settings passed through to ASTAnalyzer.

    Returns
    -------
    tuple[str, list[Finding]]
        ``(tier, findings)`` where *tier* is one of the strings above and
        *findings* is the list returned by the analyzer (may be empty on
        error).
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        analyzer = ASTAnalyzer(str(file_path), settings)
        findings = analyzer.analyze(source)

        if any(f.severity == "HIGH" for f in findings):
            return "HIGH", findings
        if any(f.severity == "MEDIUM" for f in findings):
            return "MEDIUM", findings
        return "LOW", findings
    except Exception:
        return "LOW", []
